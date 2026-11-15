"""FastAPI app. Run from backend/:  uvicorn app.main:app --port 8000"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from astropy.time import Time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import assistant, fetcher
from .cache import FRAMES_DIR, disk_cache
from .config import settings
from .jobs import gaia_radius, json_safe, manager, prune_frames
from .overlays.gaia import alignment, gaia_sources, positions_at, to_display
from .overlays.wise import wise_sources
from .processing import measure_batch, reproject_only
from .products import load_stack, render_diff_any
from .render import apply_stretch, gif_bytes, png_bytes, stretch_params
from .processing import target_wcs
from .metadata import collections, query_images
from .resolver import resolve as resolve_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start_pool()
    removed = await asyncio.to_thread(prune_frames, settings.frames_max_age_days)
    logging.getLogger(__name__).info("startup: pruned %d old frame folders", removed)
    yield
    manager.shutdown()
    await fetcher.http.aclose()


app = FastAPI(title="SPHEREx Sky-Motion Viewer API", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)

RA = Query(..., ge=0, lt=360)
DEC = Query(..., ge=-90, le=90)


class JobRequest(BaseModel):
    ra: float = Field(..., ge=0, lt=360)
    dec: float = Field(..., ge=-90, le=90)
    fov: float = Field(settings.fov_default_deg, ge=settings.fov_min_deg, le=settings.fov_max_deg)
    detectors: list[int] = Field(default_factory=list)


@app.get("/api/health")
async def health():
    running = sum(1 for j in manager.jobs.values() if j.status == "running")
    return {"ok": True, "collections": collections(), "data_source": settings.data_source,
            "ai_enabled": bool(settings.gemini_api_key), "jobs_running": running, "jobs_in_memory": len(manager.jobs),
            "cache_bytes": disk_cache.volume(), "stats": dict(fetcher.STATS)}


def parse_coords(text: str):
    """'133.795 -7.245', '08h55m10.83s -07d14m42.5s', '08:55:10.83 -07:14:42.5' -> deg.
    Returns None when the text isn't coordinates (then it's treated as a name)."""
    import re

    t = text.strip().replace(",", " ")
    parts = t.split()
    try:
        if len(parts) == 2 and all(re.fullmatch(r"[+-]?\d+(\.\d*)?", x) for x in parts):
            ra, dec = float(parts[0]), float(parts[1])
        elif re.search(r"\d[hH:]|\d\s+\d", t):
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(t, unit=(u.hourangle, u.deg))
            ra, dec = float(c.ra.deg), float(c.dec.deg)
        else:
            return None
    except Exception:
        return None
    if 0 <= ra < 360 and -90 <= dec <= 90:
        return ra, dec
    return None


@app.get("/api/resolve")
async def resolve(name: str = Query(..., min_length=1, max_length=100)):
    """A name ('Barnard's Star', resolved with SIMBAD and moved by proper motion to the
    SPHEREx epoch) or coordinates in decimal degrees or sexagesimal."""
    c = parse_coords(name)
    if c:
        return {"name": name, "main_id": "coordinates", "ra": c[0], "dec": c[1],
                "ra_cat": c[0], "dec_cat": c[1], "pmra": None, "pmdec": None,
                "epoch": None, "source": "coordinates"}
    try:
        return await asyncio.to_thread(resolve_name, name.strip())
    except Exception as e:
        raise HTTPException(404, f"could not resolve {name!r}: {e}")


@app.get("/api/query")
async def query(ra: float = RA, dec: float = DEC):
    rows = await query_images(ra, dec)
    return {"n": len(rows), "rows": rows}


@app.post("/api/jobs")
async def create_job(req: JobRequest):
    if any(d not in range(1, 7) for d in req.detectors):
        raise HTTPException(422, "detectors must be 1-6")
    job = manager.submit(req.ra, req.dec, req.fov, req.detectors)
    return {"job_id": job.id, "status": job.status}


def _job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _job(job_id)
    return {"job_id": job.id, "status": job.status, "params": job.params,
            "manifest": json_safe(job.manifest)}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = _job(job_id)

    async def gen():
        async for ev in job.stream():
            yield {"event": ev["type"], "data": json.dumps(json_safe(ev), allow_nan=False)}

    return EventSourceResponse(gen())


async def _wait_manifest(job_id: str) -> dict:
    job = _job(job_id)
    if job.task and not job.task.done():
        await asyncio.shield(job.task)
    if job.manifest is None:
        raise HTTPException(500, "job failed")
    return job.manifest


def _same_pos(p: dict, ra, dec) -> bool:
    return ra is None or dec is None or (abs(ra - p["ra"]) * 3600 < 1.5 and abs(dec - p["dec"]) * 3600 < 1.5)


async def _photometry(job_id: str, ra: float | None, dec: float | None) -> tuple[dict, list]:
    """Photometry at the target (from the manifest) or at ANY position in the field,
    measured on the job's cached cutouts: no new downloads, a few seconds at most."""
    m = await _wait_manifest(job_id)
    p = m["params"]
    if _same_pos(p, ra, dec):
        return m, m["photometry"]
    key = ("phot3", job_id, round(ra, 5), round(dec, 5))
    if (hit := disk_cache.get(key)) is not None:
        return m, hit
    exps = m.get("exposures", [])
    items = [(e["uid"], disk_cache.get(e["ckey"])) for e in exps]
    items = [(u, d) for u, d in items if d is not None]
    visit_of = {e["uid"]: e["visit"] for e in exps}
    chunks = [items[i:i + 25] for i in range(0, len(items), 25)]
    res = await asyncio.gather(*[manager.run_cpu(measure_batch, c, ra, dec, list(settings.mask_flags))
                                 for c in chunks])
    pts = [dict(x, visit=visit_of.get(x["uid"])) for part in res for x in part]
    pts = json_safe(sorted(pts, key=lambda r: r["wavelength_um"] or 0))
    disk_cache.set(key, pts, expire=settings.metadata_ttl_s)
    return m, pts


async def _blends(m: dict, ra: float, dec: float, radius_arcsec: float = 15.0) -> list[dict]:
    """Catalogued stars close enough to fall in the 12.4" photometry aperture (SPHEREx
    pixels are 6.2"): their light is mixed into the spectrum. Found on WISE 0855, which in
    2025-26 sits 7.9" from a G=16.3 star that dominates its short-wavelength flux.
    Uses only cached catalogues (never waits on a slow service)."""
    import math as _m
    p = m["params"]
    epoch = float(Time(m["visits"][-1]["t_end"], format="mjd").jyear) if m["visits"] else 2026.0
    out = []
    for name, key, fn_key in (("Gaia", "phot_g_mean_mag", "gaia"), ("WISE", "w1", "wise")):
        rad = gaia_radius(p["fov"])
        from .cache import tile_key
        ck = ("gaia", tile_key(p["ra"], p["dec"], arcmin=0.05), round(rad, 4), settings.gaia_gmax)             if fn_key == "gaia" else ("wise", tile_key(p["ra"], p["dec"], arcmin=0.05), round(rad, 4))
        cat = disk_cache.get(ck)
        if not cat:
            continue
        pos = positions_at(cat["sources"], [epoch], cat["epoch"])[:, 0, :]
        cosd = _m.cos(_m.radians(dec))
        d = np.hypot((pos[:, 0] - ra) * cosd, pos[:, 1] - dec) * 3600
        for i in np.argsort(d):
            if d[i] > radius_arcsec:
                break
            if d[i] < 1.0:  # that's the object itself
                continue
            out.append({"catalog": name, "id": str(cat["sources"][i].get("source_id") or cat["sources"][i].get("id")),
                        "sep_arcsec": round(float(d[i]), 1), "mag": cat["sources"][i].get(key)})
    return json_safe(out)


@app.get("/api/jobs/{job_id}/spectrum")
async def job_spectrum(job_id: str, ra: float | None = Query(None, ge=0, lt=360),
                       dec: float | None = Query(None, ge=-90, le=90)):
    """Spectrum (flux vs wavelength) at the target, or at any clicked position (ra, dec)."""
    m, phot = await _photometry(job_id, ra, dec)
    pts = [p for p in phot if p["wavelength_um"] and p["flux_mjy"] is not None]
    blends = await _blends(m, ra if ra is not None else m["params"]["ra"],
                           dec if dec is not None else m["params"]["dec"])
    return {"job_id": job_id, "ra": ra or m["params"]["ra"], "dec": dec or m["params"]["dec"],
            "blends": blends,
            "saturated": [p for p in phot if p.get("flag") == "core_masked" and p.get("wings_rel") is not None],
            "units": {"wavelength": "um", "flux": "mJy"},
            "note": "aperture r=2 px (12.4\"), local sigma-clipped background, no aperture correction",
            "points": pts, "n_masked": sum(1 for p in phot if p.get("flag"))}


@app.get("/api/jobs/{job_id}/lightcurve")
async def job_lightcurve(job_id: str, detector: int | None = Query(None, ge=1, le=6),
                         ra: float | None = Query(None, ge=0, lt=360),
                         dec: float | None = Query(None, ge=-90, le=90)):
    m, phot = await _photometry(job_id, ra, dec)
    pts = [p for p in phot if (p["flux_mjy"] is not None or p.get("wings_rel") is not None)
           and (detector is None or p["detector"] == detector)]
    return {"job_id": job_id, "points": sorted(pts, key=lambda p: p["t_mid"]),
            "note": "wings_rel: saturated star, relative brightness from the PSF wings "
                    "(compare within one detector only)"}


@app.get("/api/spectrum")
async def spectrum(ra: float = RA, dec: float = DEC,
                   fov: float = Query(settings.fov_default_deg, ge=settings.fov_min_deg,
                                      le=settings.fov_max_deg)):
    job = manager.submit(ra, dec, fov, [])
    return await job_spectrum(job.id)


@app.get("/api/jobs/{job_id}/exposures")
async def job_exposures(job_id: str):
    """Every exposure: date, detector, wavelength at the target, contamination flags near
    the target (fraction of pixels per flag plane), flags inside the aperture, flux."""
    m = await _wait_manifest(job_id)
    keep = ("i", "uid", "visit", "date", "t_mid", "detector", "wavelength_um", "bandwidth_um",
            "flags", "frac_masked", "flux_mjy", "phot_flag", "collection")
    return {"job_id": job_id, "flag_mask": settings.mask_flags, "flags_mode": settings.flags_mode,
            "exposures": [{k: e.get(k) for k in keep} for e in m.get("exposures", [])],
            "visits": [{"index": v["index"], "date_start": v["date_start"], "date_end": v["date_end"],
                        "n": v["n"], "n_ok": v["n_ok"], "flags_mean": v.get("flags_mean", {})}
                       for v in m["visits"]]}


def _png_response(data: bytes) -> Response:
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/diff.png")
async def job_diff_any(job_id: str, a: str = Query(..., pattern=r"^(\d+|deep)$"),
                       b: str = Query(..., pattern=r"^(\d+|deep)$"),
                       band: str = Query("all", pattern=r"^(all|D[1-6])$")):
    """Subtract ANY two visits (or 'deep'), e.g. a=2&b=0: white = brighter in a."""
    m = await _wait_manifest(job_id)
    path = FRAMES_DIR / job_id / f"x_{a}_{b}_{band}.png"
    if not path.exists():
        u8 = await asyncio.to_thread(render_diff_any, FRAMES_DIR / job_id,
                                     int(a) if a.isdigit() else a, int(b) if b.isdigit() else b,
                                     band, (m.get("dlim") or {}).get(band))
        if u8 is None:
            raise HTTPException(404, "no stack for that visit/band")
        await asyncio.to_thread(path.write_bytes, png_bytes(u8))
    return _png_response(await asyncio.to_thread(path.read_bytes))


@app.get("/api/jobs/{job_id}/exposure/{i}.png")
async def job_exposure_png(job_id: str, i: int):
    """One single exposure on the job grid, same stretch as its detector's stacks."""
    m = await _wait_manifest(job_id)
    exps = m.get("exposures", [])
    if not 0 <= i < len(exps):
        raise HTTPException(404, "no such exposure")
    path = FRAMES_DIR / job_id / f"exp_{i}.png"
    if not path.exists():
        e = exps[i]
        data = disk_cache.get(e["ckey"])
        if data is None:
            raise HTTPException(410, "cutout no longer cached; reload the field")
        p = m["params"]
        w, shape = target_wcs(p["ra"], p["dec"], p["fov"], settings.pixel_arcsec)
        arr = await manager.run_cpu(reproject_only, data, w.to_header().tostring(), shape,
                                    list(settings.mask_flags))
        st = (m.get("stretch") or {}).get(f"D{e['detector']}") or stretch_params(arr)
        await asyncio.to_thread(path.write_bytes, png_bytes(apply_stretch(arr, st)))
    return _png_response(await asyncio.to_thread(path.read_bytes))


@app.get("/api/jobs/{job_id}/flipbook.gif")
async def job_gif(job_id: str, band: str = Query("all", pattern=r"^(all|D[1-6]|rgb)$"),
                  mode: str = Query("stack", pattern=r"^(stack|diff)$"),
                  ms: int = Query(700, ge=100, le=5000)):
    """Animated GIF of the flipbook (for slides, posts, Backyard-Worlds-style sharing)."""
    m = await _wait_manifest(job_id)
    paths, labels = [], []
    for v in m["visits"]:
        url = (v.get("bands", {}).get(band) or {}).get(mode)
        if url:
            paths.append(FRAMES_DIR / job_id / url.rsplit("/", 1)[-1])
            labels.append(f"{v['date_start']} {band} {mode}")
    if not paths:
        raise HTTPException(404, "no frames for that band/mode")
    data = await asyncio.to_thread(gif_bytes, paths, labels, ms)
    return Response(data, media_type="image/gif", headers={
        "Content-Disposition": f'inline; filename="spherex_{job_id}_{band}_{mode}.gif"'})


async def _catalog(job_id: str, which: str):
    m = json_safe(await _wait_manifest(job_id))
    p = m["params"]
    try:
        if which == "gaia":
            cat = await gaia_sources(p["ra"], p["dec"], gaia_radius(p["fov"]))
            mag_key, ref_epoch = "phot_g_mean_mag", cat["epoch"]
        else:
            cat = await wise_sources(p["ra"], p["dec"], gaia_radius(p["fov"]))
            mag_key, ref_epoch = "w1", cat["epoch"]
    except Exception as e:
        raise HTTPException(502, f"{which} unavailable: {e}")
    out_wcs, shape = target_wcs(p["ra"], p["dec"], p["fov"], settings.pixel_arcsec)
    n = shape[0]
    epochs = [float(Time(0.5 * (v["t_start"] + v["t_end"]), format="mjd").jyear)
              for v in m["visits"]]
    srcs = cat["sources"]
    base = {"catalog": which, "provider": cat["provider"], "catalog_epoch": ref_epoch,
            "epochs": epochs, "pixel_arcsec": settings.pixel_arcsec, "n_pix": n}
    if not srcs or not epochs:
        return {**base, "sources": [], "alignment": {"n": 0, "reliable": False}}
    radec = positions_at(srcs, epochs, ref_epoch)
    disp = to_display(out_wcs, n, radec)
    inside = ((disp > -0.02) & (disp < 1.02)).all(axis=2).any(axis=1)
    mags = np.array([s.get(mag_key) if s.get(mag_key) is not None else np.nan for s in srcs])
    align = {"n": 0, "reliable": False}
    if which == "gaia":
        first = next((v["index"] for v in m["visits"] if v.get("stack")), None)
        if first is not None:
            stack = await asyncio.to_thread(load_stack, FRAMES_DIR / job_id, first, "all")
            if stack is not None:
                align = alignment(stack, out_wcs, radec[:, first, :], mags)
                align["visit"] = first
    sources = []
    for i, s in enumerate(srcs):
        if not inside[i]:
            continue
        d = {"id": s.get("source_id") or s.get("id"), "mag": s.get(mag_key),
             "pmra": s.get("pmra"), "pmdec": s.get("pmdec"), "ra": s["ra"], "dec": s["dec"],
             "xy": np.round(disp[i], 4).tolist()}
        if which == "gaia":
            d["g"], d["parallax"] = s.get("phot_g_mean_mag"), s.get("parallax")
        else:
            d["w1"], d["w2"] = s.get("w1"), s.get("w2")
        sources.append(d)
    return json_safe({**base, "sources": sources, "alignment": align})


@app.get("/api/jobs/{job_id}/gaia")
async def job_gaia(job_id: str):
    """Gaia DR3 stars at each visit's epoch (display fractions) + alignment check."""
    return await _catalog(job_id, "gaia")


@app.get("/api/jobs/{job_id}/wise")
async def job_wise(job_id: str):
    """CatWISE2020 / AllWISE sources in the field (3.4 and 4.6 um, like SPHEREx D4-D6)."""
    return await _catalog(job_id, "wise")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list, max_length=16)
    view: dict = Field(default_factory=dict)


@app.post("/api/jobs/{job_id}/ask")
async def job_ask(job_id: str, req: AskRequest, request: Request):
    """AI guide: answer a question about this job's results (Gemini, streamed as SSE).
    The model sees a factual summary of the job, the user's current view and the frame
    on screen. The API key stays on the server."""
    if not settings.gemini_api_key:
        raise HTTPException(503, "AI guide is not configured on this server")
    ok, msg = assistant.allow(request.client.host if request.client else "?")
    if not ok:
        raise HTTPException(429, msg)
    m = json_safe(await _wait_manifest(job_id))
    view = req.view or {}

    async def cached(which):  # never make the user wait on a slow catalogue service
        try:
            return await asyncio.wait_for(_catalog(job_id, which), 4)
        except Exception:
            return None

    gaia, wise = await asyncio.gather(cached("gaia"), cached("wise"))
    pick_points, pos = None, (m["params"]["ra"], m["params"]["dec"])
    pick = view.get("pick") if isinstance(view.get("pick"), dict) else None
    if pick and isinstance(pick.get("ra"), (int, float)) and isinstance(pick.get("dec"), (int, float))             and 0 <= pick["ra"] < 360 and -90 <= pick["dec"] <= 90:
        try:
            _, pick_points = await asyncio.wait_for(_photometry(job_id, pick["ra"], pick["dec"]), 30)
            pos = (pick["ra"], pick["dec"])
        except Exception:
            pick_points = None
    blends = await _blends(m, *pos)
    target_info = None
    name = str(view.get("name") or "").strip()[:100]
    if name and not parse_coords(name):
        try:  # cached forever after the first lookup, so normally instant
            info = await asyncio.wait_for(asyncio.to_thread(resolve_name, name), 8)
            # only if the name actually refers to this job's field
            if abs(info["ra"] - m["params"]["ra"]) * 3600 < 30 and abs(info["dec"] - m["params"]["dec"]) * 3600 < 30:
                target_info = info
        except Exception:
            target_info = None
    context = assistant.build_context(m, view, gaia, wise, pick_points, blends, target_info)
    f = assistant.frame_file(view.get("frame_src"), job_id, FRAMES_DIR)
    image = await asyncio.to_thread(f.read_bytes) if f else None
    question = req.question.strip()

    async def gen():
        async for ev in assistant.stream_answer(question, req.history, context, image):
            yield "data: " + json.dumps(ev) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/frame/{key}/{name}.png")
async def frame(key: str, name: str):
    if not (key.isalnum() and name.replace("_", "").isalnum()):
        raise HTTPException(400, "bad frame name")
    path = FRAMES_DIR / key / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "frame not rendered (yet)")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui/")


if FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND, html=True), name="ui")
