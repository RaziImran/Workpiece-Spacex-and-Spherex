"""AI guide: explains what the viewer is showing, using Google Gemini.

The browser never sees the API key: it posts a question to /api/jobs/{id}/ask, and this
module builds a compact, factual summary of the job (target, visits, per-detector
photometry, light-curve ranges, saturation and blend warnings, Gaia/WISE highlights,
contamination flags, the user's current view) plus the frame on screen, and streams
Gemini's answer back.

Model choice (tested 2026-11-15 with this key): gemini-3.5-flash and gemini-2.5-flash
answer with first text in ~2 s at low thinking; the "gemini-flash-latest" alias returned
503 (overloaded) and, with default thinking, took > 60 s. Models are tried in order.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import statistics
import time
from collections import defaultdict, deque

import httpx

from .config import settings

log = logging.getLogger(__name__)
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

SYSTEM = """You are the built-in guide of the "SPHEREx Sky-Motion Viewer", a website that shows
NASA SPHEREx infrared images of one patch of sky over time. Your job: help curious members of
the public, students and amateur astronomers understand what the tool is showing them and
what it might mean. Be friendly, clear and concise (usually 80-200 words; use short
paragraphs or bullet points). Explain jargon the first time you use it.

GROUND RULES
- Base statements about THIS field on the DATA SUMMARY and the attached image. Quote the
  numbers you use. If something is not in the data, say so rather than guessing.
- You may add general astronomy knowledge (e.g. what a brown dwarf or a Mira variable is,
  or published facts about a well-known named star), but label it as background knowledge.
- Never claim a discovery. If something looks unusual, suggest how the user could check it
  with the tool (Difference view, a different detector band, the Gaia/WISE overlays, the
  sky map, WiseView) and say professionals would need to confirm it.
- Point out caveats that are in the data: saturation, blending with nearby stars, few
  images, masked pixels, an unreliable alignment check.
- Use plain text with simple markdown only (**bold**, bullet lists). No tables, no code,
  no LaTeX or $...$ maths: write numbers and units plainly (e.g. G = 8.2, 10.4"/yr).

HOW THE TOOL WORKS (use this to explain)
- SPHEREx surveys the whole sky every ~6 months in 102 infrared colours from 0.75 to 5.0 um,
  using 6 detectors (D1 0.75-1.09, D2 1.10-1.62, D3 1.63-2.41, D4 2.42-3.82, D5 3.83-4.41,
  D6 4.42-5.00 um). Each detector has a filter whose wavelength changes across it, so every
  exposure sees a star at a slightly different wavelength. Collecting them gives a spectrum.
- A "visit" groups the exposures from one ~6-month pass. "Stack" = median of a visit's
  images. "Difference" = a visit minus the first visit: unchanged stars cancel to grey; a
  moving star leaves a black (old position) / white (new position) pair; a star that got
  brighter turns white, fainter turns black. "A - B" subtracts any two visits. "Deep"
  averages all visits (shows fainter things). "Blink" flips between two visits.
- Bands: "all" stacks every detector; D1..D6 one detector each; "RGB" maps D1+D2 to blue,
  D3+D4 to green, D5+D6 to red, so cold objects like brown dwarfs look red and hot stars
  look white/blue.
- SPHEREx pixels are 6.2 arcseconds; the photometry aperture has a 2-pixel radius (12.4").
  Nearby stars within ~15" blend into the measurement. Fluxes are in millijanskys (mJy),
  aperture sums without aperture correction (good for comparing, not exact totals).
- Very bright stars saturate SPHEREx: their core is masked; the light curve then uses the
  unsaturated PSF wings ("wings_rel", relative units, compare only within one detector).
- Overlays: green circles = Gaia DR3 stars at each visit's date (orange = fast proper
  motion, arrow = where it will be in 10 years); purple = WISE sources (W1 3.4 um,
  W2 4.6 um; W1-W2 > 1 means very red, often a brown dwarf). The yellow cross marks the
  target; a pink circle marks a star the user clicked.
- The Gaia alignment check measures how far our frames are from Gaia positions; well under
  1 pixel means the images are aligned and apparent motion is real.
- Contamination flags are SPHEREx quality bits per pixel. SOURCE only means "a known star is
  here" (not a problem). TRANSIENT/OUTLIER often mean cosmic rays; HALO/GHOST mean light
  from a bright nearby star; SATURATION-like flags (OVERFLOW, SUR_ERROR, NONLINEAR) mean a
  star is too bright; NONFUNC/HOT/COLD are bad detector pixels.
- Names are moved by their proper motion to the SPHEREx date (2025.85) before searching.
"""

# ------------------------------------------------------------------ rate limiting
_hits: dict[str, deque] = defaultdict(deque)
_global: deque = deque()


def allow(client: str) -> tuple[bool, str]:
    """Simple in-memory limits so a public deployment can't drain the API key."""
    now = time.time()
    q = _hits[client]
    while q and now - q[0] > 600:
        q.popleft()
    while _global and now - _global[0] > 3600:
        _global.popleft()
    if len(q) >= settings.ai_per_client_per_10min:
        return False, "Too many questions from you in the last 10 minutes. Please wait a bit."
    if len(_global) >= settings.ai_global_per_hour:
        return False, "The AI guide is busy right now. Please try again later."
    q.append(now)
    _global.append(now)
    return True, ""


# ------------------------------------------------------------------ context
def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(x, n)


def _phot_summary(points: list[dict]) -> dict:
    """Per-detector spectrum and light-curve summary from photometry points."""
    by = defaultdict(list)
    for p in points:
        by[p["detector"]].append(p)
    out = {}
    for d in sorted(by):
        ps = by[d]
        good = [p for p in ps if p.get("flux_mjy") is not None]
        wings = [p for p in ps if p.get("flux_mjy") is None and p.get("wings_rel") is not None]
        waves = [p["wavelength_um"] for p in ps if p.get("wavelength_um")]
        e = {"n": len(ps), "wavelength_um": [_r(min(waves)), _r(max(waves))] if waves else None}
        if good:
            f = [p["flux_mjy"] for p in good]
            e.update(flux_mjy_median=_r(statistics.median(f)), flux_mjy_min=_r(min(f)),
                     flux_mjy_max=_r(max(f)),
                     typical_error_mjy=_r(statistics.median([p["err_mjy"] or 0 for p in good])))
        if wings:
            w = [p["wings_rel"] for p in wings]
            e.update(saturated_n=len(wings), wings_rel_min=_r(min(w), 1), wings_rel_max=_r(max(w), 1))
        out[f"D{d}"] = e
    return out


def target_motion(info: dict | None, m: dict) -> dict | None:
    """Catalogue facts about the named target, incl. how far it should move in our data.
    (WISE 0855 is too faint for Gaia, so without this the guide missed its 8"/yr motion.)"""
    if not info or info.get("source") == "coordinates":
        return None
    out = {"catalogue_name": info.get("main_id"), "resolved_by": info.get("source")}
    pmra, pmdec = info.get("pmra"), info.get("pmdec")
    if pmra is not None and pmdec is not None:
        pm = math.hypot(pmra, pmdec)
        out["proper_motion_mas_per_yr"] = [_r(pmra, 1), _r(pmdec, 1)]
        out["total_proper_motion_arcsec_per_yr"] = _r(pm / 1000, 2)
        vs = m.get("visits") or []
        if len(vs) >= 2:
            years = (vs[-1]["t_end"] - vs[0]["t_start"]) / 365.25
            out["expected_shift_first_to_last_visit"] = {
                "years": _r(years, 2), "arcsec": _r(pm / 1000 * years, 1),
                "pixels": _r(pm / 1000 * years / settings.pixel_arcsec, 2)}
    if info.get("parallax"):
        out["parallax_mas"] = _r(info["parallax"], 1)
        out["distance_light_years"] = _r(3261.56 / info["parallax"], 1) if info["parallax"] > 0 else None
    return out


def build_context(m: dict, view: dict, gaia: dict | None, wise: dict | None,
                  pick_points: list[dict] | None, blends: list[dict],
                  target_info: dict | None = None) -> str:
    p = m["params"]
    visits = [{"visit": v["index"] + 1, "dates": f"{v['date_start']} to {v['date_end']}",
               "images": f"{v['n_ok']}/{v['n']}",
               "flags_near_target": dict(list(sorted((v.get("flags_mean") or {}).items(),
                                                     key=lambda kv: -kv[1]))[:5])}
              for v in m["visits"]]
    ctx = {
        "target": {"name": view.get("name") or None, "ra_deg": _r(p["ra"], 5),
                   "dec_deg": _r(p["dec"], 5), "field_of_view_arcmin": _r(p["fov"] * 60, 1),
                   "catalogue": target_motion(target_info, m)},
        "data": {"images_used": m.get("n_images"), "failed_images": m.get("n_failed"),
                 "visits": visits, "bands_available": m.get("bands")},
        "target_photometry_by_detector": _phot_summary(m.get("photometry", [])),
        "saturated_measurements_at_target": m.get("n_phot_masked", 0),
        "blends_near_target": blends,
    }
    if gaia:
        fast = [s for s in gaia.get("sources", []) if math.hypot(s.get("pmra") or 0, s.get("pmdec") or 0) > 100]
        ctx["gaia"] = {"n_stars_in_field": len(gaia.get("sources", [])),
                       "alignment_check": gaia.get("alignment"),
                       "fast_movers": [{"id": s["id"], "G": _r(s.get("g") or s.get("mag"), 2),
                                        "pm_mas_per_yr": [_r(s.get("pmra"), 0), _r(s.get("pmdec"), 0)],
                                        "parallax_mas": _r(s.get("parallax"), 1),
                                        "offset_from_target_px": _r(math.hypot(s["xy"][-1][0] - 0.5, s["xy"][-1][1] - 0.5) * gaia.get("n_pix", 1), 1)}
                                       for s in fast[:5]]}
    if wise:
        red = [s for s in wise.get("sources", []) if s.get("w1") is not None and s.get("w2") is not None
               and s["w1"] - s["w2"] > 0.8]
        ctx["wise"] = {"n_sources": len(wise.get("sources", [])), "provider": wise.get("provider"),
                       "very_red_sources_W1_minus_W2_gt_0.8": [
                           {"id": s["id"], "W1": _r(s["w1"], 2), "W2": _r(s["w2"], 2),
                            "offset_from_target_px": _r(math.hypot(s["xy"][-1][0] - 0.5, s["xy"][-1][1] - 0.5) * wise.get("n_pix", 1), 1)}
                           for s in red[:5]]}
    v = {k: view.get(k) for k in ("mode", "band", "frame_label") if view.get(k)}
    if view.get("pick"):
        v["clicked_position"] = {"ra_deg": _r(view["pick"].get("ra"), 5), "dec_deg": _r(view["pick"].get("dec"), 5)}
        if pick_points is not None:
            v["clicked_position_photometry_by_detector"] = _phot_summary(pick_points)
    ctx["current_view"] = v
    return json.dumps(ctx, separators=(",", ":"), default=str)


# ------------------------------------------------------------------ Gemini call
def _gen_config(model: str) -> dict:
    cfg = {"maxOutputTokens": settings.ai_max_output_tokens, "temperature": 0.4}
    if model.startswith("gemini-2.5"):
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        cfg["thinkingConfig"] = {"thinkingLevel": "low"}
    return cfg


def build_request(question: str, history: list[dict], context: str, image: bytes | None,
                  model: str) -> dict:
    contents = []
    for h in history[-8:]:
        role = "model" if h.get("role") == "model" else "user"
        text = str(h.get("text", ""))[:4000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    parts = [{"text": f"DATA SUMMARY (JSON, computed by the tool):\n{context}\n\nUSER QUESTION:\n{question}"}]
    if image:
        parts.append({"text": "The image currently shown in the viewer (north up, east left; "
                              "the yellow cross is the target at the centre):"})
        parts.append({"inline_data": {"mime_type": "image/png",
                                      "data": base64.b64encode(image).decode()}})
    contents.append({"role": "user", "parts": parts})
    return {"system_instruction": {"parts": [{"text": SYSTEM}]}, "contents": contents,
            "generationConfig": _gen_config(model)}


async def stream_answer(question: str, history: list[dict], context: str, image: bytes | None,
                        client: httpx.AsyncClient | None = None):
    """Async generator of events: {"type": "delta", "text"}, {"type": "done", "model"},
    {"type": "error", "message"}. Falls back to the next model on 429/5xx/timeout."""
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not key:
        yield {"type": "error", "message": "AI guide is not configured (no GEMINI_API_KEY)."}
        return
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.ai_timeout_s, connect=15))
    last = ""
    try:
        for model in settings.ai_models:
            body = build_request(question, history, context, image, model)
            sent = False
            try:
                async with client.stream("POST", API.format(model=model), params={"alt": "sse"},
                                         headers={"x-goog-api-key": key}, json=body) as r:
                    if r.status_code != 200:
                        err = (await r.aread())[:300].decode("utf-8", "replace")
                        last = f"{model}: HTTP {r.status_code}"
                        log.warning("gemini %s -> %s %s", model, r.status_code, err)
                        if r.status_code in (429, 500, 502, 503, 504):
                            continue
                        msg = "The AI service rejected the request."
                        if r.status_code in (401, 403):
                            msg = "The AI key was rejected (check GEMINI_API_KEY)."
                        yield {"type": "error", "message": msg}
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            d = json.loads(line[5:])
                        except ValueError:
                            continue
                        for c in d.get("candidates", []):
                            for part in c.get("content", {}).get("parts", []):
                                if part.get("text") and not part.get("thought"):
                                    sent = True
                                    yield {"type": "delta", "text": part["text"]}
                yield {"type": "done", "model": model}
                return
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{model}: {type(e).__name__}"
                log.warning("gemini %s failed: %r", model, e)
                if sent:  # partial answer already shown: don't restart with another model
                    yield {"type": "error", "message": "The answer was cut off. Please ask again."}
                    return
        yield {"type": "error", "message": f"The AI service is busy. Please try again shortly. ({last})"}
    finally:
        if own:
            await client.aclose()


_FRAME_RE = re.compile(r"^/api/(?:frame/(?P<k1>[0-9a-f]{16})/(?P<n1>[A-Za-z0-9_]+)\.png"
                       r"|jobs/(?P<k2>[0-9a-f]{16})/(?:exposure/(?P<exp>\d+)\.png"
                       r"|diff\.png\?a=(?P<a>\d+|deep)&b=(?P<b>\d+|deep)&band=(?P<band>all|D[1-6])))$")


def frame_file(src: str | None, job_id: str, frames_dir) -> object | None:
    """Map the frame URL the browser is showing to the rendered PNG on disk (or None).
    Only files of this job, by strict pattern: never an arbitrary path."""
    if not src:
        return None
    m = _FRAME_RE.match(src.split("#")[0])
    if not m or (m.group("k1") or m.group("k2")) != job_id:
        return None
    d = frames_dir / job_id
    if m.group("n1"):
        f = d / f"{m.group('n1')}.png"
    elif m.group("exp"):
        f = d / f"exp_{m.group('exp')}.png"
    else:
        f = d / f"x_{m.group('a')}_{m.group('b')}_{m.group('band')}.png"
    return f if f.exists() else None
