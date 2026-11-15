"""Background jobs: a request becomes a job that streams events as work finishes.

Events (in order): found -> progress... / frame... -> complete | error
The job id IS the cache key of its parameters, so identical requests share one job
(de-duplication), and a finished job is restored from disk after a restart.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone

import numpy as np
from astropy.time import Time

from . import processing
from .cache import FRAMES_DIR, disk_cache, snap
from .config import settings
from .fetcher import fetch_all
from .metadata import collections, query_images
from .fetcher import cutout_key
from .products import BANDS, WAVE_RANGES, Products

log = logging.getLogger(__name__)
TERMINAL = ("complete", "error")


def job_key(ra: float, dec: float, fov: float, detectors: list[int]) -> str:
    step = 3 / 3600
    raw = json.dumps([round(snap(ra, step), 5), round(snap(dec, step), 5), round(fov, 4),
                      sorted(set(detectors)), collections(), settings.pixel_arcsec,
                      settings.fetch_zodi, sorted(settings.mask_flags),
                      settings.max_images_per_job, settings.visit_gap_days, settings.flags_mode,
                      settings.flags_box_px, "v9"])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def json_safe(obj):
    """NaN/inf -> None, recursively. JSON has no NaN: a bare NaN broke JSON.parse in the
    browser and made the status endpoint 500 (found on R Dra's masked-core photometry)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def mjd_to_iso(mjd: float) -> str:
    return Time(mjd, format="mjd").to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d")


class Job:
    def __init__(self, key: str, params: dict):
        self.id = key
        self.params = params
        self.status = "running"
        self.events: list[dict] = []
        self.cond = asyncio.Condition()
        self.manifest: dict | None = None
        self.task: asyncio.Task | None = None
        self.background: list[asyncio.Task] = []  # keep references so tasks aren't GC'd

    async def emit(self, ev: dict):
        ev = json_safe(ev)
        async with self.cond:
            self.events.append(ev)
            if ev["type"] in TERMINAL:
                self.status = ev["type"]
            self.cond.notify_all()

    async def stream(self):
        """Replays past events, then waits for new ones. Works for many listeners."""
        i = 0
        while True:
            async with self.cond:
                while i >= len(self.events):
                    await self.cond.wait()
                batch = self.events[i:]
                i = len(self.events)
            for ev in batch:
                yield ev
                if ev["type"] in TERMINAL:
                    return


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pool: ProcessPoolExecutor | None = None
        self.job_sem: asyncio.Semaphore | None = None

    def start_pool(self):
        if self.pool is None:
            # Each worker gets ONE BLAS thread. Without this, N workers x N OpenBLAS threads
            # exhausted memory on Windows ("OpenBLAS error: Memory allocation still failed")
            # and killed the workers. Spawned children inherit these at start.
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            # Frames are ~60x60 px: a few workers keep up with the download rate.
            n = settings.process_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
            self.pool = ProcessPoolExecutor(max_workers=n)
            for _ in range(n):  # warm up: workers import astropy/reproject now, not mid-job
                self.pool.submit(_warm)

    async def run_cpu(self, fn, *args):
        """Run in the process pool; if a worker died (BrokenProcessPool), rebuild once and retry."""
        loop = asyncio.get_running_loop()
        for attempt in (1, 2):
            self.start_pool()
            pool = self.pool
            try:
                return await loop.run_in_executor(pool, fn, *args)
            except BrokenProcessPool:
                log.error("process pool broke (attempt %d); rebuilding", attempt)
                if self.pool is pool:  # only the first caller to notice rebuilds it
                    pool.shutdown(wait=False, cancel_futures=True)
                    self.pool = None
                if attempt == 2:
                    raise

    def shutdown(self):
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def get(self, job_id: str) -> Job | None:
        if job_id in self.jobs:
            return self.jobs[job_id]
        manifest = disk_cache.get(("job", job_id))
        if manifest is None:
            return None
        return self._restore(manifest)

    def _restore(self, manifest: dict) -> Job:
        job = Job(manifest["job_id"], manifest["params"])
        job.manifest = manifest
        job.status = "complete"
        job.events = [{"type": "found", "n": manifest["n_images"], "visits": manifest["visits"],
                       "cached": True}]
        job.events += [{"type": "frame", **v} for v in manifest["visits"] if v.get("stack")]
        job.events.append({"type": "complete", "manifest": manifest})
        self.jobs[job.id] = job
        return job

    def submit(self, ra: float, dec: float, fov: float, detectors: list[int]) -> Job:
        key = job_key(ra, dec, fov, detectors)
        existing = self.get(key)
        if existing and existing.status != "error":
            return existing  # running or finished: share it
        job = Job(key, {"ra": ra, "dec": dec, "fov": fov, "detectors": sorted(set(detectors))})
        self.jobs[key] = job
        job.task = asyncio.create_task(self._run_safe(job))
        return job

    async def _run_safe(self, job: Job):
        # A public server must not turn into a firehose against IRSA/S3: cap concurrent jobs.
        if self.job_sem is None:
            self.job_sem = asyncio.Semaphore(settings.max_concurrent_jobs)
        try:
            if self.job_sem.locked():
                await job.emit({"type": "queued",
                                "message": "waiting for another field to finish"})
            async with self.job_sem:
                await self._run(job)
        except Exception as e:
            log.exception("job=%s failed", job.id)
            await job.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self._prune()

    def _prune(self, keep: int = 200):
        """Finished jobs live on disk; keep only the most recent ones in memory."""
        done = [k for k, j in self.jobs.items() if j.status != "running"]
        for k in done[:-keep] if len(done) > keep else []:
            self.jobs.pop(k, None)

    async def _run(self, job: Job):
        self.start_pool()
        p = job.params
        ra, dec, fov, dets = p["ra"], p["dec"], p["fov"], p["detectors"]
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        # Gaia is slow (15-40 s): start it now so it's cached when the UI asks for it.
        gaia_task = asyncio.create_task(_prefetch_gaia(ra, dec, fov))
        job.background.append(gaia_task)

        rows = await query_images(ra, dec)
        timings["metadata_s"] = round(time.perf_counter() - t_start, 2)
        if dets:
            rows = [r for r in rows if r.get("detector") in dets]
        rows = processing.evenly_sample(rows, settings.max_images_per_job, settings.visit_gap_days)
        groups = processing.group_by_visit(rows, settings.visit_gap_days)
        visit_of = {r["uid"]: k for k, g in enumerate(groups) for r in g}
        visits = [{"index": k, "n": len(g), "t_start": g[0]["t_min"], "t_end": g[-1]["t_max"],
                   "date_start": mjd_to_iso(g[0]["t_min"]), "date_end": mjd_to_iso(g[-1]["t_max"]),
                   "stack": None, "diff": None, "bands": {}, "n_ok": 0}
                  for k, g in enumerate(groups)]
        await job.emit({"type": "found", "n": len(rows), "visits": visits})
        if not rows:
            return await self._finish(job, visits, [], [], timings, t_start, {})

        out_wcs, shape = processing.target_wcs(ra, dec, fov, settings.pixel_arcsec)
        out_header = out_wcs.to_header().tostring()
        remaining = [len(g) for g in groups]
        arrays: dict[int, dict[int, list[np.ndarray]]] = {k: {} for k in range(len(groups))}
        phot: list[dict] = []
        exposures: list[dict] = []
        failures: list[dict] = []
        origins: dict[str, int] = {}
        state = {"first_frame_s": None, "done": 0}
        prod = Products(job.id, FRAMES_DIR / job.id)
        lock = asyncio.Lock()  # products are rendered one visit at a time (shared stretch)
        diffed: set[int] = set()

        def merge(k, d):
            for band, v in d.items():
                visits[k]["bands"].setdefault(band, {}).update(v)
            allb = visits[k]["bands"].get("all", {})
            visits[k]["stack"], visits[k]["diff"] = allb.get("stack"), allb.get("diff")

        async def finalize_visit(k: int):
            async with lock:
                updated = set()
                if any(arrays[k].values()):
                    merge(k, await asyncio.to_thread(prod.add_visit, k, arrays[k]))
                    arrays[k] = {}  # free memory
                    updated.add(k)
                    if state["first_frame_s"] is None:
                        state["first_frame_s"] = round(time.perf_counter() - t_start, 2)
                # Difference frames subtract the earliest visit with data; that reference
                # is final once every earlier visit has finished.
                done_visits = sorted({kk for (kk, _b) in prod.stacks if isinstance(kk, int)})
                if done_visits:
                    ref = done_visits[0]
                    if all(remaining[i] == 0 for i in range(ref)):
                        todo = [i for i in done_visits if i not in diffed]
                        diffed.update(todo)
                        for i, d in (await asyncio.to_thread(prod.diffs, ref, todo)).items():
                            merge(i, d)
                            updated.add(i)
                for i in sorted(updated):
                    await job.emit({"type": "frame", **visits[i]})

        async def one_done(k: int):
            remaining[k] -= 1
            if remaining[k] == 0:
                await finalize_visit(k)

        async def process(row, data):
            k = visit_of[row["uid"]]
            try:
                res = await self.run_cpu(processing.prepare, data, out_header, shape, ra, dec,
                                         list(settings.mask_flags))
                arrays[k].setdefault(res["detector"], []).append(res["array"])
                visits[k]["n_ok"] += 1
                exp = {"uid": row["uid"], "visit": k, "t_mid": res["t_mid"],
                       "date": mjd_to_iso(res["t_mid"]), "detector": res["detector"],
                       "wavelength_um": res["wavelength_um"], "bandwidth_um": res["bandwidth_um"],
                       "flags": res["flags"], "frac_masked": res["frac_masked"],
                       "collection": row.get("obs_collection"),
                       "ckey": cutout_key(row, ra, dec, fov, settings.fetch_zodi)}
                if res["phot"]:
                    phot.append({"uid": row["uid"], "visit": k, "t_mid": res["t_mid"],
                                 "detector": res["detector"], **res["phot"]})
                    exp["flux_mjy"] = res["phot"]["flux_mjy"]
                    exp["phot_flag"] = res["phot"]["flag"]
                exposures.append(exp)
            except Exception as e:
                failures.append({"uid": row["uid"], "stage": "process", "error": repr(e)[:200]})
            await one_done(k)

        t_fetch = time.perf_counter()
        proc_tasks = []
        async for row, data, origin in fetch_all(rows, ra, dec, fov):
            state["done"] += 1
            origins[origin] = origins.get(origin, 0) + 1
            if isinstance(data, Exception):
                if type(data).__name__ != "NoOverlap":
                    failures.append({"uid": row["uid"], "stage": "fetch", "error": repr(data)[:200]})
                await one_done(visit_of[row["uid"]])
            else:
                proc_tasks.append(asyncio.create_task(process(row, data)))
            await job.emit({"type": "progress", "done": state["done"], "total": len(rows),
                            "failed": len(failures)})
        timings["fetch_s"] = round(time.perf_counter() - t_fetch, 2)
        await asyncio.gather(*proc_tasks)
        timings["first_frame_s"] = state["first_frame_s"]
        async with lock:
            deep = await asyncio.to_thread(prod.deep, [v["index"] for v in visits])
        # contamination summary per visit: mean fraction of pixels per flag (near target)
        for v in visits:
            fr: dict[str, list[float]] = {}
            exps = [e for e in exposures if e["visit"] == v["index"]]
            for e in exps:
                for name, f in (e["flags"] or {}).get("frac", {}).items():
                    fr.setdefault(name, []).append(f)
            v["flags_mean"] = {n: round(sum(x) / len(exps), 4) for n, x in sorted(fr.items())}
        exposures.sort(key=lambda e: e["t_mid"])
        for i, e in enumerate(exposures):
            e["i"] = i
        extra = {"exposures": exposures, "deep": deep, "stretch": prod.stretch,
                 "dlim": prod.dlim, "diff_ref": prod.diff_ref,
                 "bands": [b for b in BANDS + ["rgb"]
                           if any(b in v["bands"] for v in visits)],
                 "wave_ranges": WAVE_RANGES,
                 "grid": {"n": shape[0], "pixel_arcsec": settings.pixel_arcsec}}
        await self._finish(job, visits, phot, failures, timings, t_start, origins, extra)

    async def _finish(self, job, visits, phot, failures, timings, t_start, origins, extra=None):
        timings["total_s"] = round(time.perf_counter() - t_start, 2)
        phot.sort(key=lambda r: (r["wavelength_um"] or 0))
        n_masked = sum(1 for r in phot if r.get("flag"))
        manifest = {"job_id": job.id, "params": job.params, "n_images": sum(v["n"] for v in visits),
                    "visits": visits, "photometry": phot, "failures": failures[:50],
                    "n_failed": len(failures), "n_phot_masked": n_masked, "timings": timings,
                    "origins": origins, **(extra or {}),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest = json_safe(manifest)
        job.manifest = manifest
        # A clean result is kept as long as metadata; one with failures only briefly (retry soon).
        disk_cache.set(("job", job.id), manifest,
                       expire=settings.metadata_ttl_s if not failures else 3600)
        log.info("job=%s n=%d failed=%d origins=%s timings=%s", job.id, manifest["n_images"],
                 len(failures), origins, timings)
        await job.emit({"type": "complete", "manifest": manifest})


def prune_frames(max_age_days: float) -> int:
    """L3 cache housekeeping: delete rendered frame folders not touched for a while."""
    import shutil
    cutoff = time.time() - max_age_days * 86400
    n = 0
    for d in FRAMES_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def gaia_radius(fov: float) -> float:
    return fov / math.sqrt(2) * 1.05  # cone that covers the whole square field


async def _prefetch_gaia(ra, dec, fov):
    from .overlays.gaia import gaia_sources
    try:
        await gaia_sources(ra, dec, gaia_radius(fov))
    except Exception as e:  # the overlay is optional; never fail a job over it
        log.warning("gaia prefetch failed: %s", e)


def _warm():
    import reproject  # noqa: F401
    import photutils.aperture  # noqa: F401
    return True


manager = JobManager()
