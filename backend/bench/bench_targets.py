"""Timing harness for the challenge targets. Run from backend/:

    python -m bench.bench_targets                       # all 5 targets, 5' FOV
    python -m bench.bench_targets --targets "WISE 0855" --limit 40
    python -m bench.bench_targets --warm-only           # skip cold run (cache warm-up check)

For each target: resolve -> metadata -> cold job -> warm job (cutouts cached, job
manifest dropped so the processing really reruns). Prints a JSON + markdown table.
"""
import argparse
import asyncio
import json
import time

from app import fetcher
from app.cache import disk_cache
from app.config import settings
from app.jobs import job_key, manager
from app.metadata import query_images
from app.resolver import resolve

TARGETS = {
    "Barnard's Star": "Barnard's Star",
    "Epsilon Indi B": "eps Ind B",
    "WISE 0855": "WISE J085510.83-071442.5",
    "RW Aurigae": "RW Aur",
    "R Draconis": "R Dra",
}


async def run_job(ra, dec, fov):
    job = manager.submit(ra, dec, fov, [])
    t0, first = time.perf_counter(), None
    async for ev in job.stream():
        if ev["type"] == "frame" and first is None:
            first = time.perf_counter() - t0
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
    return job.manifest, first, time.perf_counter() - t0


async def bench(label, name, fov, warm_only):
    out = {}
    t = time.perf_counter()
    info = await asyncio.to_thread(resolve, name)
    out["resolve_s"] = round(time.perf_counter() - t, 2)
    ra, dec = info["ra"], info["dec"]
    t = time.perf_counter()
    rows = await query_images(ra, dec)
    out["metadata_s"] = round(time.perf_counter() - t, 2)
    out["n_images"] = len(rows)
    jk = job_key(ra, dec, fov, [])
    if not warm_only:
        disk_cache.delete(("job", jk))
        manager.jobs.pop(jk, None)
        before = dict(fetcher.STATS)
        m, first, total = await run_job(ra, dec, fov)
        out["cold_first_frame_s"] = round(first or -1, 2)
        out["cold_total_s"] = round(total, 2)
        out["failed"] = m["n_failed"]
        out["visits"] = len(m["visits"])
        out["origins"] = m["origins"]
        out["MB_downloaded"] = round((fetcher.STATS["bytes"] - before.get("bytes", 0)) / 1e6, 1)
        out["phot_points"] = len(m["photometry"])
    disk_cache.delete(("job", jk))
    manager.jobs.pop(jk, None)
    m, first, total = await run_job(ra, dec, fov)
    out["warm_total_s"] = round(total, 2)
    t = time.perf_counter()
    manager.submit(ra, dec, fov, [])  # finished job -> instant
    out["repeat_s"] = round(time.perf_counter() - t, 4)
    print(f"{label}: {json.dumps(out)}", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--fov", type=float, default=5 / 60)
    ap.add_argument("--limit", type=int, default=None, help="max images per job")
    ap.add_argument("--warm-only", action="store_true")
    a = ap.parse_args()
    if a.limit:
        settings.max_images_per_job = a.limit
    manager.start_pool()
    results = {}
    try:
        for label in a.targets:
            try:
                results[label] = await bench(label, TARGETS.get(label, label), a.fov, a.warm_only)
            except Exception as e:
                results[label] = {"error": repr(e)}
                print(f"{label}: ERROR {e!r}", flush=True)
    finally:
        manager.shutdown()
        await fetcher.http.aclose()
    cols = sorted({k for r in results.values() for k in r if k != "origins"})
    print("\n| target | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for label, r in results.items():
        print(f"| {label} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    asyncio.run(main())
