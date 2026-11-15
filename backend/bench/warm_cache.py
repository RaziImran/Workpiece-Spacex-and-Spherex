"""Warm a running server's cache for the challenge targets, over HTTP.

Works against local or deployed servers (it only uses the public API):

    python -m bench.warm_cache                                   # http://127.0.0.1:8000
    python -m bench.warm_cache --base-url https://your-app.example --fovs 3 5 10
    python -m bench.warm_cache --check-only                      # just verify "instant"

For each target and FOV it resolves the name, runs the job to completion (frames,
photometry), fetches the Gaia overlay, then repeats the request and checks that the
second load is under --target-s (default 2 s, Step 14's "done when").

Run it after every deploy and whenever a new collection is added to the config
(collections are part of the cache keys, so old results don't count).
"""
import argparse
import json
import sys
import time

import httpx

TARGETS = {
    "Barnard's Star": "Barnard's Star",
    "Epsilon Indi B": "eps Ind B",
    "WISE 0855": "WISE J085510.83-071442.5",
    "RW Aurigae": "RW Aur",
    "R Draconis": "R Dra",
}


def run_job(c: httpx.Client, ra: float, dec: float, fov_deg: float) -> dict:
    r = c.post("/api/jobs", json={"ra": ra, "dec": dec, "fov": fov_deg})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    t0, first, n, last = time.perf_counter(), None, 0, None
    with c.stream("GET", f"/api/jobs/{job_id}/events", timeout=None) as resp:
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if event == "found":
                    n = data["n"]
                elif event == "frame" and first is None:
                    first = time.perf_counter() - t0
                elif event == "progress" and data["done"] % 25 == 0:
                    print(f"      {data['done']}/{data['total']}", flush=True)
                elif event in ("complete", "error"):
                    last = (event, data)
                    break
    total = time.perf_counter() - t0
    if last is None or last[0] == "error":
        raise RuntimeError(f"job {job_id} failed: {last}")
    m = last[1]["manifest"]
    g = c.get(f"/api/jobs/{job_id}/gaia", timeout=120)
    gaia = g.json() if g.status_code == 200 else {"error": g.text[:100]}
    return {"job_id": job_id, "images": n, "failed": m["n_failed"],
            "visits": sum(1 for v in m["visits"] if v.get("stack")),
            "first_frame_s": round(first, 2) if first else None, "total_s": round(total, 2),
            "gaia_stars": len(gaia.get("sources", [])),
            "align_px": gaia.get("alignment", {}).get("median_abs_px")}


def main():
    ap = argparse.ArgumentParser()
    # 127.0.0.1, not localhost: on Windows "localhost" cost httpx ~2 s per request (IPv6 first)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--fovs", nargs="*", type=float, default=[3, 5, 10], help="arcmin")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--target-s", type=float, default=2.0)
    ap.add_argument("--check-only", action="store_true",
                    help="don't warm; fail if anything isn't already instant")
    a = ap.parse_args()

    ok = True
    rows = []
    with httpx.Client(base_url=a.base_url.rstrip("/"), timeout=120) as c:
        h = c.get("/api/health").json()
        print(f"server ok, collections={h['collections']}, source={h['data_source']}")
        for label in a.targets:
            r = c.get("/api/resolve", params={"name": TARGETS.get(label, label)})
            r.raise_for_status()
            pos = r.json()
            for fov in a.fovs:
                fov_deg = round(fov / 60, 6)
                print(f"{label} @ {fov}'", flush=True)
                try:
                    cold = None if a.check_only else run_job(c, pos["ra"], pos["dec"], fov_deg)
                    t0 = time.perf_counter()
                    warm = run_job(c, pos["ra"], pos["dec"], fov_deg)
                    warm_s = round(time.perf_counter() - t0, 2)
                except Exception as e:
                    print(f"   ERROR {e}")
                    ok = False
                    continue
                fast = warm_s <= a.target_s
                ok &= fast
                row = {"target": label, "fov_arcmin": fov, **(cold or warm),
                       "reload_s": warm_s, "instant": fast}
                rows.append(row)
                print("   " + json.dumps(row), flush=True)
    print(f"\n| target | FOV | images | failed | visits | cold total s | reload s | Gaia | align px |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['target']} | {r['fov_arcmin']}' | {r['images']} | {r['failed']} | {r['visits']} "
              f"| {r['total_s']} | {r['reload_s']} | {r['gaia_stars']} | {r['align_px']} |")
    print("\nALL INSTANT" if ok else "\nSOME TARGETS NOT INSTANT / FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
