"""End-to-end smoke test of every API feature against a running server.

    python -m bench.smoke                       # WISE 0855, 5' field, http://127.0.0.1:8000
    python -m bench.smoke --name "R Dra" --base-url https://your-app

Exits non-zero if any check fails. Prints one line per check.
"""
import argparse
import json
import sys
import time

import httpx

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", default="WISE J085510.83-071442.5")
    ap.add_argument("--fov", type=float, default=5 / 60)
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base_url, timeout=300)

    h = c.get("/api/health").json()
    check("health", h.get("ok") is True, f"collections={h.get('collections')}")
    r = c.get("/api/resolve", params={"name": a.name}).json()
    check("resolve name", r.get("ra") is not None, f"{r.get('main_id')} via {r.get('source')}")
    for txt in ["133.795 -7.245", "08h55m10.83s -07d14m42.5s", "08:55:10.83 -07:14:42.5"]:
        rc = c.get("/api/resolve", params={"name": txt}).json()
        check(f"resolve coords '{txt}'", abs(rc["ra"] - 133.795) < 0.01 and abs(rc["dec"] + 7.245) < 0.01)
    check("bad input -> 422", c.post("/api/jobs", json={"ra": 400, "dec": 0}).status_code == 422)

    t0 = time.perf_counter()
    jid = c.post("/api/jobs", json={"ra": r["ra"], "dec": r["dec"], "fov": a.fov}).json()["job_id"]
    events = []
    with c.stream("GET", f"/api/jobs/{jid}/events", timeout=None) as resp:
        ev = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                d = json.loads(line.split(":", 1)[1])  # must be valid JSON (no NaN)
                events.append((ev, d))
                if ev in ("complete", "error"):
                    break
    kinds = [e for e, _ in events]
    check("job streams found->frame->complete", kinds[0] == "found" and "frame" in kinds and kinds[-1] == "complete",
          f"{len(events)} events in {time.perf_counter() - t0:.1f}s")
    m = events[-1][1].get("manifest", {})
    check("no failed images", m.get("n_failed") == 0, f"n_images={m.get('n_images')} failed={m.get('n_failed')}")
    v0 = m["visits"][0]
    check("per-detector bands", len(m.get("bands", [])) >= 5, f"bands={m.get('bands')}")
    check("rgb + diff frames", "rgb" in v0["bands"] and all("diff" in v["bands"].get("all", {}) for v in m["visits"]))
    check("deep stack", "all" in m.get("deep", {}))
    for url in [v0["stack"], v0["bands"]["rgb"]["stack"], m["deep"]["all"], m["visits"][-1]["diff"]]:
        rr = c.get(url)
        check(f"frame {url.rsplit('/', 1)[-1]}", rr.status_code == 200 and rr.content[:4] == b"\x89PNG")
    n = len(m["visits"])
    common = [b for b in m["visits"][-1]["bands"] if b in v0["bands"] and b.startswith("D")]
    rr = c.get(f"/api/jobs/{jid}/diff.png", params={"a": n - 1, "b": 0, "band": common[-1] if common else "all"})
    check("A-B any visits", rr.status_code == 200 and rr.content[:4] == b"\x89PNG")
    rr = c.get(f"/api/jobs/{jid}/exposure/0.png")
    check("single exposure", rr.status_code == 200 and rr.content[:4] == b"\x89PNG")
    rr = c.get(f"/api/jobs/{jid}/flipbook.gif", params={"band": "all", "mode": "diff"})
    check("GIF export", rr.status_code == 200 and rr.content[:3] == b"GIF", f"{len(rr.content) // 1024} KB")
    s = c.get(f"/api/jobs/{jid}/spectrum").json()
    waves = [p["wavelength_um"] for p in s["points"]]
    check("target spectrum (or saturated light curve)", len(waves) > 5 or len(s.get("saturated", [])) > 5, f"{len(waves)} pts, {min(waves):.2f}-{max(waves):.2f} um" if waves else f"saturated: {len(s.get('saturated', []))} wing pts")
    t1 = time.perf_counter()
    # a spot ~1' from the target (well clear of a bright target's halo), inside the field
    off = c.get(f"/api/jobs/{jid}/spectrum", params={"ra": r["ra"], "dec": r["dec"] + min(0.016, a.fov / 3)}).json()
    check("click-anywhere spectrum", len(off["points"]) > 0, f"{len(off['points'])} pts in {time.perf_counter() - t1:.1f}s")
    lc = c.get(f"/api/jobs/{jid}/lightcurve", params={"detector": 6}).json()
    check("light curve", len(lc["points"]) > 0 and all(p["detector"] == 6 for p in lc["points"]),
          f"{len(lc['points'])} D6 pts")
    ex = c.get(f"/api/jobs/{jid}/exposures").json()
    check("exposure metadata + flags", len(ex["exposures"]) == m["n_images"] and "flags_mean" in ex["visits"][0],
          f"visit 1 flags: {ex['visits'][0]['flags_mean']}")
    g = c.get(f"/api/jobs/{jid}/gaia")
    gj = g.json() if g.status_code == 200 else {}
    check("Gaia overlay", g.status_code == 200 and len(gj.get("sources", [])) > 0,
          f"{gj.get('provider')} n={len(gj.get('sources', []))} align={gj.get('alignment')}")
    w = c.get(f"/api/jobs/{jid}/wise")
    wj = w.json() if w.status_code == 200 else {"detail": w.text[:100]}
    check("WISE overlay", w.status_code == 200 and len(wj.get("sources", [])) > 0,
          f"{wj.get('provider')} n={len(wj.get('sources', []))} {wj.get('detail', '')}")
    t2 = time.perf_counter()
    c.post("/api/jobs", json={"ra": r["ra"], "dec": r["dec"], "fov": a.fov})
    check("repeat request instant", time.perf_counter() - t2 < 1.0, f"{time.perf_counter() - t2:.3f}s")
    ui = c.get("/ui/")
    check("UI served", ui.status_code == 200 and b"Aladin" in ui.content)
    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  (job {jid})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
