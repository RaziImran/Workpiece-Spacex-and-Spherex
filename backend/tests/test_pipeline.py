import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app import jobs, metadata, processing, resolver
from app.main import app


def test_normalise_parses_detector_and_s3_key():
    raw = {"obs_id": "2025W19_1B_0116_1", "energy_bandpassname": "SPHEREx-D3",
           "access_url": "https://irsa/x/level2_2025W19_1B_0116_1D3_spx.fits", "t_min": 1.0,
           "t_max": 1.1, "cloud_access": '{"aws": {"bucket_name": "b", "key": "dr1/k.fits"}}',
           "dataproduct_subtype": "science"}
    r = metadata.normalise(raw)
    assert r["detector"] == 3 and r["s3_key"] == "dr1/k.fits" and r["uid"] == "2025W19_1B_0116_1_D3"


def test_merge_keeps_all_detectors_and_prefers_first_collection():
    def row(det, coll, t):
        return {"obs_id": "E1", "energy_bandpassname": f"SPHEREx-D{det}", "obs_collection": coll,
                "access_url": f"https://x/{coll}/E1D{det}.fits", "t_min": t, "t_max": t}
    dr1 = [row(d, "dr1", 5.0) for d in range(1, 7)]
    qr3 = [row(1, "qr3", 5.0), {**row(2, "qr3", 1.0), "obs_id": "E0"}]
    rows = metadata.merge([dr1, qr3])
    assert len(rows) == 7  # 6 detectors of E1 (from DR1) + E0
    assert all(r["obs_collection"] == "dr1" for r in rows if r["obs_id"] == "E1")
    assert rows[0]["obs_id"] == "E0"  # sorted by time


def test_group_by_visit():
    rows = [{"t_min": t} for t in [0, 1, 2, 180, 181, 400]]
    groups = processing.group_by_visit(rows, gap_days=30)
    assert [len(g) for g in groups] == [3, 2, 1]


def test_evenly_sample_covers_every_visit_and_detector():
    rows = [{"t_min": t + 0.01 * i, "detector": d, "i": i}
            for t in (0, 180, 360) for d in range(1, 7) for i in range(20)]
    out = processing.evenly_sample(rows, 24)
    assert len(out) == 24
    groups = processing.group_by_visit(out, 30)
    assert len(groups) == 3
    for g in groups:
        assert {r["detector"] for r in g} == set(range(1, 7))
    assert [r["t_min"] for r in out] == sorted(r["t_min"] for r in out)


def test_mask_bits_never_masks_source():
    h = fits.Header()
    h["MSKN0000"], h["MSKM0000"] = "TRANSIENT", 1
    h["MSKN0021"], h["MSKM0021"] = "SOURCE", 1 << 21
    bits = processing.mask_bits(h, ["TRANSIENT", "HOT"])
    assert bits & 1 and not bits & (1 << 21)


def test_target_wcs_shape():
    w, shape = processing.target_wcs(10, 20, 0.1, 6.0)
    assert shape == (60, 60)
    assert np.allclose(w.pixel_to_world_values(29.5, 29.5), (10, 20))


def test_job_key_stable_and_snapped():
    a = jobs.job_key(10.0, 20.0, 0.1, [3, 1])
    assert a == jobs.job_key(10.0 + 0.4 / 3600, 20.0, 0.1, [1, 3, 3])
    assert a != jobs.job_key(10.0, 20.0, 0.1, [])


def test_move_to_epoch_barnard():
    # Barnard's Star moves ~10.4"/yr north
    ra, dec = resolver.move_to_epoch(269.448503, 4.739420, -801.55, 10362.39, 2016.0, 2026.0, 546.98)
    assert 100 < (dec - 4.739420) * 3600 < 110


@pytest.mark.parametrize("params", [{"ra": 400, "dec": 0}, {"ra": 10, "dec": -91},
                                    {"ra": 10, "dec": 0, "fov": 5}])
def test_api_rejects_bad_input(params):
    r = TestClient(app).post("/api/jobs", json=params)
    assert r.status_code == 422


def test_api_frame_name_is_sanitised():
    r = TestClient(app).get("/api/frame/abc/..%2F..%2Fsecret.png")
    assert r.status_code in (400, 404)


def test_json_safe_removes_nan_recursively():
    import json
    out = jobs.json_safe({"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf")}], "d": "x"})
    assert out == {"a": None, "b": [1.0, None, {"c": None}], "d": "x"}
    json.dumps(out, allow_nan=False)


def test_fill_nans_fills_cores_bright_and_holes_as_sky():
    from app.render import fill_nans
    a = np.zeros((9, 9), np.float32)
    a[3:6, 3:6] = 100.0
    a[4, 4] = np.nan           # saturated core inside a bright star
    a[1, 7] = np.nan           # isolated masked pixel in empty sky
    out = fill_nans(a)
    assert out[4, 4] == 100.0 and out[1, 7] == 0.0


def test_despike_removes_lone_spikes_keeps_stars():
    rng = np.random.default_rng(3)
    img = rng.normal(0, 1, (40, 40)).astype(np.float32)
    yy, xx = np.mgrid[0:40, 0:40]
    img += 60 * np.exp(-(((xx - 10) ** 2 + (yy - 10) ** 2) / (2 * 0.8 ** 2)))  # undersampled star
    img[30, 30] = 80.0  # cosmic ray: one pixel
    out = processing.despike(img)
    assert np.isnan(out[30, 30])
    assert np.isfinite(out[8:13, 8:13]).all()


def test_fill_nans_fills_big_cores_but_not_border_gaps():
    from app.render import fill_nans
    a = np.full((40, 40), 5.0, np.float32)
    yy, xx = np.mgrid[0:40, 0:40]
    a[(xx - 20) ** 2 + (yy - 20) ** 2 <= 36] = np.nan   # 12 px saturated core
    a[:, :3] = np.nan                                     # outside the footprint
    out = fill_nans(a)
    assert np.isfinite(out[20, 20]) and abs(out[20, 20] - 5.0) < 1e-4
    assert np.isnan(out[10, 0])
