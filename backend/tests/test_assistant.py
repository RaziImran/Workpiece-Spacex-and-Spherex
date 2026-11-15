"""AI guide: context building, safety checks and the Gemini streaming client (mocked)."""
import json

import httpx
import pytest
from pydantic import SecretStr

from app import assistant
from app.config import settings

MANIFEST = {
    "params": {"ra": 133.76, "dec": -7.2422, "fov": 5 / 60, "detectors": []},
    "n_images": 24, "n_failed": 0, "n_phot_masked": 1, "bands": ["all", "D1", "D6", "rgb"],
    "visits": [{"index": 0, "date_start": "2025-05-05", "date_end": "2025-05-06", "n": 12, "n_ok": 12,
                "flags_mean": {"SOURCE": 0.5, "HALO": 0.02}}],
    "photometry": [
        {"detector": 1, "flux_mjy": 0.9, "err_mjy": 0.05, "wavelength_um": 0.9, "t_mid": 60800.0},
        {"detector": 6, "flux_mjy": 0.6, "err_mjy": 0.1, "wavelength_um": 4.8, "t_mid": 60800.0},
        {"detector": 6, "flux_mjy": None, "err_mjy": None, "wavelength_um": 4.7, "t_mid": 61000.0,
         "flag": "core_masked", "wings_rel": 1200.0},
    ],
}


def test_build_context_has_the_facts():
    gaia = {"n_pix": 49, "alignment": {"n": 3, "median_abs_px": 0.1, "reliable": True},
            "sources": [{"id": "1", "g": 8.2, "pmra": -800, "pmdec": 10360, "parallax": 547,
                         "xy": [[0.5, 0.5], [0.5, 0.47]]}]}
    ctx = json.loads(assistant.build_context(
        MANIFEST, {"mode": "diff", "band": "all", "pick": {"ra": 133.7, "dec": -7.2}},
        gaia, None, [], [{"catalog": "Gaia", "sep_arcsec": 7.9, "mag": 16.3}]))
    assert ctx["target"]["field_of_view_arcmin"] == 5.0
    assert ctx["target_photometry_by_detector"]["D6"]["saturated_n"] == 1
    assert ctx["target_photometry_by_detector"]["D1"]["flux_mjy_median"] == 0.9
    assert ctx["gaia"]["fast_movers"][0]["pm_mas_per_yr"] == [-800, 10360]
    assert ctx["blends_near_target"][0]["sep_arcsec"] == 7.9
    assert ctx["current_view"]["mode"] == "diff" and "clicked_position" in ctx["current_view"]


def test_frame_file_only_serves_this_jobs_rendered_frames(tmp_path):
    job = "0123456789abcdef"
    d = tmp_path / job
    d.mkdir()
    (d / "v0_all_stack.png").write_bytes(b"png")
    (d / "x_2_0_all.png").write_bytes(b"png")
    ok = assistant.frame_file(f"/api/frame/{job}/v0_all_stack.png", job, tmp_path)
    assert ok is not None and ok.name == "v0_all_stack.png"
    assert assistant.frame_file(f"/api/jobs/{job}/diff.png?a=2&b=0&band=all", job, tmp_path).name == "x_2_0_all.png"
    for bad in [f"/api/frame/{job}/../../secret.png", f"/api/frame/fedcba9876543210/v0_all_stack.png",
                "http://evil/x.png", f"/api/frame/{job}/v0_all_stack.png/../x", None, ""]:
        assert assistant.frame_file(bad, job, tmp_path) is None


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "ai_per_client_per_10min", 2)
    assistant._hits.clear()
    assert assistant.allow("1.2.3.4")[0] and assistant.allow("1.2.3.4")[0]
    assert not assistant.allow("1.2.3.4")[0]
    assert assistant.allow("5.6.7.8")[0]  # other users unaffected


def _sse(*texts):
    return "".join("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": t}]}}]}) + "\r\n\r\n"
                   for t in texts).encode()


async def test_stream_answer_streams_and_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-key"))
    monkeypatch.setattr(settings, "ai_models", ["busy-model", "good-model"])
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if "busy-model" in request.url.path:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, content=_sse("Hello ", "there."),
                              headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [e async for e in assistant.stream_answer("q?", [{"role": "user", "text": "hi"}],
                                                       "{}", b"\x89PNG", client=client)]
    await client.aclose()
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hello there."
    assert events[-1] == {"type": "done", "model": "good-model"}
    # key goes in a header (never the URL); the frame is attached as an image part
    assert seen[-1].headers["x-goog-api-key"] == "test-key" and "test-key" not in str(seen[-1].url)
    body = json.loads(seen[-1].content)
    assert body["contents"][-1]["parts"][-1]["inline_data"]["mime_type"] == "image/png"
    assert body["system_instruction"]["parts"][0]["text"].startswith("You are the built-in guide")


async def test_stream_answer_bad_key_message(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("bad"))
    monkeypatch.setattr(settings, "ai_models", ["m"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403, json={})))
    events = [e async for e in assistant.stream_answer("q", [], "{}", None, client=client)]
    await client.aclose()
    assert events == [{"type": "error", "message": "The AI key was rejected (check GEMINI_API_KEY)."}]


def test_ask_endpoint_disabled_without_key(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setattr(settings, "gemini_api_key", None)
    r = TestClient(app).post("/api/jobs/0123456789abcdef/ask", json={"question": "hi"})
    assert r.status_code == 503


def test_health_never_leaks_the_key(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("super-secret-value"))
    r = TestClient(app).get("/api/health")
    assert r.json()["ai_enabled"] is True and "super-secret-value" not in r.text


def test_target_motion_expected_shift():
    m = {"visits": [{"t_start": 60800.0, "t_end": 60801.0}, {"t_start": 61160.0, "t_end": 61165.25}]}
    out = assistant.target_motion({"main_id": "WISE 0855", "source": "simbad", "pmra": -8123.7,
                                   "pmdec": 673.2, "parallax": 439.0}, m)
    assert out["total_proper_motion_arcsec_per_yr"] == 8.15
    assert 7.5 < out["expected_shift_first_to_last_visit"]["arcsec"] < 8.5
    assert 1.2 < out["expected_shift_first_to_last_visit"]["pixels"] < 1.4
    assert assistant.target_motion({"source": "coordinates"}, m) is None
