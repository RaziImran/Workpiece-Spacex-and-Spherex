import numpy as np
import pytest

from app.overlays import gaia
from app.processing import target_wcs


def src(ra, dec, pmra=0.0, pmdec=0.0, g=12.0):
    return {"source_id": "1", "ra": ra, "dec": dec, "pmra": pmra, "pmdec": pmdec,
            "parallax": None, "phot_g_mean_mag": g}


def test_positions_at_barnard_moves_north():
    p = gaia.positions_at([src(269.4485, 4.7394, -801.55, 10362.39)], [2016.0, 2026.0])
    ddec = (p[0, 1, 1] - p[0, 0, 1]) * 3600
    dra = (p[0, 1, 0] - p[0, 0, 0]) * 3600 * np.cos(np.radians(4.74))
    assert abs(ddec - 103.62) < 0.1 and abs(dra - -8.02) < 0.1


def test_positions_handle_missing_pm():
    p = gaia.positions_at([src(10, 20, None, None)], [2026.0])
    assert np.allclose(p[0, 0], [10, 20])


def test_to_display_is_flipped_and_centred():
    w, (n, _) = target_wcs(10, 20, 0.1, 6.0)
    north = gaia.to_display(w, n, np.array([[10.0, 20.0 + 0.02]]))[0]
    centre = gaia.to_display(w, n, np.array([[10.0, 20.0]]))[0]
    east = gaia.to_display(w, n, np.array([[10.02, 20.0]]))[0]
    assert np.allclose(centre, [0.5, 0.5], atol=1e-3)
    assert north[1] < 0.5  # north is up on screen
    assert east[0] < 0.5   # east is left (standard sky orientation)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.6, -0.4)])
def test_alignment_recovers_known_offset(shift):
    w, (n, _) = target_wcs(150, 2, 0.1, 6.2)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    stack = rng.normal(0, 0.01, (n, n))
    stars, radec = [], []
    for x, y in [(12, 15), (40, 10), (25, 30), (45, 45), (10, 48)]:
        stack += np.exp(-(((xx - x - shift[0]) ** 2 + (yy - y - shift[1]) ** 2) / (2 * 0.9 ** 2)))
        radec.append(w.pixel_to_world_values(x, y))
        stars.append(12.0)
    out = gaia.alignment(stack, w, np.array(radec), np.array(stars))
    assert out["n"] == 5
    assert abs(out["median_dx_px"] - shift[0]) < 0.1
    assert abs(out["median_dy_px"] - shift[1]) < 0.1


def test_alignment_skips_masked_cores():
    w, (n, _) = target_wcs(150, 2, 0.1, 6.2)
    stack = np.zeros((n, n))
    stack[25, 25] = np.nan
    out = gaia.alignment(stack, w, np.array([w.pixel_to_world_values(25, 25)]), np.array([10.0]))
    assert out["n"] == 0


@pytest.mark.network
async def test_gaia_sources_live():
    out = await gaia.gaia_sources(76.95659, 30.40129, 0.05)
    assert out["provider"] in gaia.PROVIDERS and len(out["sources"]) > 10
    g = [s["phot_g_mean_mag"] for s in out["sources"]]
    assert g == sorted(g)


def test_alignment_robust_to_neighbour_halo():
    """A steep background slope (bright neighbour's halo) must not bias the centroid."""
    w, (n, _) = target_wcs(150, 2, 0.1, 6.2)
    yy, xx = np.mgrid[0:n, 0:n]
    stack = 0.08 * xx  # strong gradient across the field
    pos = [(12, 15), (40, 10), (25, 30)]
    for x, y in pos:
        stack = stack + np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.9 ** 2)))
    radec = np.array([w.pixel_to_world_values(x, y) for x, y in pos])
    out = gaia.alignment(stack, w, radec, np.array([12.0, 12.0, 12.0]))
    assert out["n"] == 3 and out["reliable"] and out["median_abs_px"] < 0.05
