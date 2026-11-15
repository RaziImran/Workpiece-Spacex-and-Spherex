"""Gaia DR3 overlay: stars in the field, moved to each visit's date by proper motion.

Also the best alignment test we have: Gaia positions (accurate to <1 mas) should land
on the stars in our reprojected stacks. `alignment()` measures exactly that.

Gaia services vary a lot (2026-11-14, 4' cone: ESA 3-18 s, MAST 5-36 s, IRSA's
gaia_dr3_source timed out at 45 s, CDS VizieR 404), so:
- providers are tried in order with a timeout (config: gaia_providers),
- results are cached 30 days per sky tile,
- jobs start the query in the background so it's usually cached by the time the UI asks.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

import numpy as np

from ..cache import disk_cache, tile_key
from ..config import settings

log = logging.getLogger(__name__)
GAIA_EPOCH = 2016.0
_COLS = ["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag"]


def _rows(t, names: dict[str, str]) -> list[dict]:
    out = []
    for r in t:
        d = {}
        for ours, theirs in names.items():
            v = r[theirs]
            try:
                v = float(v)
                if math.isnan(v):
                    v = None
            except (TypeError, ValueError):
                v = None
            d[ours] = v
        if d["ra"] is not None and d["dec"] is not None:
            d["source_id"] = str(int(d["source_id"])) if d["source_id"] is not None else ""
            out.append(d)
    return out


def _adql(table: str, ra, dec, r, gmax, point="POINT('ICRS', ra, dec)", circle="CIRCLE('ICRS', {ra}, {dec}, {r})"):
    return (f"SELECT {', '.join(_COLS)} FROM {table} WHERE "
            f"CONTAINS({point}, {circle.format(ra=ra, dec=dec, r=r)}) = 1 "
            f"AND phot_g_mean_mag < {gmax}")


def _esa(ra, dec, r, gmax):
    from astroquery.gaia import Gaia

    t = Gaia.launch_job(_adql("gaiadr3.gaia_source", ra, dec, r, gmax)).get_results()
    return _rows(t, {c: c for c in _COLS})


def _irsa(ra, dec, r, gmax):
    from astroquery.ipac.irsa import Irsa

    t = Irsa.query_tap(_adql("gaia_dr3_source", ra, dec, r, gmax)).to_table()
    return _rows(t, {c: c for c in _COLS})


def _mast(ra, dec, r, gmax):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    t = Catalogs.query_region(SkyCoord(ra, dec, unit="deg"), radius=r * u.deg,
                              catalog="Gaia", version=3)
    t = t[t["phot_g_mean_mag"] < gmax]
    return _rows(t, {c: c for c in _COLS})


def _vizier(ra, dec, r, gmax, server="vizier.cfa.harvard.edu"):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    names = {"source_id": "Source", "ra": "RA_ICRS", "dec": "DE_ICRS", "pmra": "pmRA",
             "pmdec": "pmDE", "parallax": "Plx", "phot_g_mean_mag": "Gmag"}
    v = Vizier(columns=list(names.values()), column_filters={"Gmag": f"<{gmax}"}, row_limit=-1)
    v.VIZIER_SERVER = server  # CDS (the default) returned 404 on 2026-11-14
    res = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=r * u.deg,
                         catalog="I/355/gaiadr3")
    return _rows(res[0], names) if len(res) else []


PROVIDERS = {"esa": _esa, "irsa": _irsa, "mast": _mast, "vizier": _vizier}

# Import the clients now, on the main thread. Importing astroquery modules from several
# worker threads at once failed with a circular-import error in astropy.logger.
import astroquery.gaia  # noqa: E402,F401
import astroquery.ipac.irsa  # noqa: E402,F401
import astroquery.mast  # noqa: E402,F401
import astroquery.vizier  # noqa: E402,F401


async def gaia_sources(ra: float, dec: float, radius_deg: float,
                       gmax: float | None = None) -> dict:
    """Gaia DR3 stars in a cone (epoch 2016.0), brightest first, cached 30 days."""
    gmax = gmax or settings.gaia_gmax
    key = ("gaia", tile_key(ra, dec, arcmin=0.05), round(radius_deg, 4), gmax)
    if (hit := disk_cache.get(key)) is not None:
        return hit
    errors = {}
    for name in settings.gaia_providers:
        t0 = time.perf_counter()
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(PROVIDERS[name], ra, dec, radius_deg, gmax),
                timeout=settings.gaia_timeout_s)
        except Exception as e:  # timeout, HTTP error, schema change: try the next one
            errors[name] = f"{type(e).__name__}: {str(e)[:120]}"
            log.warning("gaia provider %s failed: %s", name, errors[name])
            continue
        rows.sort(key=lambda s: s["phot_g_mean_mag"] if s["phot_g_mean_mag"] is not None else 99)
        out = {"provider": name, "epoch": GAIA_EPOCH, "gmax": gmax,
               "sources": rows[: settings.gaia_max_sources],
               "secs": round(time.perf_counter() - t0, 2)}
        disk_cache.set(key, out, expire=30 * 24 * 3600)
        log.info("stage=gaia provider=%s n=%d secs=%.1f", name, len(rows), out["secs"])
        return out
    raise RuntimeError(f"all Gaia providers failed: {errors}")


def positions_at(sources: list[dict], epochs_jyear: list[float],
                 ref_epoch: float = GAIA_EPOCH) -> np.ndarray:
    """(n_sources, n_epochs, 2) RA/Dec in degrees, linear proper motion from 2016.0.

    Linear motion is accurate to << 1 SPHEREx pixel over a decade except within a few
    degrees of the celestial poles, where we clamp cos(dec).
    """
    ra = np.array([s["ra"] for s in sources])
    dec = np.array([s["dec"] for s in sources])
    pmra = np.nan_to_num(np.array([s["pmra"] if s["pmra"] is not None else 0.0 for s in sources]))
    pmdec = np.nan_to_num(np.array([s["pmdec"] if s["pmdec"] is not None else 0.0 for s in sources]))
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-3)
    dt = np.asarray(epochs_jyear)[None, :] - ref_epoch
    out = np.empty((len(sources), len(epochs_jyear), 2))
    out[..., 0] = (ra[:, None] + pmra[:, None] * dt / 3.6e6 / cosd[:, None]) % 360
    out[..., 1] = dec[:, None] + pmdec[:, None] * dt / 3.6e6
    return out


def to_display(wcs, n: int, radec: np.ndarray) -> np.ndarray:
    """RA/Dec -> fractions (0..1) of the displayed PNG, which is flipped vertically."""
    x, y = wcs.world_to_pixel_values(radec[..., 0], radec[..., 1])
    return np.stack([(x + 0.5) / n, 1 - (y + 0.5) / n], axis=-1)


def alignment(stack: np.ndarray, wcs, radec: np.ndarray, gmag: np.ndarray,
              g_bright: float = 15.0, g_saturate: float = 9.0, box: int = 3) -> dict:
    """Centroid bright, isolated Gaia stars in a stack; report offsets in pixels.

    radec: (n_sources, 2) positions at the stack's epoch.
    """
    ny, nx = stack.shape
    x, y = wcs.world_to_pixel_values(radec[:, 0], radec[:, 1])
    offs = []
    for xi, yi, g in zip(x, y, gmag):
        # Too faint: noisy centroid. Too bright: saturated/non-linear core and huge wings
        # (R Dra, G=6.4, gave a 2.4 px "offset" on its own).
        if g is None or not np.isfinite(g) or g > g_bright or g < g_saturate:
            continue
        cx, cy = int(round(xi)), int(round(yi))
        if not (box <= cx < nx - box and box <= cy < ny - box):
            continue
        # isolation: no other star brighter than g+1.5 within 3 pixels
        d2 = (x - xi) ** 2 + (y - yi) ** 2
        if np.any((d2 > 0) & (d2 < 9) & (np.nan_to_num(gmag.astype(float), nan=99) < g + 1.5)):
            continue
        cut = stack[cy - box:cy + box + 1, cx - box:cx + box + 1].astype(float)
        if not np.isfinite(cut).all():
            continue  # saturated/masked core: centroid would be biased
        # Subtract a plane fitted to the box border, not a flat median: a bright neighbour's
        # halo (R Dra) puts a slope under the star and drags a plain centroid toward it.
        yy, xx = np.mgrid[-box:box + 1, -box:box + 1]
        edge = (np.abs(xx) == box) | (np.abs(yy) == box)
        A = np.c_[np.ones(edge.sum()), xx[edge], yy[edge]]
        coef, *_ = np.linalg.lstsq(A, cut[edge], rcond=None)
        cut = cut - (coef[0] + coef[1] * xx + coef[2] * yy)
        cut[cut < 0] = 0
        tot = cut.sum()
        if tot <= 0:
            continue
        offs.append(((cx + (xx * cut).sum() / tot) - xi, (cy + (yy * cut).sum() / tot) - yi))
    if not offs:
        return {"n": 0, "reliable": False}
    o = np.array(offs)
    return {"n": len(o), "median_dx_px": round(float(np.median(o[:, 0])), 3),
            "median_dy_px": round(float(np.median(o[:, 1])), 3),
            "median_abs_px": round(float(np.median(np.hypot(o[:, 0], o[:, 1]))), 3),
            "reliable": len(o) >= 3}
