"""WISE overlay: CatWISE2020 (IRSA; positions + proper motions, epoch 2015.4) or AllWISE
(VizieR Harvard mirror; epoch ~2010.5). WISE sees 3.4 and 4.6 um, the same wavelengths as
SPHEREx detectors D4-D6, so it's the natural comparison for red and cold objects.

Services on 2026-11-15: IRSA TAP returned 502/timeouts; VizieR CDS 404; VizieR Harvard
answered AllWISE in ~3 s but has no CatWISE table. Providers are tried in order.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

import astroquery.ipac.irsa  # noqa: F401  (import on the main thread; see gaia.py)
import astroquery.vizier  # noqa: F401

from ..cache import disk_cache, tile_key
from ..config import settings

log = logging.getLogger(__name__)


def _f(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _catwise(ra, dec, r):
    from astroquery.ipac.irsa import Irsa

    q = ("SELECT source_name, ra, dec, pmra, pmdec, w1mpro, w2mpro FROM catwise_2020 WHERE "
         f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1")
    t = Irsa.query_tap(q).to_table()
    # CatWISE proper motions are in arcsec/yr
    return 2015.4, [{"id": str(x["source_name"]), "ra": _f(x["ra"]), "dec": _f(x["dec"]),
                     "pmra": (_f(x["pmra"]) or 0) * 1000, "pmdec": (_f(x["pmdec"]) or 0) * 1000,
                     "w1": _f(x["w1mpro"]), "w2": _f(x["w2mpro"])} for x in t]


def _allwise(ra, dec, r):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    v = Vizier(columns=["AllWISE", "RAJ2000", "DEJ2000", "W1mag", "W2mag", "pmRA", "pmDE"],
               row_limit=-1, timeout=settings.gaia_timeout_s)
    v.VIZIER_SERVER = "vizier.cfa.harvard.edu"
    res = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=r * u.deg, catalog="II/328/allwise")
    if not len(res):
        return 2010.5, []
    t = res[0]
    get = lambda x, c: _f(x[c]) if c in t.colnames else None  # noqa: E731
    # AllWISE proper motions are noisy (short baseline); show them but don't move circles
    return 2010.5, [{"id": str(x["AllWISE"]), "ra": get(x, "RAJ2000"), "dec": get(x, "DEJ2000"),
                     "pmra": 0.0, "pmdec": 0.0, "w1": get(x, "W1mag"), "w2": get(x, "W2mag")}
                    for x in t]


PROVIDERS = {"irsa_catwise": _catwise, "vizier_allwise": _allwise}


async def wise_sources(ra: float, dec: float, radius_deg: float) -> dict:
    key = ("wise", tile_key(ra, dec, arcmin=0.05), round(radius_deg, 4))
    if (hit := disk_cache.get(key)) is not None:
        return hit
    errors = {}
    for name in settings.wise_providers:
        t0 = time.perf_counter()
        try:
            epoch, rows = await asyncio.wait_for(
                asyncio.to_thread(PROVIDERS[name], ra, dec, radius_deg),
                timeout=settings.gaia_timeout_s)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {str(e)[:120]}"
            log.warning("wise provider %s failed: %s", name, errors[name])
            continue
        rows = [r for r in rows if r["ra"] is not None and r["dec"] is not None]
        rows.sort(key=lambda s: s["w1"] if s["w1"] is not None else 99)
        out = {"provider": name, "epoch": epoch, "sources": rows[: settings.gaia_max_sources],
               "secs": round(time.perf_counter() - t0, 2)}
        disk_cache.set(key, out, expire=30 * 24 * 3600)
        return out
    raise RuntimeError(f"all WISE providers failed: {errors}")
