"""Stage B: which SPHEREx images contain this point? (IRSA SIA, cached per sky tile)."""
import asyncio
import json
import logging
import re
import time

import astropy.units as u
from astropy.coordinates import SkyCoord

from .cache import disk_cache, tile_key
from .config import settings

log = logging.getLogger(__name__)

_KEEP = ["obs_id", "access_url", "t_min", "t_max", "obs_collection", "energy_bandpassname",
         "em_min", "em_max", "s_ra", "s_dec", "cloud_access", "dataproduct_subtype"]


def _clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def normalise(raw: dict) -> dict | None:
    """Turn one SIA row into a plain, JSON-safe dict with a parsed detector number."""
    r = {k: _clean(raw[k]) for k in _KEEP if k in raw}
    if r.get("dataproduct_subtype", "science") not in ("science", None, ""):
        return None
    # Verified 2026-11-14: obs_id is per *exposure* (shared by all 6 detectors);
    # the detector is in energy_bandpassname, e.g. "SPHEREx-D3".
    m = re.search(r"D(\d)", str(r.get("energy_bandpassname", "")))
    if not m:
        m = re.search(r"_\d+D(\d)_", str(r.get("access_url", "")))
    r["detector"] = int(m.group(1)) if m else None
    r["s3_key"] = None
    try:
        r["s3_key"] = json.loads(r.get("cloud_access") or "{}")["aws"]["key"]
    except (ValueError, KeyError, TypeError):
        pass
    r.pop("cloud_access", None)
    for k in ("t_min", "t_max", "em_min", "em_max", "s_ra", "s_dec"):
        if r.get(k) is not None:
            r[k] = float(r[k])
    r["uid"] = f"{r.get('obs_id')}_D{r['detector']}"
    return r


def _sia_one(ra: float, dec: float, collection: str) -> list[dict]:
    from astroquery.ipac.irsa import Irsa

    t = Irsa.query_sia(pos=(SkyCoord(ra, dec, unit="deg"), settings.sia_radius_arcsec * u.arcsec),
                       collection=collection)
    cols = [c for c in _KEEP if c in t.colnames]
    return [dict(zip(cols, row)) for row in t[cols]] if len(t) else []


def merge(parts: list[list[dict]]) -> list[dict]:
    """parts are in collection-priority order; first occurrence of (exposure, detector) wins."""
    rows, seen = [], set()
    for part in parts:
        for raw in part:
            r = normalise(raw)
            if r is None or r["uid"] in seen:
                continue
            seen.add(r["uid"])
            rows.append(r)
    rows.sort(key=lambda r: r["t_min"])
    return rows


def collections() -> list[str]:
    cols = list(settings.spherex_collections)
    if settings.include_deep:
        cols += [c + "_deep" for c in settings.spherex_collections]
    return cols


async def query_images(ra: float, dec: float) -> list[dict]:
    key = ("meta", tile_key(ra, dec, arcmin=0.05), tuple(collections()))  # 3" tiles
    if (hit := disk_cache.get(key)) is not None:
        return hit
    t0 = time.perf_counter()
    cols = collections()
    parts = await asyncio.gather(*[asyncio.to_thread(_sia_one, ra, dec, c) for c in cols],
                                 return_exceptions=True)
    ok = []
    for c, p in zip(cols, parts):
        if isinstance(p, Exception):
            log.warning("SIA failed for %s: %s", c, p)
        else:
            ok.append(p)
    if not ok:
        raise RuntimeError("all SIA queries failed")
    rows = merge(ok)
    if len(ok) == len(cols):  # only cache complete answers
        disk_cache.set(key, rows, expire=settings.metadata_ttl_s)
    log.info("stage=metadata ra=%.5f dec=%.5f n=%d secs=%.2f", ra, dec, len(rows),
             time.perf_counter() - t0)
    return rows
