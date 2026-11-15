"""Name -> coordinates, moved to the SPHEREx epoch using proper motion.

Catalogue positions are for J2000 (SIMBAD) or 2016.0 (Gaia). Barnard's Star moves
~10"/yr, so without this correction a small cutout can miss it entirely.
"""
import logging
import math

import astropy.units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time

from .cache import disk_cache
from .config import settings

log = logging.getLogger(__name__)

# Fallback when SIMBAD is unreachable. Values are approximate (good to a few ").
# (ra, dec, ref_epoch, pmra mas/yr, pmdec mas/yr, parallax mas)
FALLBACK = {
    "barnard's star": (269.448503, 4.739420, 2016.0, -801.55, 10362.39, 546.98),
    "wise j085510.83-071442.5": (133.795129, -7.245147, 2010.5, -8118.0, 673.0, 439.0),
    "eps ind b": (331.0438, -56.7827, 2000.0, 3982.0, -2466.8, 274.8),
}


def move_to_epoch(ra, dec, pmra, pmdec, ref_epoch, target_epoch, parallax_mas=None):
    """Shift (ra, dec) at ref_epoch to target_epoch. PMs in mas/yr (pmra includes cos dec)."""
    if not pmra and not pmdec:
        return ra, dec
    dist = None
    if parallax_mas and parallax_mas > 0:
        dist = Distance(parallax=parallax_mas * u.mas)
    c = SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg,
        pm_ra_cosdec=(pmra or 0) * u.mas / u.yr, pm_dec=(pmdec or 0) * u.mas / u.yr,
        distance=dist, obstime=Time(ref_epoch, format="jyear"),
    )
    new = c.apply_space_motion(new_obstime=Time(target_epoch, format="jyear"))
    return float(new.ra.deg), float(new.dec.deg)


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _simbad(name: str) -> dict | None:
    from astroquery.simbad import Simbad, conf

    last = None
    for host in settings.simbad_servers:  # CDS TAP was returning 404 on 2026-11-14
        try:
            with conf.set_temp("server", host):
                s = Simbad()
                s.add_votable_fields("pmra", "pmdec", "plx_value")
                t = s.query_object(name)
            break
        except Exception as e:
            log.warning("simbad %s failed for %r: %s", host, name, str(e)[:120])
            last = e
    else:
        raise last or RuntimeError("no SIMBAD server")
    if t is None or len(t) == 0:
        return None
    r = t[0]
    cols = {c.lower(): c for c in t.colnames}
    get = lambda k: _num(r[cols[k]]) if k in cols else None  # noqa: E731
    return {
        "main_id": str(r[cols["main_id"]]) if "main_id" in cols else name,
        "ra_cat": get("ra"), "dec_cat": get("dec"), "ref_epoch": 2000.0,
        "pmra": get("pmra"), "pmdec": get("pmdec"), "parallax": get("plx_value"),
        "source": "simbad",
    }


def resolve(name: str, target_epoch: float | None = None) -> dict:
    """Blocking. Returns catalogue + epoch-corrected position. Cached forever on disk."""
    target_epoch = target_epoch or settings.target_epoch
    key = ("resolve", name.strip().lower(), target_epoch)
    if (hit := disk_cache.get(key)) is not None:
        return hit
    info = None
    try:
        info = _simbad(name)
    except Exception as e:  # network or parse error: fall through to fallbacks
        log.warning("simbad failed for %r: %s", name, e)
    if info is None and name.strip().lower() in FALLBACK:
        ra, dec, ep, pmra, pmdec, plx = FALLBACK[name.strip().lower()]
        info = {"main_id": name, "ra_cat": ra, "dec_cat": dec, "ref_epoch": ep,
                "pmra": pmra, "pmdec": pmdec, "parallax": plx, "source": "fallback"}
    if info is None:
        c = SkyCoord.from_name(name)  # CDS Sesame; raises if unknown
        info = {"main_id": name, "ra_cat": float(c.ra.deg), "dec_cat": float(c.dec.deg),
                "ref_epoch": 2000.0, "pmra": None, "pmdec": None, "parallax": None,
                "source": "sesame"}
    ra, dec = move_to_epoch(info["ra_cat"], info["dec_cat"], info["pmra"], info["pmdec"],
                            info["ref_epoch"], target_epoch, info["parallax"])
    out = {"name": name, **info, "ra": ra, "dec": dec, "epoch": target_epoch}
    # SIMBAD answers never change; a fallback answer is retried after an hour.
    disk_cache.set(key, out, expire=None if info["source"] != "fallback" else 3600)
    return out
