"""Cache layer.

L1: functools.lru_cache in the modules that need it (name lookups).
L2: diskcache (SQLite) for metadata (24 h), file layouts and cutouts (forever).
L3: rendered PNGs under <cache_dir>/frames/<job_key>/.
"""
import diskcache

from .config import settings

settings.cache_dir.mkdir(parents=True, exist_ok=True)
disk_cache = diskcache.Cache(
    str(settings.cache_dir / "l2"), size_limit=int(settings.cache_size_gb * 1e9)
)
FRAMES_DIR = settings.cache_dir / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def snap(value: float, step_deg: float) -> float:
    return round(value / step_deg) * step_deg


def tile_key(ra: float, dec: float, arcmin: float = 1.0) -> str:
    step = arcmin / 60
    return f"{snap(ra, step):.4f},{snap(dec, step):.4f}"
