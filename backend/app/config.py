"""All archive details and tunables live here. Override any field in backend/.env.

Adding a new data release (e.g. DR2) = put its collection name first in
SPHEREX_COLLECTIONS, e.g.  SPHEREX_COLLECTIONS='["spherex_dr2","spherex_dr1","spherex_qr3"]'
"""
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # --- Archive -------------------------------------------------------------
    # Newest first: when one exposure is in several releases, the first one wins.
    spherex_collections: list[str] = ["spherex_dr1", "spherex_qr3"]
    include_deep: bool = False  # also query *_deep collections (near the ecliptic poles)
    sia_radius_arcsec: float = 2.0
    simbad_servers: list[str] = ["simbad.cds.unistra.fr", "simbad.harvard.edu"]

    # Where pixel data is read from. Both serve the same files and support HTTP
    # range requests, so we read only the rows we need (see fetcher.py).
    # "irsa": HTTP multi-range reads (one request returns just the pixels we need; ~6x
    #         fewer bytes; best from home). "s3": full-row reads of the AWS copy (S3 has no
    #         multi-range; best when the server runs in AWS us-east-1). Other = fallback.
    data_source: str = "irsa"
    s3_base: str = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/"
    irsa_base: str = "https://irsa.ipac.caltech.edu"
    # IRSA's server-side cutout service. Off by default: on 2026-11-14 it returned
    # 503 for most parallel requests and ~60 s for the rest.
    use_cutout_service_fallback: bool = False

    # --- HTTP ----------------------------------------------------------------
    max_parallel_requests: int = 12  # concurrent HTTP requests across the whole app
    max_parallel_images: int = 12  # images in flight per job (keeps the pipe full, first visit lands early)
    http2: bool = False  # IRSA reset HTTP/2 streams under load; HTTP/1.1 was faster
    request_timeout_s: float = 60.0
    retries: int = 3

    # --- What to read from each file -----------------------------------------
    fetch_zodi: bool = False  # else: subtract a sigma-clipped median background
    # FLAGS: "box" = only rows within flags_box_px of the target (photometry + the
    # contamination panel; median stacks reject cosmic rays elsewhere), "full", "none".
    flags_mode: str = "box"
    flags_box_px: int = 4  # covers the photometry aperture; the annulus is sigma-clipped
    # Flag planes treated as bad pixels (names come from the FLAGS header).
    # Deliberately NOT masked: SOURCE (that's the stars!), FULLSAMPLE, DICHROIC.
    mask_flags: list[str] = [
        "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA",
        "HOT", "COLD", "NONLINEAR", "PERSIST", "OUTLIER", "STREAK", "SNOWBALL",
    ]

    # --- Limits --------------------------------------------------------------
    fov_min_deg: float = 0.01
    fov_max_deg: float = 0.5  # 30'; bigger = more rows per image
    fov_default_deg: float = 5 / 60
    max_images_per_job: int = 600
    max_concurrent_jobs: int = 2  # further jobs queue (protects IRSA/S3 and server memory)
    frames_max_age_days: float = 30.0
    visit_gap_days: float = 30.0
    pixel_arcsec: float = 6.2  # output grid (SPHEREx native is ~6.15")
    target_epoch: float = 2025.85  # proper-motion epoch for centring (mid DR1)

    # --- Gaia overlay ------------------------------------------------------------
    gaia_providers: list[str] = ["esa", "mast", "vizier"]  # tried in order
    gaia_timeout_s: float = 30.0
    wise_providers: list[str] = ["irsa_catwise", "vizier_allwise"]
    gaia_gmax: float = 18.0  # SPHEREx 6" pixels: fainter stars are clutter
    gaia_max_sources: int = 400

    # --- AI guide (Google Gemini) ----------------------------------------------
    # The key is read from backend/.env or the environment; it is never sent to browsers.
    gemini_api_key: SecretStr | None = None
    ai_models: list[str] = ["gemini-3.5-flash", "gemini-2.5-flash"]  # tried in order
    ai_timeout_s: float = 45.0
    ai_max_output_tokens: int = 1200
    ai_per_client_per_10min: int = 20
    ai_global_per_hour: int = 400

    # --- Cache ---------------------------------------------------------------
    cache_dir: Path = BACKEND_DIR / "cache"
    cache_size_gb: float = 20.0
    metadata_ttl_s: int = 24 * 3600
    process_workers: int = 0  # 0 = min(4, cores - 1)


settings = Settings()
