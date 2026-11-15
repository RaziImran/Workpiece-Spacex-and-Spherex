"""Stages D-F: open cutout -> clean -> reproject; photometry; group by visit; stack.

`prepare()` is CPU-bound and runs in a ProcessPoolExecutor, so it takes and returns
only plain picklable things (bytes, str, numpy arrays, dicts).
"""
from __future__ import annotations

import io
import math
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_area

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.filterwarnings("ignore", message=r"(?s).*SIP distortion")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"(?s).*(All-NaN|Mean of empty|invalid value)")

# Used only if a FLAGS header carries no MSKN/MSKM definitions.
DEFAULT_BAD_BITS = sum(1 << b for b in (0, 1, 2, 6, 9, 10, 11, 15, 17, 19, 25, 27))


def target_wcs(ra: float, dec: float, fov_deg: float, pix_arcsec: float):
    n = int(math.ceil(fov_deg * 3600 / pix_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    return w, (n, n)


def mask_bits(flags_header: fits.Header, names: list[str]) -> int:
    wanted = {n.upper() for n in names}
    bits, found = 0, False
    for key in flags_header:
        if key.startswith("MSKN"):
            found = True
            if str(flags_header[key]).strip().upper() in wanted:
                bits |= int(flags_header.get("MSKM" + key[4:], 1 << int(key[4:])))
    return bits if found else DEFAULT_BAD_BITS


def measure(img: np.ndarray, w: WCS, hdul, ra: float, dec: float,
            r_ap=2.0, r_in=4.0, r_out=7.0) -> dict | None:
    """Aperture photometry of the target in native pixels. Flux in mJy (no aperture correction)."""
    from astropy.coordinates import SkyCoord
    from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture,
                                    aperture_photometry)

    x, y = (float(v) for v in w.world_to_pixel(SkyCoord(ra, dec, unit="deg")))
    ny, nx = img.shape
    if not (r_out <= x < nx - r_out and r_out <= y < ny - r_out):
        return None
    bad = ~np.isfinite(img)
    ap, ann = CircularAperture((x, y), r_ap), CircularAnnulus((x, y), r_in, r_out)
    from astropy.stats import SigmaClip
    # sigma-clipped: FLAGS are only read near the target, so clip stars/cosmic rays here
    st = ApertureStats(img, ann, mask=bad, sigma_clip=SigmaClip(sigma=3.0, maxiters=5))
    bkg, sig = float(st.median), float(st.std)
    n_ann = float(getattr(st.sum_aper_area, "value", st.sum_aper_area))
    tab = aperture_photometry(img - bkg, ap, mask=bad)
    n_ap = float(ap.area_overlap(img, mask=bad))
    if n_ap < 0.7 * ap.area or not math.isfinite(bkg):
        # Saturated core (OVERFLOW/SUR_ERROR): no calibrated flux. For a light curve we
        # still measure the unsaturated PSF wings (relative brightness, per detector).
        wave, bw = wavelength_at(hdul, x, y)
        return {"x": x, "y": y, "flux_mjy": None, "err_mjy": None, "wavelength_um": wave,
                "bandwidth_um": bw, "bkg_mjysr": bkg, "flag": "core_masked",
                "wings_rel": wing_flux(img, w, x, y)}
    omega_sr = proj_plane_pixel_area(w) * (math.pi / 180) ** 2  # deg^2 -> sr
    to_mjy = omega_sr * 1e9  # MJy/sr * sr -> MJy -> mJy
    flux = float(tab["aperture_sum"][0]) * to_mjy
    err = sig * math.sqrt(n_ap * (1 + n_ap / max(n_ann, 1))) * to_mjy
    wave, bw = wavelength_at(hdul, x, y)
    return {"x": x, "y": y, "flux_mjy": flux, "err_mjy": err, "wavelength_um": wave,
            "bandwidth_um": bw, "bkg_mjysr": bkg, "flag": None}


def wavelength_at(hdul, x: float, y: float):
    """(wavelength, bandwidth) in um at a native pixel, from the WAVE-TAB lookup WCS."""
    try:
        ww = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
        wv, bwv = ww.pixel_to_world_values(x, y)
        return float(wv), float(bwv)
    except Exception:
        return None, None


def flag_names(flags_header: fits.Header) -> dict[int, str]:
    return {int(flags_header.get("MSKM" + k[4:], 1 << int(k[4:]))): str(flags_header[k]).strip()
            for k in flags_header if k.startswith("MSKN")}


def flag_stats(hdul, x: float, y: float, r: float = 2.0) -> dict:
    """Contamination flags near the target: fraction of pixels per flag plane in the rows
    we read FLAGS for, and which planes are set inside the photometry aperture."""
    if "FLAGS" not in hdul:
        return {}
    fl = hdul["FLAGS"].data
    y0 = int(hdul[0].header.get("FLAG_Y0", 0))
    y1 = int(hdul[0].header.get("FLAG_Y1", fl.shape[0]))
    if y1 <= y0:
        return {}
    box = fl[y0:y1].astype(np.int64)
    names = flag_names(hdul["FLAGS"].header)
    frac = {}
    for bit, name in names.items():
        f = float(np.mean((box & bit) != 0))
        if f > 0:
            frac[name] = round(f, 4)
    yy, xx = np.mgrid[0:fl.shape[0], 0:fl.shape[1]]
    inside = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & (yy >= y0) & (yy < y1)
    at = int(np.bitwise_or.reduce(fl[inside].astype(np.int64))) if inside.any() else 0
    return {"frac": frac, "at_target": [n for b, n in sorted(names.items()) if at & b]}


def despike(img: np.ndarray, nsigma: float = 5.0, ratio: float = 0.25) -> np.ndarray:
    """Mask single-pixel spikes (cosmic rays, hot pixels) in a native frame.

    SPHEREx stars always spill light into neighbouring pixels (Barnard's Star: core 1034,
    neighbours 801/766/481); a cosmic ray or hot pixel is a lone spike. A pixel is masked
    when it is > nsigma above the sky AND its brightest neighbour holds < ratio of its
    excess. Needed because FLAGS are only read near the target (flags_mode="box")."""
    finite = img[np.isfinite(img)]
    if finite.size < 25:
        return img
    med = float(np.median(finite))
    sig = float(1.4826 * np.median(np.abs(finite - med))) or 1e-12
    ex = img - med
    p = np.pad(ex, 1, constant_values=np.nan)
    h, w = img.shape
    nb = np.stack([p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    with np.errstate(invalid="ignore"):
        nbmax = np.nanmax(np.where(np.isfinite(nb), nb, -np.inf), axis=0)
        spike = (ex > nsigma * sig) & (nbmax < ratio * ex)
    out = img.copy()
    out[spike] = np.nan
    return out


def clean(hdul, mask_names: list[str]):
    """IMAGE -> background-subtracted float32 with bad pixels as NaN, plus its WCS."""
    hdr = hdul["IMAGE"].header
    img = hdul["IMAGE"].data.astype(np.float32)
    bad = ~np.isfinite(img)
    if "FLAGS" in hdul:
        bits = mask_bits(hdul["FLAGS"].header, mask_names)
        bad |= (hdul["FLAGS"].data.astype(np.int64) & bits) != 0
    img[bad] = np.nan
    img = despike(img)
    if "ZODI" in hdul:
        img -= hdul["ZODI"].data.astype(np.float32)
    else:  # zodi is smooth on these scales: a robust median does the job
        _, med, _ = sigma_clipped_stats(img, sigma=3.0, maxiters=3)
        img -= np.float32(med)
    return img, WCS(hdr, naxis=2)


def _t_mid(hdr) -> float:
    if "MJD-AVG" in hdr:
        return float(hdr["MJD-AVG"])
    return float(0.5 * (hdr.get("MJD-OBS", np.nan) + hdr.get("MJD-END", hdr.get("MJD-OBS", np.nan))))


def wing_flux(img: np.ndarray, w, x: float, y: float) -> float | None:
    """Flux in the unsaturated wings of a saturated star, minus a sky annulus further out,
    scaled for masked pixels. The ring starts where < 10% of pixels are masked: R Dra
    (K ~ -1) is masked out to 5-9 px, Barnard's Star to ~2 px. Units: mJy-like but NOT
    calibrated (PSF fraction in the ring unknown); fine for a per-detector light curve."""
    from astropy.stats import SigmaClip
    from photutils.aperture import ApertureStats, CircularAnnulus

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    r_in = 2.5
    while r_in < 15 and np.mean(~np.isfinite(img[(rr >= r_in) & (rr < r_in + 1)])) > 0.1:
        r_in += 1
    r_out, b_in = r_in + 3, r_in + 6
    b_out = b_in + 4
    if not (b_out <= x < nx - b_out and b_out <= y < ny - b_out):
        return None
    bad = ~np.isfinite(img)
    sky = ApertureStats(img, CircularAnnulus((x, y), b_in, b_out), mask=bad,
                        sigma_clip=SigmaClip(sigma=3.0, maxiters=5)).median
    wing = CircularAnnulus((x, y), r_in, r_out)
    st = ApertureStats(img - sky, wing, mask=bad, sigma_clip=None)
    valid = float(getattr(st.sum_aper_area, "value", st.sum_aper_area))
    if valid < 0.5 * wing.area or not np.isfinite(st.sum):
        return None
    omega_sr = proj_plane_pixel_area(w) * (math.pi / 180) ** 2
    return float(st.sum) * wing.area / valid * omega_sr * 1e9


def prepare(data: bytes, out_header: str, shape: tuple[int, int], ra: float, dec: float,
            mask_names: list[str]) -> dict:
    """Runs in a worker process: reprojected float32 frame + photometry + metadata."""
    from astropy.coordinates import SkyCoord
    from reproject import reproject_interp

    with fits.open(io.BytesIO(data), memmap=False) as hdul:
        hdr = hdul["IMAGE"].header
        img, w = clean(hdul, mask_names)
        phot = measure(img, w, hdul, ra, dec)
        x, y = (float(v) for v in w.world_to_pixel(SkyCoord(ra, dec, unit="deg")))
        wave, bw = wavelength_at(hdul, x, y)
        out_wcs = WCS(fits.Header.fromstring(out_header))
        arr, foot = reproject_interp((img, w), out_wcs, shape_out=tuple(shape), order="bilinear")
        arr[foot == 0] = np.nan
        return {"array": arr.astype(np.float32), "t_mid": _t_mid(hdr),
                "detector": int(hdr.get("DETECTOR", 0)), "phot": phot,
                "wavelength_um": wave, "bandwidth_um": bw,
                "flags": flag_stats(hdul, x, y),
                "frac_masked": round(float(np.mean(~np.isfinite(img))), 4)}


def reproject_only(data: bytes, out_header: str, shape, mask_names: list[str]) -> np.ndarray:
    """One exposure on the job grid (for the single-exposure flipbook, on demand)."""
    from reproject import reproject_interp

    with fits.open(io.BytesIO(data), memmap=False) as hdul:
        img, w = clean(hdul, mask_names)
        arr, foot = reproject_interp((img, w), WCS(fits.Header.fromstring(out_header)),
                                     shape_out=tuple(shape), order="bilinear")
        arr[foot == 0] = np.nan
        return arr.astype(np.float32)


def measure_batch(items: list[tuple[str, bytes]], ra: float, dec: float,
                  mask_names: list[str]) -> list[dict]:
    """Photometry of ANY position in many cached cutouts (spectrum of a clicked star)."""
    out = []
    for uid, data in items:
        try:
            with fits.open(io.BytesIO(data), memmap=False) as hdul:
                hdr = hdul["IMAGE"].header
                img, w = clean(hdul, mask_names)
                ph = measure(img, w, hdul, ra, dec)
                if ph is None:
                    continue
                out.append({"uid": uid, "t_mid": _t_mid(hdr),
                            "detector": int(hdr.get("DETECTOR", 0)), **ph})
        except Exception:
            continue
    return out


def group_by_visit(rows: list[dict], gap_days: float) -> list[list[dict]]:
    groups, cur, last = [], [], None
    for r in sorted(rows, key=lambda r: r["t_min"]):
        if last is not None and r["t_min"] - last > gap_days:
            groups.append(cur)
            cur = []
        cur.append(r)
        last = r["t_min"]
    if cur:
        groups.append(cur)
    return groups


def stack(arrays: list[np.ndarray]) -> np.ndarray:
    return np.nanmedian(np.stack(arrays), axis=0).astype(np.float32)


def evenly_sample(rows: list[dict], n: int, gap_days: float = 30.0) -> list[dict]:
    """Cap the number of images while keeping every visit AND every detector covered:
    round-robin over (visit, detector) groups, each group sampled evenly in time.
    (Sampling purely by time left some visits without whole detectors.)"""
    if len(rows) <= n:
        return rows
    groups: dict[tuple, list[dict]] = {}
    for k, g in enumerate(group_by_visit(rows, gap_days)):
        for r in g:
            groups.setdefault((k, r.get("detector")), []).append(r)
    queues = []
    for key in sorted(groups, key=lambda x: (x[0], x[1] or 0)):
        g = groups[key]
        order = np.linspace(0, len(g) - 1, len(g)).round().astype(int)
        # spread picks across the group: middle, ends, then fill in
        idx = list(dict.fromkeys([len(g) // 2, 0, len(g) - 1] + list(order)))
        queues.append([g[i] for i in idx])
    out, i = [], 0
    while len(out) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return sorted(out, key=lambda r: r["t_min"])
