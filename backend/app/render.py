"""Stage G: arrays -> PNG with ONE stretch per sequence and band (otherwise noise
'flickers' and looks like variability). Stretch parameters are plain numbers so they
can be stored in the job manifest and reused for frames rendered later on demand."""
import io
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def stretch_params(ref: np.ndarray) -> list[float]:
    """[lo, hi] from the 1st-99.5th percentile of a reference frame."""
    finite = ref[np.isfinite(ref)]
    if not finite.size:
        return [0.0, 1.0]
    lo, hi = np.percentile(finite, [1, 99.5])
    return [float(lo), float(hi if hi > lo else lo + 1e-6)]


def fill_nans(a: np.ndarray, max_passes: int = 40) -> np.ndarray:
    """For DISPLAY only: fill masked pixels from their valid neighbours.

    Only masked regions fully enclosed by data are filled (a saturated star core is ringed
    by bright pixels, so it fills in bright; a lone masked sky pixel fills in as sky).
    Regions touching the image border are left alone: that's outside the data footprint.
    (Drawing NaN as a flat colour gave black holes in bright stars, e.g. R Dra's ~10 px
    core, and single pure-red/green pixels in the RGB view.)"""
    from scipy import ndimage

    a = np.array(a, dtype=np.float32, copy=True)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    lab, n = ndimage.label(bad)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    todo = bad & ~np.isin(lab, list(edge))
    for _ in range(max_passes):
        if not todo.any():
            break
        p = np.pad(a, 1, constant_values=np.nan)
        nb = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        cnt = np.isfinite(nb).sum(axis=0)
        fill = todo & (cnt >= 2)
        if not fill.any():
            break
        a[fill] = (np.nansum(nb, axis=0) / np.maximum(cnt, 1))[fill]
        todo &= ~fill
    return a


def apply_stretch(a: np.ndarray, p: list[float]) -> np.ndarray:
    """asinh stretch -> uint8, masked pixels filled from neighbours (display only)."""
    a = fill_nans(a)
    lo, hi = p
    x = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    x = np.arcsinh(10 * x) / np.arcsinh(10)
    return np.nan_to_num(x * 255, nan=0).astype(np.uint8)


def diff_limit(ref: np.ndarray) -> float:
    finite = ref[np.isfinite(ref)]
    if not finite.size:
        return 1.0
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(max(8 * 1.4826 * mad, 1e-6))


def apply_diff(a: np.ndarray, lim: float) -> np.ndarray:
    """Symmetric around 0: grey = no change, white = brighter/new, black = fainter/gone."""
    x = np.clip(fill_nans(a) / lim, -1, 1)
    return np.nan_to_num((x + 1) * 127.5, nan=127.5).astype(np.uint8)


# Backwards-compatible helpers (used by tests and older code)
def make_stretch(ref):
    p = stretch_params(ref)
    return lambda a: apply_stretch(a, p)


def make_diff_stretch(ref):
    lim = diff_limit(ref)
    return lambda a: apply_diff(a, lim)


def _image(u8: np.ndarray, scale: int) -> Image.Image:
    im = Image.fromarray(u8[::-1])  # FITS origin is bottom-left
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def _save(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.png")
    im.save(tmp, optimize=True)
    os.replace(tmp, path)  # atomic: never serve a half-written PNG


def to_png(arr: np.ndarray, stretch, path: Path, scale: int = 4) -> None:
    _save(_image(stretch(arr), scale), path)


def to_png_rgb(r, g, b, params: dict, path: Path, scale: int = 4) -> None:
    """False-colour composite. params: {"r": [lo,hi], "g": ..., "b": ...}."""
    chans = [apply_stretch(c, params[k]) if c is not None else None
             for c, k in ((r, "r"), (g, "g"), (b, "b"))]
    shape = next(c.shape for c in chans if c is not None)
    chans = [c if c is not None else np.zeros(shape, np.uint8) for c in chans]
    rgb = np.dstack(chans)[::-1]
    im = Image.fromarray(rgb, "RGB")
    _save(im.resize((im.width * scale, im.height * scale), Image.NEAREST), path)


def png_bytes(u8_or_rgb: np.ndarray, scale: int = 4) -> bytes:
    im = Image.fromarray(u8_or_rgb[::-1])
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def gif_bytes(paths: list[Path], labels: list[str], ms: int) -> bytes:
    """Animated GIF from rendered PNGs, with a small date label on each frame."""
    from PIL import ImageDraw

    frames = []
    for p, label in zip(paths, labels):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 255))
        frames.append(im.convert("P", palette=Image.ADAPTIVE))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, optimize=True)
    return buf.getvalue()
