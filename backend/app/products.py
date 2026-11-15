"""Image products of a job: per-visit stacks for every band, difference frames, a deep
stack of all visits, and a false-colour composite.

Bands. SPHEREx has 6 detectors; each covers a slice of 0.75-5 um through a linear
variable filter (102 spectral channels in total). Stacking each detector separately
gives 6 broad "colours" with good signal-to-noise:
    D1 0.75-1.09  D2 1.10-1.62  D3 1.63-2.41  D4 2.42-3.82  D5 3.83-4.41  D6 4.42-5.00 um
"all" stacks every detector together (deepest), and "rgb" maps short->blue, long->red:
    blue = D1+D2, green = D3+D4, red = D5+D6
so very cold objects (brown dwarfs like WISE 0855) stand out as red.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from . import processing
from .render import apply_diff, apply_stretch, diff_limit, stretch_params, to_png, to_png_rgb

DET_BANDS = [f"D{d}" for d in range(1, 7)]
BANDS = ["all"] + DET_BANDS
RGB = {"b": ("D1", "D2"), "g": ("D3", "D4"), "r": ("D5", "D6")}
WAVE_RANGES = {"D1": (0.75, 1.09), "D2": (1.10, 1.62), "D3": (1.63, 2.41),
               "D4": (2.42, 3.82), "D5": (3.83, 4.41), "D6": (4.42, 5.00)}


def frame_name(k: int | str, band: str, mode: str) -> str:
    return f"v{k}_{band}_{mode}"


class Products:
    def __init__(self, job_id: str, frame_dir: Path):
        self.job_id = job_id
        self.dir = frame_dir
        self.stacks: dict[tuple[int, str], np.ndarray] = {}
        self.stretch: dict[str, list[float]] = {}   # band -> [lo, hi]
        self.dlim: dict[str, float] = {}            # band -> difference limit
        self.diff_ref: int | None = None

    def url(self, name: str) -> str:
        return f"/api/frame/{self.job_id}/{name}.png"

    def _path(self, name: str, ext="png") -> Path:
        return self.dir / f"{name}.{ext}"

    # ----------------------------------------------------------------- visits
    def add_visit(self, k: int, by_det: dict[int, list[np.ndarray]]) -> dict:
        """Stack one visit for every band; render PNGs. Blocking (run in a thread).
        Returns {band: {"stack": url}} plus "rgb"."""
        self.dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, dict] = {}
        everything = [a for arrs in by_det.values() for a in arrs]
        for band in BANDS:
            arrs = everything if band == "all" else by_det.get(int(band[1]), [])
            if not arrs:
                continue
            s = processing.stack(arrs)
            self.stacks[(k, band)] = s
            np.save(self._path(frame_name(k, band, "stack"), "npy"), s)
            if band not in self.stretch:
                self.stretch[band] = stretch_params(s)
            name = frame_name(k, band, "stack")
            to_png(s, lambda a, p=self.stretch[band]: apply_stretch(a, p), self._path(name))
            out[band] = {"stack": self.url(name), "n": len(arrs)}
        rgb = self._rgb(k, "stack")
        if rgb:
            out["rgb"] = {"stack": rgb}
        return out

    def _channel(self, k, chan, mode):
        arrs = []
        for b in RGB[chan]:
            s = self.stacks.get((k, b))
            if s is None:
                continue
            if mode == "diff":
                ref = self.stacks.get((self.diff_ref, b))
                if ref is None:
                    continue
                s = s - ref
            # put both detectors of a channel on a common scale before averaging
            lo, hi = self.stretch[b]
            arrs.append((s - (lo if mode == "stack" else 0)) / (hi - lo + 1e-12))
        if not arrs:
            return None
        return np.nanmean(np.stack(arrs), axis=0)

    def _rgb(self, k, mode) -> str | None:
        chans = {c: self._channel(k, c, mode) for c in RGB}
        if sum(v is not None for v in chans.values()) < 2:
            return None
        params = {}
        for c, v in chans.items():
            key = f"rgb_{c}_{mode}"
            if key not in self.stretch and v is not None:
                if mode == "stack":
                    # black point at the sky level so the background is black, not tinted
                    f = v[np.isfinite(v)]
                    hi = float(np.percentile(f, 99.5)) if f.size else 1.0
                    lo = float(np.median(f)) if f.size else 0.0
                    self.stretch[key] = [lo, hi if hi > lo else lo + 1e-6]
                else:
                    lim = diff_limit(v)
                    self.stretch[key] = [-lim, lim]
            params[c] = self.stretch.get(key, [0, 1])
        name = frame_name(k, "rgb", mode)
        to_png_rgb(chans["r"], chans["g"], chans["b"], params, self._path(name))
        return self.url(name)

    # ------------------------------------------------------------- differences
    def diffs(self, ref: int, visits: list[int]) -> dict[int, dict]:
        """Difference frames (visit - ref) for every band. Blocking."""
        self.diff_ref = ref
        out: dict[int, dict] = {}
        for k in visits:
            for band in BANDS:
                s, r = self.stacks.get((k, band)), self.stacks.get((ref, band))
                if s is None or r is None:
                    continue
                if band not in self.dlim:
                    self.dlim[band] = diff_limit(r)
                name = frame_name(k, band, "diff")
                to_png(s - r, lambda a, lim=self.dlim[band]: apply_diff(a, lim), self._path(name))
                out.setdefault(k, {})[band] = {"diff": self.url(name)}
            rgb = self._rgb(k, "diff")
            if rgb:
                out.setdefault(k, {})["rgb"] = {"diff": rgb}
        return out

    # ---------------------------------------------------------------- deep
    def deep(self, visits: list[int]) -> dict:
        """Average of all visit stacks per band: the deepest image (Blocking)."""
        out = {}
        for band in BANDS:
            arrs = [self.stacks[(k, band)] for k in visits if (k, band) in self.stacks]
            if not arrs:
                continue
            d = np.nanmean(np.stack(arrs), axis=0)
            self.stacks[("deep", band)] = d
            np.save(self._path(frame_name("deep", band, "stack"), "npy"), d)
            name = frame_name("deep", band, "stack")
            to_png(d, lambda a, p=self.stretch[band]: apply_stretch(a, p), self._path(name))
            out[band] = self.url(name)
        rgb = self._rgb("deep", "stack")
        if rgb:
            out["rgb"] = rgb
        return out


# --------------------------------------------------------- on-demand rendering
def load_stack(frame_dir: Path, k, band) -> np.ndarray | None:
    p = frame_dir / f"{frame_name(k, band, 'stack')}.npy"
    return np.load(p) if p.exists() else None


def render_diff_any(frame_dir: Path, a, b, band: str, lim: float | None) -> np.ndarray | None:
    """uint8 image of stack(a) - stack(b) for any two visits (or 'deep')."""
    sa, sb = load_stack(frame_dir, a, band), load_stack(frame_dir, b, band)
    if sa is None or sb is None:
        return None
    return apply_diff(sa - sb, lim or diff_limit(sb))


async def to_thread(fn, *a):
    return await asyncio.to_thread(fn, *a)
