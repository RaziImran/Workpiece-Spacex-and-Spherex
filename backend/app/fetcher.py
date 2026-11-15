"""Stage C: get a small cutout of each SPHEREx image, fast.

Instead of IRSA's server-side cutout service (slow and often 503 under load, and
every cutout carries a ~4 MB EPSF table), we read the original files with HTTP range
requests. SPHEREx L2 files are 66 MB but laid out simply:

    PRIMARY | IMAGE (2040x2040 float32) | FLAGS (RICE tiles, one per row) |
    VARIANCE | ZODI | EPSF (big table) | WCS-WAVE (tiny table, last)

So per image we fetch: the first 64 KB (headers), the last 23 KB (WCS-WAVE), the
FLAGS header + tile index, then only the rows that cover the field of view.
That is ~1 MB instead of 4 MB, and costs the archive nothing but static file serving.

The result is re-packed as a small MEF in the same layout as an IRSA cutout
(IMAGE, FLAGS[, ZODI], WCS-WAVE), so the rest of the pipeline doesn't care where it
came from. Cutouts are cached forever (archived exposures never change).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import time
from collections import Counter

import astropy.units as u
import httpx
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from tenacity import (AsyncRetrying, retry_if_exception, stop_after_attempt,
                      wait_exponential)

from .cache import disk_cache, snap
from .config import settings

log = logging.getLogger(__name__)
BLOCK = 2880
HEAD_BYTES = 12 * BLOCK  # SPHEREx headers are <= 8 blocks; FLAGS header + tile index ~ 12
STATS: Counter = Counter()  # requests, bytes, cache hits... (read by bench + /api/health)


class NoOverlap(Exception):
    """The requested field doesn't touch this image."""


# --------------------------------------------------------------------------- HTTP
class _Http:
    """One shared AsyncClient + semaphore per event loop (keep-alive matters)."""

    def __init__(self):
        self.loop = None
        self.client: httpx.AsyncClient | None = None
        self.sem: asyncio.Semaphore | None = None

    def get(self):
        loop = asyncio.get_running_loop()
        if self.loop is not loop:
            n = settings.max_parallel_requests
            self.client = httpx.AsyncClient(
                http2=settings.http2,
                timeout=httpx.Timeout(settings.request_timeout_s, connect=15),
                limits=httpx.Limits(max_connections=n * 2, max_keepalive_connections=n),
                headers={"User-Agent": "spherex-viewer/0.1 (educational project)"},
                follow_redirects=True,
            )
            self.sem = asyncio.Semaphore(n)
            self.loop = loop
        return self.client, self.sem

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client, self.loop = None, None


http = _Http()


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.TransportError):
        return True
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (
        408, 429, 500, 502, 503, 504)


class HttpRangeReader:
    """Fetch byte ranges. Tests swap this for an in-memory reader."""

    async def _get(self, url: str, rng: str) -> bytes:
        client, sem = http.get()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception(_retryable), reraise=True,
        ):
            with attempt:
                async with sem:
                    r = await client.get(url, headers={"Range": rng})
                STATS["requests"] += 1
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"server ignored Range header ({r.status_code})")
                STATS["bytes"] += len(r.content)
                return r.content
        raise AssertionError("unreachable")

    async def get(self, url: str, start: int, end: int) -> bytes:  # end inclusive
        return await self._get(url, f"bytes={start}-{end}")

    async def tail(self, url: str, n: int) -> bytes:
        return await self._get(url, f"bytes=-{n}")

    async def get_many(self, url: str, ranges: list[tuple]) -> list[bytes]:
        """Many byte ranges in as few requests as possible (HTTP multi-range).

        ranges: (start, end_inclusive) or (None, n) for the last n bytes.
        Only for servers that support multipart/byteranges (IRSA's Apache does;
        S3 does NOT: it answers 200 with the whole 66 MB file, so we abort on 200).
        """
        chunks = [ranges[i:i + MAX_RANGES_PER_REQUEST]
                  for i in range(0, len(ranges), MAX_RANGES_PER_REQUEST)]
        res = await asyncio.gather(*[self._get_multi(url, c) for c in chunks])
        return [b for part in res for b in part]

    async def _get_multi(self, url: str, ranges: list[tuple]) -> list[bytes]:
        spec = ",".join(f"-{b}" if a is None else f"{a}-{b}" for a, b in ranges)
        client, sem = http.get()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception(_retryable), reraise=True,
        ):
            with attempt:
                async with sem:
                    async with client.stream("GET", url, headers={"Range": "bytes=" + spec}) as r:
                        STATS["requests"] += 1
                        if r.status_code == 200:  # whole file coming: refuse it
                            raise RuntimeError("server ignored multi-range request")
                        r.raise_for_status()
                        body = await r.aread()
                        ctype = r.headers.get("content-type", "")
                        crange = r.headers.get("content-range", "")
                STATS["bytes"] += len(body)
                parts = parse_byteranges(body, ctype, crange)
                return [slice_parts(parts, a, b) for a, b in ranges]
        raise AssertionError("unreachable")


MAX_RANGES_PER_REQUEST = 100  # Apache's MaxRanges default is 200; stay well below


def parse_byteranges(body: bytes, ctype: str, crange: str) -> list[tuple[int, int, int, bytes]]:
    """-> [(start, end_incl, total, data)] from a 206 single- or multipart response."""
    def _cr(v: str):
        rng, total = v.strip().split(" ", 1)[1].split("/")
        a, b = rng.split("-")
        return int(a), int(b), (int(total) if total != "*" else -1)

    if "multipart/byteranges" not in ctype:
        a, b, t = _cr(crange)
        return [(a, b, t, body)]
    boundary = ctype.split("boundary=", 1)[1].split(";")[0].strip().strip('"').encode()
    parts = []
    for chunk in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in chunk:
            continue
        head, data = chunk.split(b"\r\n\r\n", 1)
        cr = [ln for ln in head.decode("latin-1").split("\r\n")
              if ln.lower().startswith("content-range")]
        if not cr:
            continue
        a, b, t = _cr(cr[0].split(":", 1)[1])
        parts.append((a, b, t, data[: b - a + 1]))
    return parts


def slice_parts(parts, a, b) -> bytes:
    """Bytes [a, b] (or the last b bytes if a is None); servers may merge ranges."""
    if a is None:
        total = max(p[2] for p in parts)
        a, b = total - b, total - 1
    for pa, pb, _, data in parts:
        if pa <= a and b <= pb:
            return data[a - pa: b - pa + 1]
    raise RuntimeError(f"range {a}-{b} missing from response")


async def get_many(reader, url, ranges):
    """Use the reader's multi-range support if it has it, else one request per range."""
    if hasattr(reader, "get_many"):
        return await reader.get_many(url, ranges)
    out = []
    for a, b in ranges:
        out.append(await (reader.tail(url, b) if a is None else reader.get(url, a, b)))
    return out


# --------------------------------------------------------------------- FITS layout
def _parse_header(buf: bytes, pos: int):
    """Parse a FITS header starting at buf[pos]. Returns (Header, padded_len) or None."""
    i = pos
    while i + 80 <= len(buf):
        if buf[i:i + 8] == b"END     ":
            raw = buf[pos:i + 80]
            hlen = math.ceil((i + 80 - pos) / BLOCK) * BLOCK
            return fits.Header.fromstring(raw.decode("ascii", "replace")), hlen
        i += 80
    return None


def _data_size(h: fits.Header) -> int:
    naxis = h.get("NAXIS", 0)
    if not naxis:
        return 0
    n = 1
    for k in range(1, naxis + 1):
        n *= h[f"NAXIS{k}"]
    size = abs(h["BITPIX"]) // 8 * h.get("GCOUNT", 1) * (h.get("PCOUNT", 0) + n)
    return math.ceil(size / BLOCK) * BLOCK


def _find_tail_hdu(tail: bytes, extname: str) -> bytes | None:
    """The file is a multiple of 2880 bytes, so HDUs start at len(tail) - k*2880."""
    p = len(tail) - BLOCK
    while p >= 0:
        if tail[p:p + 9] == b"XTENSION=":
            parsed = _parse_header(tail, p)
            if parsed and parsed[0].get("EXTNAME", "").strip() == extname:
                return tail[p:]
        p -= BLOCK
    return None


async def read_layout(reader, url: str, want_zodi: bool) -> dict:
    """Walk the HDU headers with as few requests as possible."""
    head, tail = await asyncio.gather(reader.get(url, 0, HEAD_BYTES - 1), reader.tail(url, 8 * BLOCK))
    needed = {"IMAGE", "FLAGS"} | ({"ZODI"} if want_zodi else set())
    hdus: dict[str, dict] = {}
    buf, buf_off, off = head, 0, 0
    while not needed <= hdus.keys():
        rel = off - buf_off
        parsed = _parse_header(buf, rel) if 0 <= rel < len(buf) else None
        if parsed is None:
            buf, buf_off = await reader.get(url, off, off + HEAD_BYTES - 1), off
            parsed = _parse_header(buf, 0)
            if parsed is None:
                raise RuntimeError(f"no FITS header at byte {off} of {url}")
        h, hlen = parsed
        name = str(h.get("EXTNAME", "PRIMARY")).strip()
        data_off = off + hlen
        entry = {"header": h.tostring(), "data_off": data_off}
        if h.get("ZIMAGE"):  # tile-compressed: keep the tile index (descriptor table)
            tbl_len = h["NAXIS1"] * h["NAXIS2"]
            rel_d = data_off - buf_off
            if rel_d + tbl_len <= len(buf):
                entry["table"] = buf[rel_d:rel_d + tbl_len]
            else:
                entry["table"] = await reader.get(url, data_off, data_off + tbl_len - 1)
        hdus[name] = entry
        off = data_off + _data_size(h)
        if name in ("EPSF", "WCS-WAVE"):
            break
    missing = needed - hdus.keys()
    if missing:
        raise RuntimeError(f"HDUs {missing} not found in {url}")
    return {"hdus": hdus, "wcswave": _find_tail_hdu(tail, "WCS-WAVE")}


# ------------------------------------------------------------------ pixel reading
def pixel_bbox(header: fits.Header, ra: float, dec: float, size_deg: float, margin: int = 2):
    """Detector pixel box [x0, x1) x [y0, y1) covering a size_deg square around (ra, dec)."""
    w = WCS(header, naxis=2)
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    half = size_deg / 2
    g = np.linspace(-half, half, 5)
    dlon, dlat = np.meshgrid(g, g)
    pts = SkyCoord(ra, dec, unit="deg").spherical_offsets_by(dlon.ravel() * u.deg,
                                                            dlat.ravel() * u.deg)
    x, y = w.world_to_pixel(pts)
    x0 = max(int(math.floor(np.nanmin(x))) - margin, 0)
    y0 = max(int(math.floor(np.nanmin(y))) - margin, 0)
    x1 = min(int(math.ceil(np.nanmax(x))) + margin + 1, nx)
    y1 = min(int(math.ceil(np.nanmax(y))) + margin + 1, ny)
    if x0 >= x1 or y0 >= y1:
        raise NoOverlap()
    return x0, x1, y0, y1


_DTYPES = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


async def read_rows(reader, url, entry, x0, x1, y0, y1) -> np.ndarray:
    h = fits.Header.fromstring(entry["header"])
    nx, bp = h["NAXIS1"], abs(h["BITPIX"]) // 8
    start = entry["data_off"] + y0 * nx * bp
    raw = await reader.get(url, start, start + (y1 - y0) * nx * bp - 1)
    arr = np.frombuffer(raw, dtype=_DTYPES[h["BITPIX"]]).reshape(y1 - y0, nx)[:, x0:x1]
    if h.get("BSCALE", 1) != 1 or h.get("BZERO", 0) != 0:
        arr = arr * h.get("BSCALE", 1) + h.get("BZERO", 0)
    return np.ascontiguousarray(arr)


async def read_rows_compressed(reader, url, entry, x0, x1, y0, y1) -> np.ndarray:
    """Read rows from a tile-compressed image whose tiles are full-width row bands.

    Fetch only the needed tiles' bytes from the heap, then let astropy decompress
    them by building a tiny compressed HDU with the descriptors rebased.
    """
    h = fits.Header.fromstring(entry["header"])
    zt1, zt2 = h.get("ZTILE1", h["ZNAXIS1"]), h.get("ZTILE2", 1)
    if zt1 != h["ZNAXIS1"]:
        raise NotImplementedError("tiles are not full-width rows")
    width = h["NAXIS1"]  # bytes per table row: 8 = 1P (int32 pair), 16 = 1Q (int64 pair)
    desc = np.frombuffer(entry["table"], dtype=">i4" if width == 8 else ">i8").reshape(-1, 2)
    t0, t1 = y0 // zt2, math.ceil(y1 / zt2)
    sel = desc[t0:t1]
    lo, hi = int(sel[:, 1].min()), int((sel[:, 1] + sel[:, 0]).max())
    heap_start = entry["data_off"] + h.get("THEAP", h["NAXIS1"] * h["NAXIS2"])
    heap = await reader.get(url, heap_start + lo, heap_start + hi - 1)

    h2 = h.copy()
    nrows = min(t1 * zt2, h["ZNAXIS2"]) - t0 * zt2
    h2["NAXIS2"], h2["ZNAXIS2"], h2["PCOUNT"] = t1 - t0, nrows, len(heap)
    for k in ("THEAP", "CHECKSUM", "DATASUM", "ZHECKSUM", "ZDATASUM"):
        h2.remove(k, ignore_missing=True)
    new = sel.copy()
    new[:, 1] -= lo
    body = new.astype(desc.dtype).tobytes() + heap
    body += b"\0" * (-len(body) % BLOCK)
    blob = (fits.PrimaryHDU().header.tostring() + h2.tostring()).encode("ascii") + body
    with fits.open(io.BytesIO(blob), memmap=False) as hd:
        data = np.asarray(hd[1].data)
    r0 = y0 - t0 * zt2
    return np.ascontiguousarray(data[r0:r0 + (y1 - y0), x0:x1])


def decompress_tiles(entry: dict, t0: int, t1: int, heap: bytes, lo: int) -> np.ndarray:
    """Decompress tile rows t0..t1 of a row-tiled compressed image from their heap bytes."""
    h = fits.Header.fromstring(entry["header"])
    zt2 = h.get("ZTILE2", 1)
    desc = np.frombuffer(entry["table"], dtype=">i4" if h["NAXIS1"] == 8 else ">i8").reshape(-1, 2)
    sel = desc[t0:t1]
    h2 = h.copy()
    h2["NAXIS2"] = t1 - t0
    h2["ZNAXIS2"] = min(t1 * zt2, h["ZNAXIS2"]) - t0 * zt2
    h2["PCOUNT"] = len(heap)
    for k in ("THEAP", "CHECKSUM", "DATASUM", "ZHECKSUM", "ZDATASUM"):
        h2.remove(k, ignore_missing=True)
    new = sel.copy()
    new[:, 1] -= lo
    body = new.astype(desc.dtype).tobytes() + heap
    body += b"\0" * (-len(body) % BLOCK)
    blob = (fits.PrimaryHDU().header.tostring() + h2.tostring()).encode("ascii") + body
    with fits.open(io.BytesIO(blob), memmap=False) as hd:
        return np.asarray(hd[1].data)


def flag_rows(ih: fits.Header, ra: float, dec: float, y0: int, y1: int) -> tuple[int, int]:
    """Detector rows to read FLAGS for: all of [y0, y1), or a box around the target.

    FLAGS is ~40% of the bytes of a full-field read. Near the target is where they matter
    (photometry, the contamination panel); median stacking rejects cosmic rays elsewhere.
    """
    if settings.flags_mode == "full":
        return y0, y1
    if settings.flags_mode == "none":
        return y0, y0
    _, ty = WCS(ih, naxis=2).world_to_pixel_values(ra, dec)
    b = settings.flags_box_px
    fy0 = max(y0, int(round(float(ty))) - b)
    fy1 = min(y1, int(round(float(ty))) + b + 1)
    return (fy0, fy1) if fy0 < fy1 else (y0, y0)


def _heap_span(entry: dict, t0: int, t1: int):
    h = fits.Header.fromstring(entry["header"])
    desc = np.frombuffer(entry["table"], dtype=">i4" if h["NAXIS1"] == 8 else ">i8").reshape(-1, 2)
    sel = desc[t0:t1]
    lo, hi = int(sel[:, 1].min()), int((sel[:, 1] + sel[:, 0]).max())
    heap_start = entry["data_off"] + h.get("THEAP", h["NAXIS1"] * h["NAXIS2"])
    return lo, hi, heap_start


# Where IMAGE data starts, per detector. Seeded from DR1 (2026-11-15: D1/D2 headers are 9
# blocks, D3-D6 8 blocks) and updated from every file we parse, so a new release with
# different headers costs one extra request per detector, once.
_IMG_OFF_SEED = {1: 28800, 2: 28800, 3: 25920, 4: 25920, 5: 25920, 6: 25920}


def _detector_of(url: str) -> int | None:
    import re
    m = re.search(r"_\d+D(\d)_", url.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else None


def _img_off_hint(det) -> int:
    return disk_cache.get(("layout_hint", det)) or _IMG_OFF_SEED.get(det, 25920)


async def read_layout_multi(reader, url: str) -> dict:
    """Headers in ONE multi-range request: start of file, FLAGS header + tile index (at
    the offset every SPHEREx L2 file uses, verified after parsing), and the tail."""
    # Exact sizes seen in DR1 files: PRIMARY 1 block, IMAGE header 8 blocks, FLAGS header
    # 6 blocks + 16320-byte tile index. Asking for just that keeps headers ~70 KB/image;
    # if a future release has bigger headers we re-read with more room.
    det = _detector_of(url)
    img_off = _img_off_hint(det)
    guess = img_off + 16646400  # headers + 2040x2040 float32 IMAGE
    span = 12 * BLOCK
    head, fl_buf, tail = await get_many(reader, url, [(0, img_off - 1),
                                                     (guess, guess + span - 1), (None, 2 * BLOCK)])
    p = _parse_header(head, 0)
    if p is None:
        raise RuntimeError("no primary header")
    off = p[1] + _data_size(p[0])
    p = _parse_header(head, off)
    if p is None:  # bigger header than DR1: read more
        head = (await get_many(reader, url, [(0, 4 * HEAD_BYTES - 1)]))[0]
        p = _parse_header(head, off)
    if p is None or str(p[0].get("EXTNAME", "")).strip() != "IMAGE":
        raise RuntimeError("IMAGE header not where expected")
    ih, hlen = p
    img = {"header": ih.tostring(), "data_off": off + hlen}
    if off + hlen != img_off:
        STATS["layout_hint_miss"] += 1
        disk_cache.set(("layout_hint", det), off + hlen)
    fl_off = off + hlen + _data_size(ih)
    if fl_off != guess:  # unusual layout: read the FLAGS header where it really is
        fl_buf = (await get_many(reader, url, [(fl_off, fl_off + span - 1)]))[0]
    p = _parse_header(fl_buf, 0)
    if p is None:
        fl_buf = (await get_many(reader, url, [(fl_off, fl_off + 4 * span - 1)]))[0]
        p = _parse_header(fl_buf, 0)
    if p is None or str(p[0].get("EXTNAME", "")).strip() != "FLAGS":
        raise RuntimeError("FLAGS header not where expected")
    fh, flen = p
    flags = {"header": fh.tostring(), "data_off": fl_off + flen}
    if fh.get("ZIMAGE"):
        n = fh["NAXIS1"] * fh["NAXIS2"]
        if flen + n <= len(fl_buf):
            flags["table"] = fl_buf[flen:flen + n]
        else:
            flags["table"] = (await get_many(reader, url, [(fl_off + flen, fl_off + flen + n - 1)]))[0]
    wave = _find_tail_hdu(tail, "WCS-WAVE") if tail else None
    if wave is None:  # bigger tail, one more try
        tail = (await get_many(reader, url, [(None, 8 * BLOCK)]))[0]
        wave = _find_tail_hdu(tail, "WCS-WAVE")
    return {"hdus": {"IMAGE": img, "FLAGS": flags}, "wcswave": wave}


async def range_cutout_multi(reader, url: str, layout_key: str, ra: float, dec: float,
                             size: float) -> bytes:
    """Cutout in ~2 requests: headers (cached per file), then exactly the pixels needed."""
    lkey = ("layout", layout_key, False)
    layout = disk_cache.get(lkey)
    if layout is None:
        layout = await read_layout_multi(reader, url)
        disk_cache.set(lkey, layout)
    else:
        STATS["layout_hits"] += 1
    hdus = layout["hdus"]
    ih = fits.Header.fromstring(hdus["IMAGE"]["header"])
    x0, x1, y0, y1 = pixel_bbox(ih, ra, dec, size)
    nx, bp = ih["NAXIS1"], abs(ih["BITPIX"]) // 8
    doff = hdus["IMAGE"]["data_off"]
    ranges = [(doff + (y * nx + x0) * bp, doff + (y * nx + x1) * bp - 1) for y in range(y0, y1)]
    fl = hdus["FLAGS"]
    fy0, fy1 = flag_rows(ih, ra, dec, y0, y1)
    want_flags = fy1 > fy0 and "table" in fl
    if want_flags:
        lo, hi, hs = _heap_span(fl, fy0, fy1)
        ranges.append((hs + lo, hs + hi - 1))
    parts = await get_many(reader, url, ranges)
    dt = _DTYPES[ih["BITPIX"]]
    image = np.stack([np.frombuffer(b, dtype=dt) for b in parts[: y1 - y0]])
    flags = np.zeros(image.shape, dtype=np.int32)
    if want_flags:
        rows = decompress_tiles(fl, fy0, fy1, parts[-1], lo)
        flags[fy0 - y0:fy1 - y0] = rows[:, x0:x1]
    else:
        fy0 = fy1 = y0
    return pack_cutout(layout, image, flags, None, x0, y0, url, flag_y=(fy0 - y0, fy1 - y0))


def _shift_crpix(h: fits.Header, x0: int, y0: int):
    """Keep every WCS (sky, alternate 'A' detector coords, 'W' wavelength) valid after cropping."""
    for key in list(h.keys()):
        if key.startswith("CRPIX1"):
            h[key] = h[key] - x0
        elif key.startswith("CRPIX2"):
            h[key] = h[key] - y0


def _flags_header(h: fits.Header) -> fits.Header:
    out = fits.Header()
    for card in h.cards:
        if card.keyword.startswith(("MSKN", "MSKM", "MSKD")):
            out.append(card)
    out["EXTNAME"] = "FLAGS"
    return out


def pack_cutout(layout: dict, image, flags, zodi, x0, y0, src_url, flag_y=None) -> bytes:
    hdus = layout["hdus"]
    ih = fits.Header.fromstring(hdus["IMAGE"]["header"])
    for k in ("CHECKSUM", "DATASUM"):
        ih.remove(k, ignore_missing=True)
    _shift_crpix(ih, x0, y0)
    prim = fits.PrimaryHDU()
    prim.header["SRCURL"] = src_url[-68:]
    prim.header["CUT_X0"], prim.header["CUT_Y0"] = x0, y0
    fy = flag_y or (0, image.shape[0])
    prim.header["FLAG_Y0"], prim.header["FLAG_Y1"] = int(fy[0]), int(fy[1])  # rows with FLAGS
    out = [prim, fits.ImageHDU(image.astype(np.float32), header=ih, name="IMAGE"),
           fits.ImageHDU(flags.astype(np.int32),
                         header=_flags_header(fits.Header.fromstring(hdus["FLAGS"]["header"])),
                         name="FLAGS")]
    if zodi is not None:
        out.append(fits.ImageHDU(zodi.astype(np.float32), name="ZODI"))
    buf = io.BytesIO()
    fits.HDUList(out).writeto(buf)
    return buf.getvalue() + (layout["wcswave"] or b"")


async def range_cutout(reader, url: str, layout_key: str, ra: float, dec: float,
                       size: float, want_zodi: bool) -> bytes:
    lkey = ("layout", layout_key, want_zodi)
    layout = disk_cache.get(lkey)
    if layout is None:
        layout = await read_layout(reader, url, want_zodi)
        disk_cache.set(lkey, layout)
    else:
        STATS["layout_hits"] += 1
    hdus = layout["hdus"]
    ih = fits.Header.fromstring(hdus["IMAGE"]["header"])
    x0, x1, y0, y1 = pixel_bbox(ih, ra, dec, size)
    fl = hdus["FLAGS"]
    flag_reader = read_rows_compressed if "table" in fl else read_rows
    fy0, fy1 = flag_rows(ih, ra, dec, y0, y1)
    jobs = [read_rows(reader, url, hdus["IMAGE"], x0, x1, y0, y1)]
    if fy1 > fy0:
        jobs.append(flag_reader(reader, url, fl, x0, x1, fy0, fy1))
    if want_zodi:
        jobs.append(read_rows(reader, url, hdus["ZODI"], x0, x1, y0, y1))
    parts = await asyncio.gather(*jobs)
    flags = np.zeros(parts[0].shape, dtype=np.int32)
    if fy1 > fy0:
        flags[fy0 - y0:fy1 - y0] = parts[1]
    zodi = parts[-1] if want_zodi else None
    return pack_cutout(layout, parts[0], flags, zodi, x0, y0, url, flag_y=(fy0 - y0, fy1 - y0))


# ---------------------------------------------------------------- sources + cache
def source_urls(row: dict) -> dict[str, str]:
    key = row.get("s3_key")
    if not key and "/ibe/data/spherex/" in row["access_url"]:
        key = row["access_url"].split("/ibe/data/spherex/", 1)[1]
    urls = {"irsa": row["access_url"]}
    if key:
        urls["s3"] = settings.s3_base + key
    return urls


def cutout_url(access_url: str, ra: float, dec: float, size_deg: float) -> str:
    """IRSA server-side cutout (verified: appending to the SIA access_url works)."""
    return f"{access_url}?center={ra:.6f},{dec:.6f}d&size={size_deg:.4f}"


def cutout_key(row: dict, ra: float, dec: float, size: float, want_zodi: bool) -> str:
    step = 3 / 3600  # snap centre to 3" so nearby requests share the cache
    raw = (f"{row['access_url'].rsplit('/', 1)[-1]}|{snap(ra, step):.5f}|"
           f"{snap(dec, step):.5f}|{size:.4f}|{int(want_zodi)}|"
           f"{settings.flags_mode}{settings.flags_box_px}")
    return "cut:" + hashlib.sha1(raw.encode()).hexdigest()


def snapped_center(ra: float, dec: float) -> tuple[float, float]:
    step = 3 / 3600
    return snap(ra, step), snap(dec, step)


async def fetch_one(row: dict, ra: float, dec: float, size: float, reader=None):
    """Returns (row, fits_bytes, origin). Raises NoOverlap or the last download error.

    Strategies, in order (config data_source picks the first):
      irsa: multi-range reads of IRSA's copy (few KB per image; best from home)
      s3:   full-row range reads of the AWS copy (best when running inside AWS)
    """
    want_zodi = settings.fetch_zodi
    key = cutout_key(row, ra, dec, size, want_zodi)
    if (data := disk_cache.get(key)) is not None:
        STATS["cutout_hits"] += 1
        return row, data, "cache"
    ra, dec = snapped_center(ra, dec)  # cutout must match its cache key exactly
    reader = reader or HttpRangeReader()
    urls = source_urls(row)
    order = [settings.data_source] + [s for s in ("irsa", "s3") if s != settings.data_source]
    last: Exception | None = None
    layout_key = row["access_url"].rsplit("/", 1)[-1]
    for src in order:
        if src not in urls:
            continue
        try:
            if src == "irsa" and not want_zodi:
                data = await range_cutout_multi(reader, urls[src], layout_key, ra, dec, size)
            else:
                data = await range_cutout(reader, urls[src], layout_key, ra, dec, size, want_zodi)
            STATS[f"cutouts_{src}"] += 1
            disk_cache.set(key, data)
            return row, data, src
        except NoOverlap:
            raise
        except Exception as e:  # try the next source
            STATS[f"errors_{src}"] += 1
            log.warning("range read via %s failed for %s: %r", src, row["uid"], e)
            last = e
    if settings.use_cutout_service_fallback:
        client, sem = http.get()
        async with sem:
            r = await client.get(cutout_url(row["access_url"], ra, dec, size))
        r.raise_for_status()
        STATS["cutouts_service"] += 1
        disk_cache.set(key, r.content)
        return row, r.content, "service"
    raise last or RuntimeError("no data source available")


async def fetch_all(rows, ra, dec, size):
    """Async generator yielding (row, bytes | Exception, origin) as each one finishes.

    Tasks are created up front in the given order, so earlier rows (earlier visits)
    tend to finish first. One failure never stops the others.
    """
    async def wrapped(r):
        t0 = time.perf_counter()
        try:
            row, data, origin = await fetch_one(r, ra, dec, size)
            return row, data, origin
        except Exception as e:
            return r, e, "error"
        finally:
            STATS["fetch_secs"] += time.perf_counter() - t0

    img_sem = asyncio.Semaphore(settings.max_parallel_images)

    async def limited(r):
        async with img_sem:  # images start in order -> earliest visit completes first
            return await wrapped(r)

    tasks = [asyncio.create_task(limited(r)) for r in rows]
    try:
        for fut in asyncio.as_completed(tasks):
            yield await fut
    finally:
        for t in tasks:
            t.cancel()


def open_fits(data: bytes) -> fits.HDUList:
    return fits.open(io.BytesIO(data), memmap=False)
