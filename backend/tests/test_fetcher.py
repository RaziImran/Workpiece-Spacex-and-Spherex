"""The range reader against a synthetic file with the real SPHEREx layout (no network)."""
import io

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from app import fetcher

NX = NY = 300
RA0, DEC0 = 150.0, 2.0


def make_file() -> tuple[bytes, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    img = rng.normal(1, 0.1, (NY, NX)).astype(np.float32)
    flags = rng.integers(0, 1 << 22, (NY, NX)).astype(np.int32)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.pc = [[-0.0016, 0.0005], [0.0005, 0.0016]]  # rotated, ~6" pixels
    h = w.to_header()
    h["DETECTOR"] = 3
    h["MJD-OBS"] = 60800.5
    fh = fits.Header()
    fh["MSKN0000"], fh["MSKM0000"] = "TRANSIENT", 1
    fh["MSKN0021"], fh["MSKM0021"] = "SOURCE", 1 << 21
    wave = fits.BinTableHDU.from_columns(
        [fits.Column(name="X", format="2J", array=np.array([[0, 1]]))], name="WCS-WAVE")
    epsf = fits.BinTableHDU.from_columns(
        [fits.Column(name="P", format="100D", array=np.zeros((50, 100)))], name="EPSF")
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(img, header=h, name="IMAGE"),
        fits.CompImageHDU(flags, header=fh, name="FLAGS", compression_type="RICE_1",
                          tile_shape=(1, NX)),
        fits.ImageHDU(np.ones_like(img), name="VARIANCE"),
        fits.ImageHDU(np.full_like(img, 0.5), name="ZODI"),
        epsf, wave,
    ])
    buf = io.BytesIO()
    hdul.writeto(buf)
    return buf.getvalue(), img, flags


class MemReader:
    def __init__(self, blob):
        self.blob, self.calls, self.bytes = blob, 0, 0

    async def get(self, url, start, end):
        self.calls += 1
        out = self.blob[start:end + 1]
        self.bytes += len(out)
        return out

    async def tail(self, url, n):
        self.calls += 1
        return self.blob[-n:]


@pytest.fixture(scope="module")
def synthetic():
    return make_file()


async def test_layout_finds_all_hdus(synthetic):
    blob, _, _ = synthetic
    lay = await fetcher.read_layout(MemReader(blob), "mem://x", want_zodi=True)
    assert {"IMAGE", "FLAGS", "ZODI"} <= lay["hdus"].keys()
    assert "table" in lay["hdus"]["FLAGS"]
    assert lay["wcswave"] is not None and b"WCS-WAVE" in lay["wcswave"][:2880]


@pytest.mark.parametrize("zodi", [False, True])
async def test_range_cutout_matches_direct_crop(synthetic, zodi):
    blob, img, flags = synthetic
    reader = MemReader(blob)
    data = await fetcher.range_cutout(reader, "mem://x", f"synthetic-{zodi}", RA0 + 0.01,
                                      DEC0 - 0.02, 0.05, want_zodi=zodi)
    with fits.open(io.BytesIO(data)) as cut:
        x0, y0 = cut[0].header["CUT_X0"], cut[0].header["CUT_Y0"]
        ny, nx = cut["IMAGE"].data.shape
        np.testing.assert_array_equal(cut["IMAGE"].data, img[y0:y0 + ny, x0:x0 + nx])
        fy0, fy1 = cut[0].header["FLAG_Y0"], cut[0].header["FLAG_Y1"]
        from app.config import settings
        assert fy1 - fy0 == 2 * settings.flags_box_px + 1  # box around the target
        np.testing.assert_array_equal(cut["FLAGS"].data[fy0:fy1],
                                      flags[y0 + fy0:y0 + fy1, x0:x0 + nx])
        assert ("ZODI" in cut) == zodi
        assert "WCS-WAVE" in cut
        assert cut["FLAGS"].header["MSKN0021"] == "SOURCE"
        # WCS still points at the same sky position after cropping
        w_full = WCS(fits.open(io.BytesIO(blob))["IMAGE"].header)
        w_cut = WCS(cut["IMAGE"].header)
        ra, dec = w_full.pixel_to_world_values(x0 + 5, y0 + 7)
        ra2, dec2 = w_cut.pixel_to_world_values(5, 7)
        assert abs(ra - ra2) < 1e-9 and abs(dec - dec2) < 1e-9
    # Read only part of the file (fixed 64 KB header reads dominate in this tiny file;
    # on a real 66 MB SPHEREx file it is ~2%).
    assert reader.bytes < len(blob) / 2


async def test_no_overlap_raises(synthetic):
    blob, _, _ = synthetic
    with pytest.raises(fetcher.NoOverlap):
        await fetcher.range_cutout(MemReader(blob), "mem://x", "synthetic-far", RA0 + 5, DEC0,
                                   0.05, want_zodi=False)


def test_cutout_key_snaps_to_3_arcsec():
    row = {"access_url": "https://x/y/file.fits"}
    a = fetcher.cutout_key(row, 10.0, 20.0, 0.1, False)
    assert a == fetcher.cutout_key(row, 10.0 + 0.5 / 3600, 20.0 - 0.5 / 3600, 0.1, False)
    assert a != fetcher.cutout_key(row, 10.0 + 10 / 3600, 20.0, 0.1, False)
    assert a != fetcher.cutout_key(row, 10.0, 20.0, 0.2, False)


def test_source_urls_prefers_s3_key():
    row = {"access_url": "https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/a/b.fits",
           "s3_key": None}
    urls = fetcher.source_urls(row)
    assert urls["s3"].endswith("/dr1/a/b.fits") and urls["irsa"] == row["access_url"]


def test_cutout_url_format():
    u = fetcher.cutout_url("https://h/f.fits", 1.5, -2.25, 0.1)
    assert u == "https://h/f.fits?center=1.500000,-2.250000d&size=0.1000"


@pytest.mark.parametrize("mode", ["box", "full"])
async def test_multi_range_cutout_matches_direct_crop(synthetic, mode, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "flags_mode", mode)
    blob, img, flags = synthetic
    reader = MemReader(blob)
    data = await fetcher.range_cutout_multi(reader, "mem://x", f"synthetic-multi-{mode}",
                                            RA0 - 0.01, DEC0 + 0.01, 0.05)
    with fits.open(io.BytesIO(data)) as cut:
        x0, y0 = cut[0].header["CUT_X0"], cut[0].header["CUT_Y0"]
        fy0, fy1 = cut[0].header["FLAG_Y0"], cut[0].header["FLAG_Y1"]
        ny, nx = cut["IMAGE"].data.shape
        np.testing.assert_array_equal(cut["IMAGE"].data, img[y0:y0 + ny, x0:x0 + nx])
        np.testing.assert_array_equal(cut["FLAGS"].data[fy0:fy1], flags[y0 + fy0:y0 + fy1, x0:x0 + nx])
        if mode == "full":
            assert (fy0, fy1) == (0, ny)
        assert "WCS-WAVE" in cut


def test_parse_multipart_byteranges_and_merged_ranges():
    crlf = b"\r\n"
    body = (b"--BOUND" + crlf + b"Content-Type: application/fits" + crlf
            + b"Content-Range: bytes 10-19/1000" + crlf + crlf + bytes(range(10, 20)) + crlf
            + b"--BOUND" + crlf + b"Content-Range: bytes 900-999/1000" + crlf + crlf
            + bytes(100) + crlf + b"--BOUND--" + crlf)
    parts = fetcher.parse_byteranges(body, "multipart/byteranges; boundary=BOUND", "")
    assert fetcher.slice_parts(parts, 12, 14) == bytes([12, 13, 14])  # inside a merged part
    assert len(fetcher.slice_parts(parts, None, 50)) == 50  # suffix range
    single = fetcher.parse_byteranges(b"abc", "application/fits", "bytes 5-7/100")
    assert fetcher.slice_parts(single, 5, 7) == b"abc"
    with pytest.raises(RuntimeError):
        fetcher.slice_parts(parts, 30, 40)


async def test_multi_range_refuses_full_file(monkeypatch):
    """A server that ignores multi-range (S3) answers 200 + the whole file: must abort."""
    import httpx

    def handler(request):
        return httpx.Response(200, content=b"x" * 1000)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    import asyncio
    monkeypatch.setattr(fetcher.http, "get", lambda: (client, asyncio.Semaphore(2)))
    with pytest.raises(RuntimeError, match="ignored multi-range"):
        await fetcher.HttpRangeReader().get_many("https://s3/x", [(0, 9), (20, 29)])
    await client.aclose()
