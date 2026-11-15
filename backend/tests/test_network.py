"""Talks to the real archives. Run before a release:  pytest -m network"""
import io

import pytest
from astropy.io import fits

from app import fetcher, metadata
from app.resolver import resolve

pytestmark = pytest.mark.network

WISE0855 = (133.795129, -7.245147)


async def test_metadata_finds_all_detectors():
    rows = await metadata.query_images(*WISE0855)
    assert len(rows) > 50
    assert {r["detector"] for r in rows} == {1, 2, 3, 4, 5, 6}
    assert all(r["s3_key"] for r in rows)


async def test_range_cutout_matches_irsa_cutout_service():
    rows = await metadata.query_images(*WISE0855)
    row = rows[0]
    ra, dec = fetcher.snapped_center(*WISE0855)
    data = await fetcher.range_cutout(fetcher.HttpRangeReader(), fetcher.source_urls(row)["s3"],
                                      row["uid"] + "-nt", ra, dec, 0.05, want_zodi=False)
    with fits.open(io.BytesIO(data)) as mine:
        assert {"IMAGE", "FLAGS", "WCS-WAVE"} <= {h.name for h in mine}
        assert mine["IMAGE"].header["DETECTOR"] == row["detector"]


@pytest.mark.parametrize("name", ["Barnard's Star", "RW Aur"])
def test_resolver(name):
    r = resolve(name)
    assert r["source"] in ("simbad", "sesame") and 0 <= r["ra"] < 360
