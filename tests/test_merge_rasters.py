import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from merge_rasters import merge_all_years, merge_year


def _write(path, arr, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs="EPSG:4326", transform=transform, nodata=-9999) as d:
        d.write(arr, 1)


def _regions(tmp_path, dtype=np.float32, overlap=False):
    tr = from_origin(-80, 0, 0.01, 0.01)
    a = np.full((4, 5), -9999, dtype=dtype)
    b = a.copy()
    a[:2, :] = 1
    b[2:, :] = 2
    if overlap:
        b[1, 0] = 3
    y = tmp_path / "res" / "2001"
    _write(y / "cultivo1" / "Sierra Alta" / "BU_sum_2001_cultivo1_Sierra Alta.tif", a, tr)
    _write(y / "cultivo2" / "Costa y Valle Interandino" / "BU_sum_2001_cultivo2_Costa y Valle Interandino.tif", b, tr)
    return y


def test_merge_year(tmp_path):
    y = _regions(tmp_path)
    out = merge_year(y, "BU", variety="resistant")
    assert out.name == "BU_resistant_2001.tif"
    with rasterio.open(out) as s:
        m = s.read(1)
        assert s.crs.to_epsg() == 4326 and s.nodata == -9999
        assert not s.profile.get("tiled") and s.profile.get("compress") is None
        assert s.overviews(1) == []
        assert s.tags()["n_valid_pixels"] == "20"
        assert "cultivo1:Sierra Alta" in s.tags()["regions_merged"]
    assert (m[:2] == 1).all() and (m[2:] == 2).all()
    assert merge_year(y, "APP", variety="resistant") is None  # no APP inputs


def test_merge_overlap_raises(tmp_path):
    y = _regions(tmp_path, overlap=True)
    with pytest.raises(ValueError):
        merge_year(y, "BU", variety="resistant")


def test_merge_all_years_skips_non_year_dirs(tmp_path):
    y = _regions(tmp_path)
    (y.parent / "not_a_year").mkdir()
    out = merge_all_years(y.parent, variety="susceptible", variables=("BU",))
    assert [p.name for p in out] == ["BU_susceptible_2001.tif"]


def test_plain_layout_reads_with_pillow(tmp_path):
    from PIL import Image

    y = _regions(tmp_path)
    out = merge_year(y, "BU", variety="resistant")
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) == 1
    assert 322 not in im.tag_v2 and im.tag_v2.get(259) == 1  # no tiles, uncompressed
    with rasterio.open(out) as s:
        np.testing.assert_array_equal(np.array(im), s.read(1))
