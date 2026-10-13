import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from point_attrs import _backfill_one, add_point_columns

ATTRS = pd.DataFrame({"ID": [0, 1, 2], "lon": [-72.0, -77.0, -75.5],
                      "lat": [-13.5, -12.0, -10.1], "altitude": [3500, 120, 2600]})


def test_add_point_columns_pandas():
    df = pd.DataFrame({"date": pd.to_datetime(["2001-01-01"] * 3), "BU": [1, 2, 3], "ID": [2, 0, 2]})
    out = add_point_columns(df, ATTRS)
    assert out.columns.tolist() == ["ID", "lon", "lat", "altitude", "date", "BU"]
    assert out.lon.tolist() == [-75.5, -72.0, -75.5]
    assert out.altitude.tolist() == [2600, 3500, 2600]
    assert out.BU.tolist() == [1, 2, 3]


def test_add_point_columns_unknown_id():
    with pytest.raises(KeyError):
        add_point_columns(pd.DataFrame({"ID": [9]}), ATTRS)


def test_backfill_one(tmp_path):
    p = tmp_path / "simcast_results.parquet"
    df = pd.DataFrame({"date": pd.date_range("2001-01-01", periods=4), "APP": [0, 1, 1, 2], "ID": [0, 0, 1, 1]})
    df.to_parquet(p)
    assert _backfill_one(str(p), ATTRS, force=False) == str(p)
    t = pq.read_table(p)
    assert t.column_names == ["ID", "lon", "lat", "altitude", "date", "APP"]
    back = t.to_pandas()
    assert back.lat.tolist() == [-13.5, -13.5, -12.0, -12.0]
    assert back.APP.tolist() == [0, 1, 1, 2]
    assert _backfill_one(str(p), ATTRS, force=False) is None  # already has the columns
