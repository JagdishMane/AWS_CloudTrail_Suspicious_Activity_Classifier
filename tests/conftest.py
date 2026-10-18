"""Shared pytest fixtures: a synthetic feature mart built through the real pipeline code."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import write_cloudtrail_files  # noqa: E402

from cloudtrail_mlops import features, ingest  # noqa: E402


@pytest.fixture(scope="session")
def event_parquet_dir(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("events")
    ingest.convert_to_parquet(write_cloudtrail_files(raw, n_events=6000, seed=1), out)
    return out


@pytest.fixture(scope="session")
def feature_mart(event_parquet_dir):
    con = features.connect()
    audit = features.build_feature_mart(con, (event_parquet_dir / "part-*.parquet").as_posix())
    mart = con.execute("SELECT * FROM feature_mart").fetchdf()
    mart.attrs["audit"] = audit
    return mart
