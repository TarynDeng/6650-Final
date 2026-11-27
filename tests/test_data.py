"""Data checks: committed price files, checksums, validation, statistics."""

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig
from src.data import (
    CHECKSUM_FILE,
    RAW_DIR,
    file_sha256,
    load_price_data,
    load_ticker,
    read_checksums,
    split_by_date,
    validate_prices,
)
from tests.helpers import make_features

CFG = DataConfig()


@pytest.mark.parametrize("ticker", CFG.tickers)
def test_committed_prices_match_checksums_and_cover_all_splits(ticker):
    path = RAW_DIR / f"{ticker}.csv"
    assert path.exists(), f"{path} should be committed"
    assert read_checksums()[path.name] == file_sha256(path)
    splits = load_ticker(ticker, CFG)
    assert splits["train"].index.min() <= pd.Timestamp("2015-01-05")
    assert splits["train"].index.max() < splits["val"].index.min()
    assert splits["val"].index.max() < splits["test"][0].index.min()
    assert all(len(w) >= 245 for w in splits["test"])


def test_modified_price_file_is_rejected(tmp_path):
    src = RAW_DIR / "AAPL.csv"
    (tmp_path / "AAPL.csv").write_text(src.read_text().replace("2020-01-02,", "2020-01-02,1", 1))
    (tmp_path / "SHA256SUMS").write_text(CHECKSUM_FILE.read_text())
    with pytest.raises(ValueError, match="checksum"):
        load_price_data("AAPL", CFG.download_start, CFG.download_end, cache_dir=tmp_path)


def test_validate_prices_catches_bad_data():
    dates = pd.bdate_range("2020-01-01", periods=30)
    good = pd.DataFrame({"Close": np.linspace(10, 20, 30)}, index=dates)
    validate_prices(good, "X", "2020-01-01", str(dates[-1].date()))
    bad = good.copy()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(bad, "X", "2020-01-01", str(dates[-1].date()))
    with pytest.raises(ValueError, match="gap"):
        validate_prices(good.drop(dates[10:20]), "X", "2020-01-01", str(dates[-1].date()))
    with pytest.raises(ValueError, match="covers"):
        validate_prices(good, "X", "2019-01-01", str(dates[-1].date()))
    neg = good.copy()
    neg.iloc[3, 0] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(neg, "X", "2020-01-01", str(dates[-1].date()))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_prices(good.iloc[::-1], "X", "2020-01-01", str(dates[-1].date()))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_prices(pd.concat([good, good.iloc[[5]]]).sort_index(), "X", "2020-01-01",
                        str(dates[-1].date()))


def test_split_rejects_windows_without_data():
    f = make_features(n=60)  # 2020 only: train/val/test windows from config are empty
    with pytest.raises(ValueError, match="trading days"):
        split_by_date(f, CFG)


def test_empty_price_data_is_rejected():
    empty = pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="no price data"):
        validate_prices(empty, "X", "2020-01-01", "2020-12-31")


def test_missing_committed_file_is_not_redownloaded(tmp_path):
    (tmp_path / "SHA256SUMS").write_text(CHECKSUM_FILE.read_text())
    with pytest.raises(FileNotFoundError, match="git checkout"):
        load_price_data("AAPL", CFG.download_start, CFG.download_end, cache_dir=tmp_path)
