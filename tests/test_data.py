"""Data checks: committed price files, checksums, validation, statistics."""

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig
from src.environment import (CHECKSUM_FILE, RAW_DIR, file_sha256, load_price_data, load_ticker,
                             read_checksums, validate_prices)
from src.evaluation import paired_comparison, summarize

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


def _fake_results():
    rows = []
    for t in ("A", "B"):
        for w in ("2023", "2024"):
            base = {"A": 0.1, "B": -0.1}[t] + {"2023": 0.0, "2024": 0.05}[w]
            rows.append({"method": "bh", "ticker": t, "window": w, "seed": -1,
                         "terminal_return": base, "loss_category": 1 if base >= 0 else 2})
            for s in range(3):
                r = base - 0.02 + 0.01 * s
                rows.append({"method": "rl", "ticker": t, "window": w, "seed": s,
                             "terminal_return": r, "loss_category": 1 if r >= 0 else 2})
    return pd.DataFrame(rows)


def test_summary_uses_market_windows_as_samples():
    s = summarize(_fake_results())
    assert s.loc["rl", "n_windows"] == 4 and s.loc["rl", "n_episodes"] == 12
    assert s.loc["bh", "n_windows"] == 4 and s.loc["bh", "n_episodes"] == 4
    # Seed averaging: rl is exactly 0.01 below bh in every window.
    assert s.loc["rl", "mean"] == pytest.approx(s.loc["bh", "mean"] - 0.01)
    assert s.loc["rl", "seed_std"] == pytest.approx(0.01)
    assert s.loc["rl", ["share_cat1", "share_cat2"]].sum() == pytest.approx(1.0)


def test_paired_comparison():
    c = paired_comparison(_fake_results(), "rl", "bh")
    assert c["mean_diff"] == pytest.approx(-0.01)
    assert c["a_wins"] == 0 and c["n_windows"] == 4
    assert c["ci_low"] == pytest.approx(-0.01) and c["ci_high"] == pytest.approx(-0.01)
