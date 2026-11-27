"""Price data: download, cache, checksums, validation, features, date split.

Default settings live in ``src/config.py`` (``DataConfig``). The price files
for the default tickers are committed in ``data/raw/`` together with
``data/raw/SHA256SUMS``: yfinance re-scales adjusted prices whenever a stock
pays a new dividend, so a fresh download would not reproduce our numbers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

MARKET_FEATURES = ["ret5", "ret20", "vol20"]

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
CHECKSUM_FILE = RAW_DIR / "SHA256SUMS"
MIN_SPLIT_DAYS = 2  # an episode needs at least two trading days


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checksums(path: Path = CHECKSUM_FILE) -> dict[str, str]:
    """``{filename: sha256}`` from a ``sha256sum``-style file (empty if missing)."""
    if not Path(path).exists():
        return {}
    pairs = (line.split() for line in Path(path).read_text().splitlines() if line.strip())
    return {name: digest for digest, name in pairs}


def validate_prices(df: pd.DataFrame, ticker: str, start: str, end: str) -> None:
    """Fail loudly on data problems instead of training on bad prices."""
    if df.empty:
        raise ValueError(f"{ticker}: no price data between {start} and {end}")
    problems = []
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        problems.append("dates are not strictly increasing")
    close = df["Close"]
    if close.isna().any() or (close <= 0).any():
        problems.append("missing or non-positive closing prices")
    gaps = df.index.to_series().diff().dt.days.dropna()
    if (gaps > 7).any():
        problems.append(f"gap of {int(gaps.max())} calendar days between trading days")
    # The data must cover the requested range (allowing for weekends/holidays).
    week = pd.Timedelta(days=7)
    if df.index.min() > pd.Timestamp(start) + week or df.index.max() < pd.Timestamp(end) - week:
        problems.append(f"covers {df.index.min().date()}..{df.index.max().date()}, not {start}..{end}")
    if problems:
        raise ValueError(f"{ticker} price data failed checks: " + "; ".join(problems) + ".")


def _read_cached(path: Path, expected: str | None, verify_checksum: bool) -> pd.DataFrame:
    if verify_checksum:
        if expected is None:
            print(f"Warning: {path.name} has no checksum in SHA256SUMS, so it is not verified. "
                  "Commit it and add its checksum (see data/README.md).")
        elif file_sha256(path) != expected:
            raise ValueError(f"{path} does not match its recorded checksum in SHA256SUMS. Restore "
                             f"the committed file with `git checkout -- data/raw/{path.name}`.")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def _download(ticker: str, start: str, end: str, adjusted: bool, path: Path) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=adjusted, progress=False,
                     multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Downloaded {ticker} to {path}. Commit it and add its checksum to "
          "data/raw/SHA256SUMS so teammates use the same prices.")
    return df


def load_price_data(ticker: str, start: str, end: str, cache_dir: Path = RAW_DIR,
                    adjusted: bool = True, verify_checksum: bool = True) -> pd.DataFrame:
    """Daily OHLCV bars for one ticker from the committed CSV, else from yfinance.

    A committed file is checked against ``SHA256SUMS``; a committed ticker
    whose file is missing is never silently re-downloaded.
    """
    path = Path(cache_dir) / f"{ticker}.csv"
    expected = read_checksums(Path(cache_dir) / "SHA256SUMS").get(path.name)
    if expected and not path.exists():
        raise FileNotFoundError(f"{path} is missing but has a recorded checksum. Restore it "
                                f"with `git checkout -- data/raw/{path.name}`.")
    if path.exists():
        df = _read_cached(path, expected, verify_checksum)
    else:
        df = _download(ticker, start, end, adjusted, path)
    df = df.sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
    validate_prices(df, ticker, start, end)
    return df


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ret5``, ``ret20`` and ``vol20`` computed from closing prices.

    Every feature on day t uses closes up to and including day t only.
    Compute features on the full series *before* splitting, so the first day
    of each split already has a 20-day history. Rows without a full history
    are dropped.
    """
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"].astype(float)
    log_close = np.log(out["Close"])
    out["ret5"] = log_close - log_close.shift(5)
    out["ret20"] = log_close - log_close.shift(20)
    out["vol20"] = log_close.diff().rolling(20).std()
    return out.dropna()


def slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with ``start <= date <= end`` (both inclusive)."""
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def split_by_date(features: pd.DataFrame, cfg: DataConfig | None = None) -> dict:
    """Chronological split into ``train``, ``val`` and a list of ``test`` windows."""
    cfg = cfg or DataConfig()
    splits = {
        "train": slice_dates(features, *cfg.train),
        "val": slice_dates(features, *cfg.val),
        "test": [slice_dates(features, s, e) for s, e in cfg.test_windows],
    }
    named = [("train", splits["train"]), ("val", splits["val"])]
    named += [(f"test {s}..{e}", d) for (s, e), d in zip(cfg.test_windows, splits["test"], strict=True)]
    for name, d in named:
        if len(d) < MIN_SPLIT_DAYS:
            raise ValueError(f"The {name} split has {len(d)} trading days; check the dates in "
                             "src/config.py and the price data.")
    return splits


def load_ticker(ticker: str, cfg: DataConfig | None = None) -> dict:
    """Load (or download) one ticker's prices, add features, and split by date."""
    cfg = cfg or DataConfig()
    prices = load_price_data(ticker, cfg.download_start, cfg.download_end, adjusted=cfg.adjusted)
    return split_by_date(compute_market_features(prices), cfg)
