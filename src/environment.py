"""Historical stock-trading environment.

See ``docs/proposal.md`` Sections 5 and 7. Default settings live in
``src/config.py``.

Responsibilities:
    * Load daily historical market data for one stock (yfinance, cached under
      ``data/raw/``).
    * Compute market features: 5-day log return, 20-day log return, and
      20-day volatility of daily log returns.
    * Split data chronologically into train / validation / test windows.
    * Provide a Gymnasium environment with the 9-action discrete
      trade-sizing space.

We use a small custom environment instead of an existing trading library:
Gym Trading Env's actions are target positions rather than "buy with x% of
cash / sell x% of holdings" orders, and a short custom environment is easier
to verify (see ``tests/test_environment.py``).

Rules the implementation respects:
    * One stock per episode. No leverage and no short selling.
    * Transaction costs are deducted, so they show up in the portfolio value.
    * Reward is the per-step log return of portfolio value,
      ``r_t = ln(V_{t+1} / V_t)``, so the rewards of an episode sum to
      ``ln(V_T / V_0)`` with fees included.
    * Timing: the observation at day t uses closes up to and including day t,
      the order fills at day t's close, and the portfolio is marked to market
      at day t+1's close. No action uses future information.
    * Feature statistics (bins, normalization) are fit by the agents on the
      training split only.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from .config import DataConfig, EnvConfig

# Discrete trade-sizing actions (tentative; see proposal Section 5.4).
# Each entry is (side, fraction): "buy" uses a fraction of available cash,
# "sell" sells a fraction of current holdings.
ACTIONS: dict[int, tuple[str, float]] = {
    0: ("hold", 0.0),
    1: ("buy", 0.25),
    2: ("buy", 0.50),
    3: ("buy", 0.75),
    4: ("buy", 1.00),
    5: ("sell", 0.25),
    6: ("sell", 0.50),
    7: ("sell", 0.75),
    8: ("sell", 1.00),
}
N_ACTIONS = len(ACTIONS)

MARKET_FEATURES = ["ret5", "ret20", "vol20"]
# Observation vector: market features followed by the position fraction
# (market value of holdings / portfolio value). The cash fraction is
# 1 - position fraction, so it is not included separately.
OBS_NAMES = MARKET_FEATURES + ["position_frac"]

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


CHECKSUM_FILE = RAW_DIR / "SHA256SUMS"
# Trades worth less than this (in dollars) are skipped, so float round-off
# leftovers never count as trades.
MIN_TRADE_VALUE = 0.01


def file_sha256(path: Path) -> str:
    import hashlib

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
    if df.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7) or \
            df.index.max() < pd.Timestamp(end) - pd.Timedelta(days=7):
        problems.append(f"covers {df.index.min().date()}..{df.index.max().date()}, "
                        f"not {start}..{end}")
    if problems:
        raise ValueError(f"{ticker} price data failed checks: " + "; ".join(problems) + ".")


def load_price_data(ticker: str, start: str, end: str, cache_dir: Path = RAW_DIR,
                    adjusted: bool = True, verify_checksum: bool = True) -> pd.DataFrame:
    """Load daily OHLCV bars for one ticker, using a CSV cache.

    The first call downloads from yfinance and writes ``{cache_dir}/{ticker}.csv``;
    later calls read the cached file. The CSVs for the default tickers are
    committed to git together with ``data/raw/SHA256SUMS``, because yfinance
    re-scales adjusted prices whenever a new dividend is paid: a fresh
    download would not reproduce our numbers. A cached file whose checksum
    does not match the recorded one is rejected.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{ticker}.csv"
    expected = read_checksums(cache_dir / "SHA256SUMS").get(path.name)
    if expected and not path.exists():
        # A committed ticker must never be silently re-downloaded: the new
        # prices would differ from everyone else's.
        raise FileNotFoundError(f"{path} is missing but has a recorded checksum. Restore it "
                                f"with `git checkout -- data/raw/{path.name}`.")
    if path.exists():
        if verify_checksum:
            if expected is None:
                print(f"Warning: {path.name} has no checksum in SHA256SUMS, so it is not "
                      "verified. Commit it and add its checksum (see data/README.md).")
            elif file_sha256(path) != expected:
                raise ValueError(f"{path} does not match its recorded checksum in SHA256SUMS. "
                                 f"Restore the committed file with `git checkout -- data/raw/{path.name}`.")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=adjusted,
                         progress=False, multi_level_index=False)
        if df.empty:
            raise RuntimeError(f"No data downloaded for {ticker}")
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"Downloaded {ticker} to {path}. Commit it and add its checksum to "
              f"data/raw/SHA256SUMS so teammates use the same prices.")
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
    daily = log_close.diff()
    out["ret5"] = log_close - log_close.shift(5)
    out["ret20"] = log_close - log_close.shift(20)
    out["vol20"] = daily.rolling(20).std()
    return out.dropna()


def slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with ``start <= date <= end`` (both inclusive)."""
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def split_by_date(features: pd.DataFrame, cfg: DataConfig = DataConfig()) -> dict:
    """Chronological split into ``train``, ``val`` and a list of ``test`` windows."""
    return {
        "train": slice_dates(features, *cfg.train),
        "val": slice_dates(features, *cfg.val),
        "test": [slice_dates(features, s, e) for s, e in cfg.test_windows],
    }


def load_ticker(ticker: str, cfg: DataConfig = DataConfig()) -> dict:
    """Download (or read cached) data for one ticker, add features, and split."""
    prices = load_price_data(ticker, cfg.download_start, cfg.download_end,
                             adjusted=cfg.adjusted)
    return split_by_date(compute_market_features(prices), cfg)


class TradingEnv(gym.Env):
    """Single-stock trading environment with discrete trade sizing.

    Parameters
    ----------
    features:
        DataFrame indexed by date with ``Close`` and the market features.
    cfg:
        Environment settings (initial cash, fee rate, share rounding).
    episode_len:
        If given, each ``reset`` starts at a random day and the episode lasts
        ``episode_len`` trading days (used for training). If ``None``, the
        episode covers the whole DataFrame (used for validation and test).

    An episode over N trading days has N - 1 steps: orders fill at the close
    of days 0 .. N-2 and are marked at the close of days 1 .. N-1. The end of
    an episode is a time limit, so it is reported as ``truncated`` (agents
    should still bootstrap from the final state).
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, cfg: EnvConfig = EnvConfig(),
                 episode_len: int | None = None):
        super().__init__()
        if episode_len is not None and episode_len > len(features):
            raise ValueError("episode_len is longer than the data")
        self.cfg = cfg
        self.episode_len = episode_len
        self.dates = features.index
        self.close = features["Close"].to_numpy(dtype=np.float64)
        self.market = features[MARKET_FEATURES].to_numpy(dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),),
                                                dtype=np.float32)
        self._t = self._end = 0
        self.cash = self.shares = 0.0

    # ----------------------------------------------------------------- helpers
    def portfolio_value(self, t: int | None = None) -> float:
        t = self._t if t is None else t
        return self.cash + self.shares * self.close[t]

    def _obs(self) -> np.ndarray:
        value = self.portfolio_value()
        pos = self.shares * self.close[self._t] / value if value > 0 else 0.0
        return np.append(self.market[self._t], np.float32(pos)).astype(np.float32)

    def _info(self, fee: float = 0.0, traded_shares: float = 0.0) -> dict:
        return {
            "date": self.dates[self._t],
            "portfolio_value": self.portfolio_value(),
            "cash": self.cash,
            "shares": self.shares,
            "price": self.close[self._t],
            "fee": fee,
            "traded_shares": traded_shares,
        }

    def _execute(self, action: int, price: float) -> tuple[float, float]:
        """Fill the order at ``price``. Returns (fee paid, signed shares traded)."""
        side, frac = ACTIONS[int(action)]
        fee_rate = self.cfg.fee_rate
        if side == "buy":
            # Spend frac * cash in total, fee included.
            shares = frac * self.cash / (price * (1.0 + fee_rate))
            if not self.cfg.fractional_shares:
                shares = float(np.floor(shares))
            if shares * price < MIN_TRADE_VALUE:
                return 0.0, 0.0
            cost = shares * price
            fee = cost * fee_rate
            self.cash = max(self.cash - cost - fee, 0.0)  # clamp float round-off
            self.shares += shares
            return fee, shares
        if side == "sell":
            shares = self.shares if frac >= 1.0 else frac * self.shares
            if not self.cfg.fractional_shares and frac < 1.0:
                shares = float(np.floor(shares))
            if shares * price < MIN_TRADE_VALUE:
                return 0.0, 0.0
            proceeds = shares * price
            fee = proceeds * fee_rate
            self.cash += proceeds - fee
            self.shares -= shares
            return fee, -shares
        return 0.0, 0.0  # hold, or nothing to buy / sell with

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        n = len(self.close)
        if self.episode_len is None:
            start, length = 0, n
        else:
            start = int(self.np_random.integers(0, n - self.episode_len + 1))
            length = self.episode_len
        self._t, self._end = start, start + length - 1
        self.cash, self.shares = float(self.cfg.initial_cash), 0.0
        return self._obs(), self._info()

    def step(self, action):
        if self._t >= self._end:
            raise RuntimeError("Episode is over; call reset()")
        value_before = self.portfolio_value()  # marked at day t's close
        fee, traded = self._execute(action, self.close[self._t])
        self._t += 1
        value_after = self.portfolio_value()  # marked at day t+1's close
        reward = float(np.log(value_after / value_before))
        truncated = self._t >= self._end
        return self._obs(), reward, False, truncated, self._info(fee, traded)
