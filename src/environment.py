"""Historical stock-trading environment.

See ``docs/proposal.md`` Sections 5 and 7.

Responsibilities:
    * Load daily historical market data (price, volume, ...) for one stock.
    * Compute state features: market features (e.g. 5-day return, 20-day
      return, 20-day volatility) and portfolio features (cash fraction,
      position fraction).
    * Split data chronologically into train / validation / test periods.
    * Provide a Gymnasium-compatible environment with the 9-action discrete
      trade-sizing space, either by wrapping an existing environment (e.g.
      Gym Trading Env) with an action adapter or with a minimal custom env.

Rules the implementation must respect:
    * One stock per episode. No leverage and no short selling.
    * Transaction costs are deducted, so they show up in the portfolio value.
    * Reward is the per-step log return of portfolio value,
      ``r_t = ln(V_t / V_{t-1})``, which requires ``V_t > 0``.
    * Define explicitly when market data are observed and when orders are
      executed, so that no action uses information from the future
      (no look-ahead bias).
    * Normalization statistics are fit on the training split only.

Still open (see ``docs/proposal.md`` Section 12): data source and tickers,
which environment to use, fractional vs. integer shares, commission model,
execution-price rule, and what happens when cash cannot cover fees. Keep
these as configuration, not hard-coded constants.
"""

from __future__ import annotations

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


def load_price_data(ticker, start, end):
    """Load daily historical market data for one ticker.

    TODO: choose the data source (e.g. yfinance) and cache raw files under
    ``data/`` so runs are reproducible without re-downloading.
    """
    raise NotImplementedError


def split_by_date(df, train_end, val_end):
    """Split a time-indexed DataFrame chronologically into train / val / test.

    TODO: default boundaries are tentative (train 2015-2021, val 2022,
    test 2023-2025). The test period may be cut into fixed 12-month windows.
    """
    raise NotImplementedError


def compute_market_features(df):
    """Compute market state features from price data.

    TODO: e.g. 5-day return, 20-day return, 20-day volatility. Use only data
    available at observation time.
    """
    raise NotImplementedError


class TradingEnv:
    """Single-stock trading environment with discrete trade sizing.

    TODO: implement the Gymnasium ``reset`` / ``step`` API (or wrap an
    existing environment). ``step`` should turn an action from ``ACTIONS``
    into a legal order, update cash, holdings, and portfolio value after
    fees, and return the log-return reward. ``info`` should expose the
    portfolio value, trade size, and fees so evaluation can log them.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
