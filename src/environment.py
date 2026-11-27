"""Historical stock-trading environment (see ``docs/proposal.md`` Section 5).

We use a small custom Gymnasium environment instead of an existing trading
library: Gym Trading Env's actions are target positions rather than "buy with
x% of cash / sell x% of holdings" orders, and a short custom environment is
easier to verify (see ``tests/test_environment.py``). Data loading lives in
``src/data.py``; settings in ``src/config.py`` (``EnvConfig``).

Rules the implementation respects:
    * One stock per episode. No leverage and no short selling.
    * Transaction costs are deducted, so they show up in the portfolio value.
    * Reward is the per-step log return of portfolio value,
      ``r_t = ln(V_{t+1} / V_t)``, so the rewards of an episode sum to
      ``ln(V_T / V_0)`` with fees included.
    * Timing: the observation at day t uses closes up to and including day t,
      the order fills at day t's close, and the portfolio is marked to market
      at day t+1's close. No action uses future information.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from .config import EnvConfig
from .data import MARKET_FEATURES

# The 9 discrete trade-sizing actions (proposal Section 5.4).
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

# Observation vector: market features followed by the position fraction
# (market value of holdings / portfolio value). The cash fraction is
# 1 - position fraction, so it is not included separately.
OBS_NAMES = MARKET_FEATURES + ["position_frac"]

# Trades worth less than this (in dollars) are skipped, so float round-off
# leftovers never count as trades.
MIN_TRADE_VALUE = 0.01


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

    def __init__(self, features: pd.DataFrame, cfg: EnvConfig | None = None,
                 episode_len: int | None = None):
        super().__init__()
        cfg = cfg or EnvConfig()
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
