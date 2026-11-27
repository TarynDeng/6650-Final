"""Tabular RL: state discretization, Q-learning, and SARSA.

See ``docs/proposal.md`` Sections 4 and 5.3. Builds on HW-1. Default
settings live in ``src/config.py`` (``TabularConfig``).

Rules the implementation respects:
    * Discretization bins are fit on the training split only, then reused
      unchanged for validation and test.
    * Epsilon-greedy exploration during training; greedy actions for
      evaluation.
    * Q-learning is off-policy (max over next actions); SARSA is on-policy
      (uses the next action actually taken).
    * The discount factor gamma is shared with DQN (``TrainConfig.gamma``).

All agents in this project share one interface, used by ``src/training.py``
and ``src/evaluation.py``:

    act(obs, explore)                         -> action
    update(obs, action, reward, next_obs, next_action, terminated)
    set_epsilon(epsilon)
    snapshot() / restore(snapshot)            # copy of the learned values
    save(path) / load(path)

``obs`` is always the raw observation from ``TradingEnv``; each agent applies
its own preprocessing (here: the discretizer).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TabularConfig
from .environment import MARKET_FEATURES, N_ACTIONS


class StateDiscretizer:
    """Map an observation to a single integer state.

    Each market feature is cut into quantile bins whose edges come from the
    training split only. The position fraction (0 to 1) is rounded to one of
    ``n_position_levels`` evenly spaced levels.
    """

    def __init__(self, n_bins=(3, 3, 3), n_position_levels=5):
        self.n_bins = tuple(n_bins)
        self.n_position_levels = n_position_levels
        self.edges: list[np.ndarray] | None = None

    @property
    def n_states(self) -> int:
        return int(np.prod(self.n_bins) * self.n_position_levels)

    def fit(self, train_features: pd.DataFrame) -> "StateDiscretizer":
        self.edges = []
        for name, n in zip(MARKET_FEATURES, self.n_bins):
            qs = np.linspace(0, 1, n + 1)[1:-1]  # interior quantiles
            self.edges.append(np.quantile(train_features[name].to_numpy(), qs))
        return self

    def transform(self, obs) -> int:
        if self.edges is None:
            raise RuntimeError("Call fit() on the training split first")
        idx = [int(np.searchsorted(e, obs[i], side="right")) for i, e in enumerate(self.edges)]
        pos = float(np.clip(obs[len(MARKET_FEATURES)], 0.0, 1.0))
        idx.append(int(round(pos * (self.n_position_levels - 1))))
        return int(np.ravel_multi_index(idx, self.n_bins + (self.n_position_levels,)))


class _TabularAgent:
    """Shared Q-table, epsilon-greedy policy, and persistence."""

    name = "tabular"

    def __init__(self, discretizer: StateDiscretizer, *, gamma: float,
                 cfg: TabularConfig = TabularConfig(), seed: int | None = None):
        self.disc = discretizer
        self.gamma = gamma
        self.cfg = cfg
        self.alpha = cfg.alpha
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((discretizer.n_states, N_ACTIONS))

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def act(self, obs, explore: bool = True) -> int:
        s = self.disc.transform(obs)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.q[s]
        if explore:  # random tie-break while learning
            return int(self.rng.choice(np.flatnonzero(q == q.max())))
        return int(np.argmax(q))  # deterministic: lowest index (Hold) on ties

    def _td_update(self, obs, action, target: float) -> None:
        s = self.disc.transform(obs)
        self.q[s, action] += self.alpha * (target - self.q[s, action])

    def snapshot(self):
        return self.q.copy()

    def restore(self, snapshot) -> None:
        self.q = snapshot.copy()

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "edges": self.disc.edges, "n_bins": self.disc.n_bins,
                         "n_position_levels": self.disc.n_position_levels,
                         "gamma": self.gamma}, f)

    def load(self, path) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q, self.disc.edges = d["q"], d["edges"]


class QLearningAgent(_TabularAgent):
    """Tabular Q-learning (off-policy TD control)."""

    name = "qlearning"

    def update(self, obs, action, reward, next_obs, next_action, terminated) -> None:
        bootstrap = 0.0 if terminated else self.gamma * self.q[self.disc.transform(next_obs)].max()
        self._td_update(obs, action, reward + bootstrap)


class SarsaAgent(_TabularAgent):
    """Tabular SARSA (on-policy TD control)."""

    name = "sarsa"

    def update(self, obs, action, reward, next_obs, next_action, terminated) -> None:
        bootstrap = 0.0
        if not terminated:
            bootstrap = self.gamma * self.q[self.disc.transform(next_obs), next_action]
        self._td_update(obs, action, reward + bootstrap)
