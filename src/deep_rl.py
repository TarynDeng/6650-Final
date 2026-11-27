"""Deep RL: DQN with experience replay and a target network.

See ``docs/proposal.md`` Section 4. Adapted from the HW-2 PyTorch DQN
(ReLU MLP, replay memory, Huber loss, AdamW with amsgrad, soft target-network
update with tau = 0.005). Default settings live in ``src/config.py``
(``DQNConfig``).

Scope: the basic DQN taught in the course. No Double / Dueling / Rainbow
extensions, PPO, or SAC.

Rules the implementation respects:
    * Input is the continuous observation from ``TradingEnv``. Market
      features are standardized with training-split statistics only; the
      position fraction (already in [0, 1]) is passed through.
    * The discount factor gamma is shared with the tabular agents.
    * The end of an episode is a time limit (truncation), so the target
      still bootstraps from the final state.

Same agent interface as ``src/tabular_rl.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import DQNConfig
from .environment import MARKET_FEATURES, N_ACTIONS, OBS_NAMES


class Standardizer:
    """Z-score the market features with training-split mean / std."""

    def __init__(self):
        self.mean = self.std = None

    def fit(self, train_features: pd.DataFrame) -> "Standardizer":
        x = train_features[MARKET_FEATURES].to_numpy(dtype=np.float64)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-8
        return self

    def transform(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        out = obs.copy()
        k = len(MARKET_FEATURES)
        out[..., :k] = (obs[..., :k] - self.mean) / self.std
        return out


class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', terminated)."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.size = self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, batch_size)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.actions[idx]),
                torch.as_tensor(self.rewards[idx]), torch.as_tensor(self.next_obs[idx]),
                torch.as_tensor(self.terminated[idx]))


class QNetwork(nn.Module):
    """MLP mapping a state vector to one Q-value per action (ReLU, as in HW-2)."""

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))  # no activation on Q-values
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN with an online network, a soft-updated target network, and replay."""

    name = "dqn"

    def __init__(self, standardizer: Standardizer, *, gamma: float,
                 cfg: DQNConfig = DQNConfig(), seed: int | None = None):
        self.std = standardizer
        self.gamma = gamma
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        dim = len(OBS_NAMES)
        self.policy_net = QNetwork(dim, N_ACTIONS, cfg.hidden_sizes)
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.lr, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.buffer = ReplayBuffer(cfg.buffer_capacity, dim, seed)
        self.n_updates = 0

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        x = torch.as_tensor(self.std.transform(obs)).unsqueeze(0)
        return self.policy_net(x)[0].numpy()

    def act(self, obs, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.q_values(obs)))

    def update(self, obs, action, reward, next_obs, next_action, terminated) -> None:
        """Store the transition, then take one gradient step once warmed up."""
        self.buffer.push(self.std.transform(obs), action, reward,
                         self.std.transform(next_obs), terminated)
        if len(self.buffer) < max(self.cfg.learning_starts, self.cfg.batch_size):
            return
        s, a, r, s2, done = self.buffer.sample(self.cfg.batch_size)
        q_sa = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - done) * self.target_net(s2).max(1).values
        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        # Soft update: theta' <- tau * theta + (1 - tau) * theta'
        with torch.no_grad():
            for p_t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                p_t.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)
        self.n_updates += 1

    def snapshot(self):
        return copy.deepcopy(self.policy_net.state_dict())

    def restore(self, snapshot) -> None:
        self.policy_net.load_state_dict(snapshot)
        self.target_net.load_state_dict(snapshot)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy_net": self.policy_net.state_dict(), "mean": self.std.mean,
                    "std": self.std.std, "gamma": self.gamma}, path)

    def load(self, path) -> None:
        d = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(d["policy_net"])
        self.target_net.load_state_dict(d["policy_net"])
        self.std.mean, self.std.std = d["mean"], d["std"]
