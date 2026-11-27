"""One training loop shared by Q-learning, SARSA, and DQN.

Every algorithm gets the same environment, step budget, epsilon schedule,
discount factor, and validation checks, so learning curves are comparable in
environment steps (``docs/proposal.md`` Section 6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluation import run_policy


def linear_epsilon(step: int, total: int, start: float, end: float, decay_frac: float) -> float:
    """Exploration rate: linear from ``start`` to ``end`` over ``decay_frac`` of training."""
    decay_steps = max(1, int(decay_frac * total))
    return end + (start - end) * max(0.0, 1.0 - step / decay_steps)


@dataclass
class _Progress:
    """Counters and log rows shared by the helpers below."""

    total_steps: int
    eps: tuple[float, float, float]  # start, end, decay fraction
    step: int = 0
    episode: int = 0
    t0: float = field(default_factory=time.time)
    rows: list = field(default_factory=list)

    def epsilon(self) -> float:
        return linear_epsilon(self.step, self.total_steps, *self.eps)

    def log(self, **row) -> None:
        self.rows.append({"env_step": self.step, "episode": self.episode,
                          "wall_time": time.time() - self.t0, **row})


class _Validator:
    """Greedy runs on the validation split; optionally remembers the best snapshot."""

    def __init__(self, agent, val_env):
        self.agent, self.val_env = agent, val_env
        self.best, self.best_snapshot, self.last = -np.inf, None, np.nan

    def __call__(self, progress: _Progress) -> None:
        res = run_policy(self.val_env, lambda o: self.agent.act(o, explore=False))
        self.last = res["terminal_return"]
        progress.log(train_return=np.nan, eval_return=self.last, epsilon=self.agent.epsilon,
                     complete=np.nan)
        if self.last > self.best:
            self.best, self.best_snapshot = self.last, self.agent.snapshot()


def _run_episode(agent, env, seed: int, progress: _Progress, validate, eval_every: int) -> None:
    """One training episode (cut short if the step budget runs out)."""
    obs, info = env.reset(seed=seed * 1_000_003 + progress.episode)
    v0 = info["portfolio_value"]
    agent.set_epsilon(progress.epsilon())
    action = agent.act(obs, explore=True)
    done = False
    while not done and progress.step < progress.total_steps:
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        progress.step += 1
        agent.set_epsilon(progress.epsilon())
        # Always pick the next action: SARSA bootstraps from it even when the
        # window ends, because the window end is a time limit, not a real
        # termination. Q-learning and DQN ignore it.
        next_action = agent.act(next_obs, explore=True)
        agent.update(obs, action, reward, next_obs, next_action, terminated)
        obs, action = next_obs, next_action
        if validate is not None and progress.step % eval_every == 0:
            validate(progress)
    progress.log(train_return=info["portfolio_value"] / v0 - 1.0, eval_return=np.nan,
                 epsilon=agent.epsilon, complete=done)
    progress.episode += 1


def train_agent(agent, train_env, *, total_steps: int, eps_start: float, eps_end: float,
                eps_decay_frac: float, seed: int, val_env=None, eval_every: int = 5_000,
                keep_best: bool = False, run_name: str = "") -> pd.DataFrame:
    """Train ``agent`` for ``total_steps`` environment steps.

    Episodes are random fixed-length windows of the training period. Every
    ``eval_every`` steps the greedy policy is run on ``val_env`` (monitoring
    only). With ``keep_best=True`` the snapshot with the best validation
    return is restored at the end; the experiments use the final model.

    Returns a log with one row per training episode and per validation check:
    ``env_step, episode, train_return, eval_return, epsilon, complete,
    wall_time, run``. ``train_return`` is the episode's terminal return
    (V_T - V_0) / V_0; ``complete`` is False for the last episode if the step
    budget cut it short. ``log.attrs["val_return"]`` is the validation return
    of the returned model.
    """
    progress = _Progress(total_steps, (eps_start, eps_end, eps_decay_frac))
    validate = _Validator(agent, val_env) if val_env is not None else None
    while progress.step < total_steps:
        _run_episode(agent, train_env, seed, progress, validate, eval_every)
    if validate is not None and progress.step % eval_every != 0:
        progress.episode -= 1  # the final check belongs to the last episode
        validate(progress)
        progress.episode += 1
    if keep_best and validate is not None and validate.best_snapshot is not None:
        agent.restore(validate.best_snapshot)
    log = pd.DataFrame(progress.rows)
    log["run"] = run_name
    if validate is not None:
        log.attrs["val_return"] = validate.best if keep_best else validate.last
    return log
