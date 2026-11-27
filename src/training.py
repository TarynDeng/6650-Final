"""One training loop shared by Q-learning, SARSA, and DQN.

Every algorithm gets the same environment, step budget, epsilon schedule
shape, discount factor, and validation procedure, so learning curves are
comparable in environment steps (``docs/proposal.md`` Section 6).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluation import run_policy


def linear_epsilon(step: int, total: int, start: float, end: float, decay_frac: float) -> float:
    decay_steps = max(1, int(decay_frac * total))
    return end + (start - end) * max(0.0, 1.0 - step / decay_steps)


def train_agent(agent, train_env, *, total_steps: int, eps_start: float, eps_end: float,
                eps_decay_frac: float, seed: int, val_env=None, eval_every: int = 5_000,
                keep_best: bool = True, run_name: str = "") -> pd.DataFrame:
    """Train ``agent`` for ``total_steps`` environment steps.

    Episodes are random fixed-length windows of the training period. Every
    ``eval_every`` steps the greedy policy is run on ``val_env``; if
    ``keep_best`` is set, the snapshot with the best validation terminal
    return is restored at the end.

    Returns a log with one row per training episode and per validation check:
    ``env_step, episode, train_return, eval_return, epsilon, wall_time``
    (``train_return`` is the episode's terminal return, (V_T - V_0) / V_0).
    """
    rows, step, episode, t0 = [], 0, 0, time.time()
    best_val, best_snap = -np.inf, None

    def validate():
        nonlocal best_val, best_snap
        res = run_policy(val_env, lambda o: agent.act(o, explore=False))
        rows.append({"env_step": step, "episode": episode, "train_return": np.nan,
                     "eval_return": res["terminal_return"], "epsilon": agent.epsilon,
                     "wall_time": time.time() - t0})
        if res["terminal_return"] > best_val:
            best_val, best_snap = res["terminal_return"], agent.snapshot()

    while step < total_steps:
        obs, info = train_env.reset(seed=seed * 1_000_003 + episode)
        v0 = info["portfolio_value"]
        agent.set_epsilon(linear_epsilon(step, total_steps, eps_start, eps_end, eps_decay_frac))
        action = agent.act(obs, explore=True)
        done = False
        while not done and step < total_steps:
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            step += 1
            agent.set_epsilon(linear_epsilon(step, total_steps, eps_start, eps_end, eps_decay_frac))
            next_action = agent.act(next_obs, explore=True)
            # Always pick the next action: SARSA bootstraps from it even when the
            # window ends, because the window end is a time limit, not a real
            # termination. Q-learning and DQN ignore it.
            agent.update(obs, action, reward, next_obs, next_action, terminated)
            obs, action = next_obs, next_action
            if val_env is not None and step % eval_every == 0:
                validate()
        rows.append({"env_step": step, "episode": episode,
                     "train_return": info["portfolio_value"] / v0 - 1.0, "eval_return": np.nan,
                     "epsilon": agent.epsilon, "wall_time": time.time() - t0})
        episode += 1

    if val_env is not None and step % eval_every != 0:
        validate()
    if keep_best and best_snap is not None:
        agent.restore(best_snap)
    log = pd.DataFrame(rows)
    log["run"] = run_name
    log.attrs["best_val_return"] = best_val
    return log
