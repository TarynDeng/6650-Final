"""Evaluation metrics, policy rollouts, baselines, and plots.

See ``docs/proposal.md`` Section 8.

Metrics:
    * Terminal return (reconstructed): ``R_T = (V_T - V_0) / V_0``.
    * Terminal loss (reconstructed): ``L_T = max(0, -R_T)``, classified at
      the end of each test window into
        Category 1: profit or break-even (L_T == 0)
        Category 2: 0% < L_T <= 20%
        Category 3: 20% < L_T <= 40%
        Category 4: L_T > 40%
      Terminal loss is NOT maximum drawdown; drawdown is never used as a
      substitute.
    * Stability: mean and std across seeds, stocks, and test windows.
    * Trading behavior: action frequencies, trade count, total fees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .environment import ACTIONS, N_ACTIONS


def terminal_return(portfolio_values) -> float:
    v = np.asarray(portfolio_values, dtype=float)
    return float((v[-1] - v[0]) / v[0])


def terminal_loss(portfolio_values) -> float:
    return max(0.0, -terminal_return(portfolio_values))


def loss_category(loss: float) -> int:
    """Terminal-loss category 1-4 (see module docstring)."""
    if loss <= 0.0:
        return 1
    if loss <= 0.20:
        return 2
    if loss <= 0.40:
        return 3
    return 4


def run_policy(env, policy, seed: int = 0) -> dict:
    """Roll out ``policy(obs) -> action`` over one full episode of ``env``.

    Returns the portfolio-value path (V_0 .. V_T), the actions taken, the
    fees paid, and summary metrics.
    """
    obs, info = env.reset(seed=seed)
    dates, values, actions, fees, n_trades = [info["date"]], [info["portfolio_value"]], [], 0.0, 0
    done = False
    while not done:
        a = int(policy(obs))
        obs, _, terminated, truncated, info = env.step(a)
        actions.append(a)
        dates.append(info["date"])
        values.append(info["portfolio_value"])
        fees += info["fee"]
        n_trades += int(info["traded_shares"] != 0)
        done = terminated or truncated
    loss = terminal_loss(values)
    counts = np.bincount(actions, minlength=N_ACTIONS)
    return {
        "dates": pd.DatetimeIndex(dates), "values": np.asarray(values), "actions": np.asarray(actions),
        "terminal_return": terminal_return(values), "terminal_loss": loss,
        "loss_category": loss_category(loss), "n_trades": n_trades, "total_fees": fees,
        "action_freq": counts / max(len(actions), 1),
    }


def buy_and_hold_policy():
    """Non-RL benchmark: buy with 100% of cash at the first step, then hold.

    Runs through the same environment, so it pays the same fees.
    """
    state = {"bought": False}

    def policy(obs):
        if not state["bought"]:
            state["bought"] = True
            return 4
        return 0

    return policy


def result_row(result: dict, **labels) -> dict:
    """Flatten a ``run_policy`` result into one table row."""
    row = dict(labels)
    for k in ("terminal_return", "terminal_loss", "loss_category", "n_trades", "total_fees"):
        row[k] = result[k]
    row["final_value"] = float(result["values"][-1])
    for a, (side, frac) in ACTIONS.items():
        row[f"freq_a{a}"] = float(result["action_freq"][a])
    return row


def summarize(results: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    """Mean / median / std of terminal return and loss-category counts."""
    g = results.groupby(list(by))
    out = g["terminal_return"].agg(["mean", "median", "std", "count"])
    cats = results.pivot_table(index=list(by), columns="loss_category", values="terminal_return",
                               aggfunc="count", fill_value=0)
    cats = cats.reindex(columns=[1, 2, 3, 4], fill_value=0)
    cats.columns = [f"cat{c}" for c in cats.columns]
    return out.join(cats)


def plot_learning_curves(logs: pd.DataFrame, out_path, title: str = "") -> None:
    """Validation terminal return vs. environment steps, mean +/- std over seeds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ev = logs.dropna(subset=["eval_return"])
    for algo, d in ev.groupby("algo"):
        stats = d.groupby("env_step")["eval_return"].agg(["mean", "std"]).fillna(0)
        ax.plot(stats.index, stats["mean"], label=f"{algo} (n={d['seed'].nunique()} seeds)")
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        alpha=0.2)
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.set(xlabel="Environment steps", ylabel="Validation terminal return", title=title)
    ax.legend()
    _save(fig, out_path)


def plot_portfolio_values(trajectories: dict, out_path, title: str = "") -> None:
    """Portfolio value over time, one line per method, normalized to V_0 = 1."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, res in trajectories.items():
        ax.plot(res["dates"], res["values"] / res["values"][0], label=name)
    ax.axhline(1, ls="--", color="gray", lw=1)
    ax.set(xlabel="Date", ylabel="Portfolio value / initial capital", title=title)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _save(fig, out_path)


def _save(fig, out_path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
