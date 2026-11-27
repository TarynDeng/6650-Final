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


def plot_learning_curves_by_ticker(logs: pd.DataFrame, out_path, benchmarks: dict | None = None) -> None:
    """One panel per ticker: validation return vs. env steps, mean +/- std over seeds.

    ``benchmarks`` maps ticker -> buy-and-hold validation return (dashed line).
    """
    ev = logs.dropna(subset=["eval_return"])
    tickers = sorted(ev["ticker"].unique())
    fig, axes = plt.subplots(1, len(tickers), figsize=(4.5 * len(tickers), 3.8), squeeze=False)
    for ax, ticker in zip(axes[0], tickers):
        d_t = ev[ev["ticker"] == ticker]
        for algo, d in d_t.groupby("algo"):
            stats = d.groupby("env_step")["eval_return"].agg(["mean", "std"]).fillna(0)
            ax.plot(stats.index, stats["mean"], color=METHOD_COLORS.get(algo),
                    label=f"{algo} (n={d['seed'].nunique()} seeds)")
            ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                            color=METHOD_COLORS.get(algo), alpha=0.15)
        if benchmarks and ticker in benchmarks:
            ax.axhline(benchmarks[ticker], ls="--", color=METHOD_COLORS["buy_and_hold"],
                       label="buy-and-hold")
        ax.set(title=f"{ticker}: validation (2022)", xlabel="Environment steps",
               ylabel="Validation terminal return")
        ax.legend(fontsize=7)
    fig.tight_layout()
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


METHOD_COLORS = {"buy_and_hold": "#7f7f7f", "qlearning": "#1f77b4", "sarsa": "#ff7f0e",
                 "dqn": "#d62728", "weighted_ens": "#2ca02c", "equal_ens": "#9467bd"}


def plot_method_returns(results: pd.DataFrame, order, out_path) -> None:
    """Mean test terminal return per method and ticker; error bars = std across seeds."""
    tickers = sorted(results["ticker"].unique())
    seed_means = results.groupby(["method", "ticker", "seed"])["terminal_return"].mean()
    stats = seed_means.groupby(["method", "ticker"]).agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(tickers)), 4))
    width = 0.8 / len(order)
    x = np.arange(len(tickers))
    for i, m in enumerate(order):
        vals = [stats.loc[(m, t)] if (m, t) in stats.index else pd.Series({"mean": np.nan, "std": 0})
                for t in tickers]
        ax.bar(x + (i - (len(order) - 1) / 2) * width, [v["mean"] for v in vals], width,
               yerr=[v["std"] for v in vals], capsize=2, label=m, color=METHOD_COLORS.get(m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, tickers)
    ax.set(ylabel="Mean test terminal return", title="Test terminal return (mean over windows; bars = std across seeds)")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, out_path)


def plot_loss_categories(results: pd.DataFrame, order, out_path) -> None:
    """Share of test episodes in each terminal-loss category, per method."""
    counts = (results.pivot_table(index="method", columns="loss_category", values="terminal_return",
                                  aggfunc="count", fill_value=0)
              .reindex(index=order, columns=[1, 2, 3, 4], fill_value=0))
    share = counts.div(counts.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(share))
    colors = ["#2ca02c", "#ffdd57", "#ff7f0e", "#d62728"]
    labels = ["1: profit / break-even", "2: loss <= 20%", "3: loss 20-40%", "4: loss > 40%"]
    for c, col, lab in zip([1, 2, 3, 4], colors, labels):
        ax.bar(share.index, share[c], bottom=bottom, color=col, label=lab)
        bottom += share[c].to_numpy()
    ax.set(ylabel="Share of test episodes", title="Terminal-loss categories (not max drawdown)")
    ax.legend(fontsize=8, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=20)
    _save(fig, out_path)


def plot_action_frequencies(behavior: pd.DataFrame, out_path) -> None:
    """Heatmap of how often each method picks each of the 9 actions on test."""
    freq = behavior[[f"freq_a{a}" for a in range(N_ACTIONS)]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(freq) + 1.5))
    im = ax.imshow(freq.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    names = ["hold"] + [f"{s} {int(f * 100)}%" for s, f in list(ACTIONS.values())[1:]]
    ax.set_xticks(range(N_ACTIONS), names, rotation=45, ha="right")
    ax.set_yticks(range(len(freq)), freq.index)
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            ax.text(j, i, f"{freq.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Share of test steps")
    ax.set_title("Action frequencies on the test windows")
    _save(fig, out_path)


def _save(fig, out_path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
