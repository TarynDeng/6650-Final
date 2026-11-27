"""Figures for the paper and slides. Every method has one color everywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .environment import ACTIONS, N_ACTIONS  # noqa: E402

# Same colors as the slide deck (slides/build_deck.js).
METHOD_COLORS = {"buy_and_hold": "#8A8F8C", "qlearning": "#1F6E4E", "sarsa": "#6FA287",
                 "dqn": "#2F6F8F", "weighted_ens": "#E0A43A", "equal_ens": "#C9772B"}
ENSEMBLES = {"weighted_ens", "equal_ens"}
LOSS_COLORS = ["#6FA287", "#E0A43A", "#C9772B", "#8E3B2F"]
LOSS_LABELS = ["1: profit / break-even", "2: loss up to 20%", "3: loss 20-40%", "4: loss over 40%"]


def _save(fig, out_path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves_by_ticker(logs: pd.DataFrame, out_path, val_label: str,
                                   benchmarks: dict | None = None) -> None:
    """One panel per ticker: validation return vs. env steps, mean +/- std over seeds.

    ``benchmarks`` maps ticker -> buy-and-hold validation return (dashed line).
    """
    ev = logs.dropna(subset=["eval_return"])
    tickers = sorted(ev["ticker"].unique())
    fig, axes = plt.subplots(1, len(tickers), figsize=(4.5 * len(tickers), 3.8), squeeze=False)
    for ax, ticker in zip(axes[0], tickers, strict=True):
        for algo, d in ev[ev["ticker"] == ticker].groupby("algo"):
            stats = d.groupby("env_step")["eval_return"].agg(["mean", "std"]).fillna(0)
            color = METHOD_COLORS.get(algo)
            ax.plot(stats.index, stats["mean"], color=color, label=f"{algo} (n={d['seed'].nunique()} seeds)")
            ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                            color=color, alpha=0.15)
        if benchmarks and ticker in benchmarks:
            ax.axhline(benchmarks[ticker], ls="--", color=METHOD_COLORS["buy_and_hold"], label="buy-and-hold")
        ax.set(title=f"{ticker}: validation ({val_label})", xlabel="Environment steps",
               ylabel="Validation terminal return")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)


def plot_portfolio_values(trajectories: dict, out_path, title: str = "") -> None:
    """Portfolio value over time, one line per method (ensembles dashed), V_0 = 1."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, res in trajectories.items():
        ax.plot(res["dates"], res["values"] / res["values"][0], label=name,
                color=METHOD_COLORS.get(name), ls="--" if name in ENSEMBLES else "-")
    ax.axhline(1, ls=":", color="gray", lw=1)
    ax.set(xlabel="Date", ylabel="Portfolio value / initial capital", title=title)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _save(fig, out_path)


def plot_method_returns(results: pd.DataFrame, order, out_path) -> None:
    """Mean test return per method and ticker; error bars: spread of the seed means."""
    tickers = sorted(results["ticker"].unique())
    seed_means = results.groupby(["method", "ticker", "seed"])["terminal_return"].mean()
    stats = seed_means.groupby(["method", "ticker"]).agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(tickers)), 4))
    width = 0.8 / len(order)
    x = np.arange(len(tickers))
    for i, m in enumerate(order):
        mean = [stats.loc[(m, t), "mean"] if (m, t) in stats.index else np.nan for t in tickers]
        err = [stats.loc[(m, t), "std"] if (m, t) in stats.index else 0 for t in tickers]
        ax.bar(x + (i - (len(order) - 1) / 2) * width, mean, width, yerr=err, capsize=2, label=m,
               color=METHOD_COLORS.get(m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, tickers)
    ax.set(ylabel="Mean test terminal return",
           title="Test terminal return per stock (mean over test years; bars: std across seeds)")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, out_path)


def plot_loss_categories(results: pd.DataFrame, order, out_path) -> None:
    """Share of test episodes (stock-years x seeds) in each terminal-loss category."""
    counts = (results.pivot_table(index="method", columns="loss_category", values="terminal_return",
                                  aggfunc="count", fill_value=0)
              .reindex(index=order, columns=[1, 2, 3, 4], fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(share))
    for c, col, lab in zip([1, 2, 3, 4], LOSS_COLORS, LOSS_LABELS, strict=True):
        ax.bar(share.index, share[c], bottom=bottom, color=col, label=lab)
        bottom += share[c].to_numpy()
    ax.set(ylabel="Share of test episodes", title="Terminal-loss categories (not max drawdown)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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
            value = freq.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Share of test steps")
    ax.set_title("Action frequencies on the test years")
    _save(fig, out_path)
