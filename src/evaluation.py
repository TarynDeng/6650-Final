"""Evaluation: metrics, policy rollouts, buy-and-hold, summary statistics.

See ``docs/proposal.md`` Section 8. Plots are in ``src/plots.py``.

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
    * Stability: spread across seeds, stocks, and test years.
    * Trading behavior: action frequencies, trade count, total fees, and the
      average share of the portfolio held in stock.

A "stock-year" is one stock in one test year (column ``window`` holds the
year). Seeds repeat the same market, so summaries average them first.
"""

from __future__ import annotations

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

    Returns the portfolio-value path (V_0 .. V_T), the actions taken, and
    summary metrics, including ``avg_position``: the mean share of the
    portfolio held in stock at each day's closing value.
    """
    obs, info = env.reset(seed=seed)
    dates, values, actions, positions = [info["date"]], [info["portfolio_value"]], [], []
    fees, n_trades, done = 0.0, 0, False
    while not done:
        a = int(policy(obs))
        obs, _, terminated, truncated, info = env.step(a)
        actions.append(a)
        dates.append(info["date"])
        values.append(info["portfolio_value"])
        positions.append(info["shares"] * info["price"] / info["portfolio_value"])
        fees += info["fee"]
        n_trades += int(info["traded_shares"] != 0)
        done = terminated or truncated
    loss = terminal_loss(values)
    return {
        "dates": pd.DatetimeIndex(dates), "values": np.asarray(values), "actions": np.asarray(actions),
        "terminal_return": terminal_return(values), "terminal_loss": loss,
        "loss_category": loss_category(loss), "n_trades": n_trades, "total_fees": fees,
        "avg_position": float(np.mean(positions)),
        "action_freq": np.bincount(actions, minlength=N_ACTIONS) / max(len(actions), 1),
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
    for k in ("terminal_return", "terminal_loss", "loss_category", "n_trades", "total_fees",
              "avg_position"):
        row[k] = result[k]
    row["final_value"] = float(result["values"][-1])
    for a in ACTIONS:
        row[f"freq_a{a}"] = float(result["action_freq"][a])
    return row


def window_means(results: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds: one terminal return per (method, ticker, test window).

    Seeds only measure training randomness; each stock-year is one market
    sample. Stock-years are not independent: stocks in the same year share
    the market's move. Summaries and comparisons are built on these.
    """
    return (results.groupby(["method", "ticker", "window"])["terminal_return"].mean()
            .reset_index())


def summarize(results: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    """Summary of terminal returns, with market windows as the unit.

    Columns:
        mean, median, std_windows: over (ticker, window) samples, after
            averaging seeds; n_windows: number of those samples.
        seed_std: std across seeds within a window, averaged over windows
            (training randomness; NaN for methods with one run per window).
        n_episodes: all evaluated episodes (windows x seeds).
        share_cat1..4: share of episodes in each terminal-loss category.
    """
    by = list(by)
    keys = list(dict.fromkeys(by + ["method", "ticker", "window"]))
    win = results.groupby(keys)["terminal_return"].mean().reset_index()
    out = win.groupby(by)["terminal_return"].agg(mean="mean", median="median",
                                                 std_windows="std", n_windows="count")
    out["seed_std"] = results.groupby(keys)["terminal_return"].std().groupby(by).mean()
    out["n_episodes"] = results.groupby(by).size()
    cats = (results.pivot_table(index=by, columns="loss_category", values="terminal_return",
                                aggfunc="count", fill_value=0)
            .reindex(columns=[1, 2, 3, 4], fill_value=0))
    shares = cats.div(cats.sum(axis=1), axis=0)
    shares.columns = [f"share_cat{c}" for c in shares.columns]
    return out.join(shares)


def paired_comparison(results: pd.DataFrame, a: str, b: str, n_boot: int = 10_000,
                      seed: int = 0) -> dict:
    """Compare methods ``a`` and ``b`` stock-year by stock-year (seeds averaged first).

    Returns the mean difference (a - b) in terminal return, how many
    stock-years ``a`` wins, a 95% bootstrap interval that resamples
    stock-years as if independent (``ci_low``/``ci_high``; too narrow, since
    stocks in the same year move together), and the smallest and largest
    of the per-year mean differences (``year_min``/``year_max``). With only a
    few test years, read all of these as rough indications, not proof.
    """
    w = window_means(results).pivot_table(index=["ticker", "window"], columns="method",
                                          values="terminal_return")
    diff = (w[a] - w[b]).dropna()
    d = diff.to_numpy()
    boot = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    yearly = diff.groupby(level="window").mean()
    return {"method_a": a, "method_b": b, "mean_diff": d.mean(),
            "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "year_min": yearly.min(), "year_max": yearly.max(),
            "a_wins": int((d > 0).sum()), "n_windows": len(d), "n_years": len(yearly)}
