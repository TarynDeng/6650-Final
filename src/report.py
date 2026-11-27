"""Phase 3: final tables and figures comparing all six methods on the test years."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig
from .data import load_ticker
from .ensemble import EnsemblePolicy
from .environment import N_ACTIONS, TradingEnv
from .evaluation import buy_and_hold_policy, paired_comparison, run_policy, summarize
from .phases import ENSEMBLES, ensemble_weights, load_members
from .plots import (
    plot_action_frequencies,
    plot_learning_curves_by_ticker,
    plot_loss_categories,
    plot_method_returns,
    plot_portfolio_values,
)
from .runs import phase1_settings

METHOD_ORDER = ["buy_and_hold", "qlearning", "sarsa", "dqn", "weighted_ens", "equal_ens"]
COMPARISONS = [("qlearning", "buy_and_hold"), ("sarsa", "buy_and_hold"), ("dqn", "buy_and_hold"),
               ("weighted_ens", "buy_and_hold"), ("equal_ens", "buy_and_hold"),
               ("qlearning", "dqn"), ("sarsa", "dqn"), ("qlearning", "sarsa"),
               ("weighted_ens", "equal_ens"), ("equal_ens", "qlearning")]


def _load_test_results(out: Path, tickers, seeds) -> pd.DataFrame:
    """Phase 1 and phase 2 test rows, checked to cover every stock and seed."""
    if not (out / "tables" / "phase2_test.csv").exists():
        sys.exit("Phase 2 results not found: run --phase 2 first.")
    test = pd.concat([pd.read_csv(out / "tables" / f"phase{p}_test.csv") for p in (1, 2)],
                     ignore_index=True)
    expected = {(t, s) for t in tickers for s in seeds}
    for method, d in test.query("seed >= 0").groupby("method"):
        got = set(zip(d["ticker"], d["seed"], strict=True))
        if got != expected:
            sys.exit(f"{method}: results cover {sorted(got)}, expected {sorted(expected)}. "
                     "Re-run phases 1-3 together.")
    return test


def write_tables(test: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary tables; returns (summary by method, paired comparisons)."""
    t = out / "tables"
    order = [m for m in METHOD_ORDER if m in set(test["method"])]
    test.to_csv(t / "phase3_test.csv", index=False)
    by_method = summarize(test).reindex(order)
    by_method.to_csv(t / "phase3_summary_by_method.csv")
    summarize(test, by=("method", "ticker")).to_csv(t / "phase3_summary_by_ticker.csv")
    summarize(test, by=("method", "window")).to_csv(t / "phase3_summary_by_year.csv")
    per_seed = (test.query("seed >= 0").groupby(["method", "ticker", "seed"])["terminal_return"]
                .mean().groupby(["method", "ticker"]).agg(["mean", "std", "min", "max"]))
    per_seed.to_csv(t / "phase3_seed_stability.csv")
    paired = pd.DataFrame([paired_comparison(test, a, b) for a, b in COMPARISONS])
    paired.to_csv(t / "phase3_paired_comparisons.csv", index=False)
    cols = [f"freq_a{a}" for a in range(N_ACTIONS)] + ["n_trades", "total_fees", "avg_position"]
    test.groupby("method")[cols].mean().reindex(order).to_csv(t / "phase3_trading_behavior.csv")
    return by_method, paired


def write_figures(test: pd.DataFrame, out: Path, manifest: dict, cfg: ExperimentConfig) -> None:
    figs = out / "figures"
    order = [m for m in METHOD_ORDER if m in set(test["method"])]
    plot_method_returns(test, order, figs / "phase3_terminal_return.png")
    plot_loss_categories(test, order, figs / "phase3_loss_categories.png")
    behavior = pd.read_csv(out / "tables" / "phase3_trading_behavior.csv", index_col="method")
    plot_action_frequencies(behavior, figs / "phase3_action_frequencies.png")
    logs = pd.concat([pd.read_csv(out / "logs" / f"{run}.csv") for run in manifest["runs"]],
                     ignore_index=True)
    val1 = pd.read_csv(out / "tables" / "phase1_val.csv")
    bh_val = val1.query("method == 'buy_and_hold'").set_index("ticker")["terminal_return"].to_dict()
    plot_learning_curves_by_ticker(logs, figs / "phase3_learning_curves.png",
                                   val_label=cfg.data.val[0][:4], benchmarks=bh_val)


def plot_example_years(out: Path, tickers, seed: int, cfg: ExperimentConfig) -> None:
    """Portfolio value over each test year, all six methods, for one seed."""
    val1 = pd.read_csv(out / "tables" / "phase1_val.csv")
    for ticker in tickers:
        splits = load_ticker(ticker, cfg.data)
        members = load_members(ticker, seed, splits, cfg, out / "models")
        weights = ensemble_weights(val1, ticker, seed, cfg)
        policies = {"buy_and_hold": buy_and_hold_policy}
        policies |= {a: (lambda m=m: (lambda o: m.act(o, explore=False))) for a, m in members.items()}
        for name in ENSEMBLES:
            ens = EnsemblePolicy(list(members.values()), weights[name], cfg.ensemble.tie_break)
            policies[name] = lambda e=ens: e.act
        for (start, _), df in zip(cfg.data.test_windows, splits["test"], strict=True):
            traj = {n: run_policy(TradingEnv(df, cfg.env), make()) for n, make in policies.items()}
            plot_portfolio_values(traj, out / "figures" / f"phase3_values_{ticker}_{start[:4]}.png",
                                  title=f"{ticker} test {start[:4]} (seed {seed}; ensembles dashed)")


def run_phase3(args) -> None:
    """Final out-of-sample comparison of all six methods: tables and figures."""
    cfg = ExperimentConfig()
    out = args.results_dir
    tickers, seeds, manifest = phase1_settings(args, out, cfg)
    torch.set_num_threads(1)
    test = _load_test_results(out, tickers, seeds)
    by_method, paired = write_tables(test, out)
    write_figures(test, out, manifest, cfg)
    plot_example_years(out, tickers, seeds[0], cfg)
    print("Test years, all methods (seeds averaged within each stock-year):")
    print(by_method.round(3).to_string())
    print("\nPaired comparisons over stock-years (a - b):")
    print(paired.round(3).to_string(index=False))
