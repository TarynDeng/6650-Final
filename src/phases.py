"""Phase 1 (train and evaluate the agents) and phase 2 (ensembles).

Outputs go under ``results/``, named ``{algo}_{TICKER}_seed{n}`` (see
``results/README.md``). Phase 3 (final tables and figures) is in
``src/report.py``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig
from .data import load_ticker
from .deep_rl import DQNAgent, Standardizer
from .ensemble import EnsemblePolicy, equal_weights, softmax_weights
from .environment import TradingEnv
from .evaluation import buy_and_hold_policy, result_row, run_policy, summarize
from .plots import plot_learning_curves_by_ticker, plot_portfolio_values
from .runs import ALGORITHMS, check_results_dir, phase1_settings, write_manifest
from .tabular_rl import QLearningAgent, SarsaAgent, StateDiscretizer
from .training import train_agent

ENSEMBLES = ("weighted_ens", "equal_ens")


def make_agent(algo: str, splits: dict, cfg: ExperimentConfig, seed: int):
    """Build an agent whose preprocessing is fit on the training split only.

    Returns ``(agent, (eps_start, eps_end, eps_decay_frac), file_extension)``.
    """
    if algo in ("qlearning", "sarsa"):
        t = cfg.tabular
        disc = StateDiscretizer(t.n_bins, t.n_position_levels).fit(splits["train"])
        cls = QLearningAgent if algo == "qlearning" else SarsaAgent
        agent = cls(disc, gamma=cfg.train.gamma, cfg=t, seed=seed)
        return agent, (t.epsilon_start, t.epsilon_end, t.epsilon_decay_frac), "pkl"
    d = cfg.dqn
    agent = DQNAgent(Standardizer().fit(splits["train"]), gamma=cfg.train.gamma, cfg=d, seed=seed)
    return agent, (d.epsilon_start, d.epsilon_end, d.epsilon_decay_frac), "pt"


def evaluate_all(make_policy, splits: dict, cfg: ExperimentConfig, labels: dict):
    """Greedy evaluation on the validation year and every test year.

    ``make_policy()`` returns a fresh ``policy(obs) -> action`` for each run.
    Returns ``(rows, trajectories)``; trajectories are keyed by (split, year).
    """
    rows, trajectories = [], {}
    windows = [("val", cfg.data.val, splits["val"])]
    windows += [("test", w, df) for w, df in zip(cfg.data.test_windows, splits["test"], strict=True)]
    for split, (start, _), df in windows:
        res = run_policy(TradingEnv(df, cfg.env), make_policy())
        rows.append(result_row(res, split=split, window=start[:4], **labels))
        trajectories[(split, start[:4])] = res
    return rows, trajectories


def greedy(agent):
    """Policy factory for ``evaluate_all``: the agent's greedy action."""
    return lambda: (lambda obs: agent.act(obs, explore=False))


def load_members(ticker: str, seed: int, splits: dict, cfg: ExperimentConfig, models_dir: Path):
    """Load the three phase 1 agents trained with ``seed`` on ``ticker``."""
    members = {}
    for algo in ALGORITHMS:
        agent, _, ext = make_agent(algo, splits, cfg, seed)
        agent.load(models_dir / f"{algo}_{ticker}_seed{seed}.{ext}")
        members[algo] = agent
    return members


def _clear_results(out: Path) -> None:
    """Phase 1 starts a new experiment: delete the previous run's outputs."""
    for sub in ("models", "logs", "tables", "figures"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir(parents=True)
    (out / "run_manifest.json").unlink(missing_ok=True)


def _train_one(algo, ticker, seed, splits, cfg, steps, out):
    """Train, save and evaluate one run; returns (log, eval rows, trajectories)."""
    run = f"{algo}_{ticker}_seed{seed}"
    agent, eps, ext = make_agent(algo, splits, cfg, seed)
    t0 = time.time()
    log = train_agent(agent, TradingEnv(splits["train"], cfg.env, episode_len=cfg.env.train_episode_len),
                      total_steps=steps, eps_start=eps[0], eps_end=eps[1], eps_decay_frac=eps[2],
                      seed=seed, val_env=TradingEnv(splits["val"], cfg.env),
                      eval_every=cfg.train.eval_every, keep_best=cfg.train.keep_best_on_val, run_name=run)
    log["algo"], log["ticker"], log["seed"] = algo, ticker, seed
    log.to_csv(out / "logs" / f"{run}.csv", index=False)
    agent.save(out / "models" / f"{run}.{ext}")
    rows, traj = evaluate_all(greedy(agent), splits, cfg, {"method": algo, "ticker": ticker, "seed": seed})
    test_mean = pd.DataFrame(rows).query("split == 'test'")["terminal_return"].mean()
    print(f"  {run:<24} {time.time() - t0:6.1f}s  val {log.attrs['val_return']:+.3f}"
          f"  mean test {test_mean:+.3f}")
    return log, rows, traj


def run_phase1(args) -> None:
    """Train Q-learning, SARSA and DQN; evaluate them and buy-and-hold."""
    cfg = ExperimentConfig()
    tickers = args.tickers or list(cfg.data.tickers)
    seeds = args.seeds if args.seeds is not None else list(cfg.train.seeds)
    steps = args.steps if args.steps is not None else cfg.train.total_env_steps
    out = args.results_dir
    check_results_dir(out)
    # Validate every ticker before deleting anything from the previous run.
    all_splits = {t: load_ticker(t, cfg.data) for t in tickers}
    _clear_results(out)
    torch.set_num_threads(1)
    eval_rows, logs = [], []
    for ticker, splits in all_splits.items():
        print(f"== {ticker}: train {len(splits['train'])} days, val {len(splits['val'])}, "
              f"test {[len(d) for d in splits['test']]}")
        rows, bh = evaluate_all(buy_and_hold_policy, splits, cfg,
                                {"method": "buy_and_hold", "ticker": ticker, "seed": -1})
        eval_rows += rows
        sample = {"buy_and_hold": bh}
        for seed in seeds:
            for algo in args.algos:
                log, rows, traj = _train_one(algo, ticker, seed, splits, cfg, steps, out)
                logs.append(log)
                eval_rows += rows
                if seed == seeds[0]:
                    sample[algo] = traj
        for split, year in bh:
            plot_portfolio_values({m: t[(split, year)] for m, t in sample.items()},
                                  out / "figures" / f"phase1_values_{ticker}_{split}{year}.png",
                                  title=f"{ticker} {split} {year} (seed {seeds[0]})")
    _write_phase1_tables(out, cfg, eval_rows, logs)
    # Written last, so it only exists when phase 1 finished.
    write_manifest(out, tickers, seeds, args.algos, steps, cfg)


def _write_phase1_tables(out: Path, cfg: ExperimentConfig, eval_rows, logs) -> None:
    results = pd.DataFrame(eval_rows)
    for split in ("val", "test"):
        results.query("split == @split").to_csv(out / "tables" / f"phase1_{split}.csv", index=False)
    all_logs = pd.concat(logs, ignore_index=True)
    bh_val = results.query("method == 'buy_and_hold' and split == 'val'")
    plot_learning_curves_by_ticker(all_logs, out / "figures" / "phase1_learning_curves.png",
                                   val_label=cfg.data.val[0][:4],
                                   benchmarks=bh_val.set_index("ticker")["terminal_return"].to_dict())
    print("\nTest years, terminal return by method:")
    print(summarize(results.query("split == 'test'")).round(3).to_string())


def ensemble_weights(val_table: pd.DataFrame, ticker: str, seed: int, cfg: ExperimentConfig) -> dict:
    """Validation returns of the three members, and both ensembles' weights."""
    val_ret = [val_table.query("method == @a and ticker == @ticker and seed == @seed")
               ["terminal_return"].item() for a in ALGORITHMS]
    return {"val_returns": val_ret,
            "weighted_ens": softmax_weights(val_ret, cfg.ensemble.temperature),
            "equal_ens": equal_weights(len(ALGORITHMS))}


def _weight_row(name, ticker, seed, w, val_ret) -> dict:
    top = int(w.argmax())
    return {"method": name, "ticker": ticker, "seed": seed,
            **{f"val_{a}": r for a, r in zip(ALGORITHMS, val_ret, strict=True)},
            **{f"w_{a}": x for a, x in zip(ALGORITHMS, w, strict=True)},
            "top_member": ALGORITHMS[top], "top_weight": float(w[top]),
            "follows_top_member": bool(w[top] > 0.5)}


def run_phase2(args) -> None:
    """Weighted and equal-weight ensembles from the phase 1 agents.

    For each stock and seed, the members are the Q-learning, SARSA and DQN
    agents trained with that seed; weights use the validation year only.
    Also evaluates the equal-weight ensemble under alternative tie-break
    rules as a sensitivity check (``phase2_tiebreak_sensitivity.csv``).
    """
    cfg = ExperimentConfig()
    out = args.results_dir
    tickers, seeds, _ = phase1_settings(args, out, cfg)
    torch.set_num_threads(1)
    val1 = pd.read_csv(out / "tables" / "phase1_val.csv")
    eval_rows, weight_rows, sens_rows = [], [], []
    for ticker in tickers:
        splits = load_ticker(ticker, cfg.data)
        for seed in seeds:
            members = list(load_members(ticker, seed, splits, cfg, out / "models").values())
            w = ensemble_weights(val1, ticker, seed, cfg)
            for name in ENSEMBLES:
                weight_rows.append(_weight_row(name, ticker, seed, w[name], w["val_returns"]))
                ens = EnsemblePolicy(members, w[name], cfg.ensemble.tie_break)
                rows, _ = evaluate_all(lambda e=ens: e.act, splits, cfg,
                                       {"method": name, "ticker": ticker, "seed": seed})
                eval_rows += rows
            for rule in cfg.ensemble.tie_break_sensitivity:
                ens = EnsemblePolicy(members, w["equal_ens"], rule)
                rows, _ = evaluate_all(lambda e=ens: e.act, splits, cfg,
                                       {"method": "equal_ens", "tie_break": rule, "ticker": ticker,
                                        "seed": seed})
                sens_rows += rows
    _write_phase2_tables(out, eval_rows, weight_rows, sens_rows)


def _write_phase2_tables(out: Path, eval_rows, weight_rows, sens_rows) -> None:
    results = pd.DataFrame(eval_rows)
    for split in ("val", "test"):
        results.query("split == @split").to_csv(out / "tables" / f"phase2_{split}.csv", index=False)
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(out / "tables" / "phase2_weights.csv", index=False)
    sens = pd.DataFrame(sens_rows).query("split == 'test'")
    sens.groupby("tie_break")["terminal_return"].agg(["mean", "std", "count"]).to_csv(
        out / "tables" / "phase2_tiebreak_sensitivity.csv")
    w = weights.query("method == 'weighted_ens'")
    print(f"Weighted ensemble follows a single member (top weight > 0.5) in "
          f"{int(w['follows_top_member'].sum())} of {len(w)} stock-seed runs; top members: "
          f"{w['top_member'].value_counts().to_dict()}")
    print("\nTest years, ensembles:")
    print(summarize(results.query("split == 'test'")).round(3).to_string())
