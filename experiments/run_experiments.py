"""Entry point for training and evaluation runs.

Usage::

    # Phase 1: train Q-learning, SARSA, and DQN, then evaluate each trained
    # policy and buy-and-hold on the validation year and the test windows.
    python experiments/run_experiments.py --phase 1 --tickers AAPL --seeds 0
    python experiments/run_experiments.py --phase 1                 # all tickers / seeds in config

    python experiments/run_experiments.py --phase 2   # ensembles (not implemented yet)
    python experiments/run_experiments.py --phase 3   # final comparison (not implemented yet)

Phases (``docs/proposal.md`` Section 6):
    1. Train Q-learning, SARSA, and DQN under identical data splits, actions,
       costs, initial capital, reward, and seed logging.
    2. Measure validation performance and fix the weighted / equal-weight
       ensemble weights.
    3. Evaluate all methods plus buy-and-hold on the test windows.

Defaults come from ``src/config.py``. Outputs go under ``results/`` (git-ignored),
named ``{algo}_{TICKER}_seed{n}`` as described in ``results/README.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Allow ``from src import ...`` when run as ``python experiments/run_experiments.py``.
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import ExperimentConfig  # noqa: E402
from src.deep_rl import DQNAgent, Standardizer  # noqa: E402
from src.environment import TradingEnv, load_ticker  # noqa: E402
from src.evaluation import (buy_and_hold_policy, plot_learning_curves,  # noqa: E402
                            plot_portfolio_values, result_row, run_policy, summarize)
from src.tabular_rl import QLearningAgent, SarsaAgent, StateDiscretizer  # noqa: E402
from src.training import train_agent  # noqa: E402

ALGORITHMS = ("qlearning", "sarsa", "dqn")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--algos", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--tickers", nargs="+", help="default: DataConfig.tickers")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: TrainConfig.seeds")
    parser.add_argument("--steps", type=int, help="env steps per run (default: TrainConfig)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def make_agent(algo: str, splits: dict, cfg: ExperimentConfig, seed: int):
    """Build an agent whose preprocessing is fit on the training split only."""
    gamma = cfg.train.gamma
    if algo in ("qlearning", "sarsa"):
        disc = StateDiscretizer(cfg.tabular.n_bins, cfg.tabular.n_position_levels).fit(splits["train"])
        cls = QLearningAgent if algo == "qlearning" else SarsaAgent
        agent = cls(disc, gamma=gamma, cfg=cfg.tabular, seed=seed)
        eps = (cfg.tabular.epsilon_start, cfg.tabular.epsilon_end, cfg.tabular.epsilon_decay_frac)
        ext = "pkl"
    else:
        agent = DQNAgent(Standardizer().fit(splits["train"]), gamma=gamma, cfg=cfg.dqn, seed=seed)
        eps = (cfg.dqn.epsilon_start, cfg.dqn.epsilon_end, cfg.dqn.epsilon_decay_frac)
        ext = "pt"
    return agent, eps, ext


def evaluate_all(policy_fn, splits, cfg, labels) -> tuple[list, dict]:
    """Greedy evaluation on the validation year and every test window."""
    rows, trajectories = [], {}
    windows = [("val", cfg.data.val, splits["val"])]
    windows += [("test", w, df) for w, df in zip(cfg.data.test_windows, splits["test"])]
    for split, (start, _), df in windows:
        res = run_policy(TradingEnv(df, cfg.env), policy_fn())
        rows.append(result_row(res, split=split, window=start[:4], **labels))
        trajectories[(split, start[:4])] = res
    return rows, trajectories


def run_phase1(args):
    cfg = ExperimentConfig()
    tickers = args.tickers or list(cfg.data.tickers)
    seeds = args.seeds if args.seeds is not None else list(cfg.train.seeds)
    steps = args.steps or cfg.train.total_env_steps
    out = args.results_dir
    for sub in ("models", "logs", "tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    run_config = {"config": dataclasses.asdict(cfg), "tickers": tickers, "seeds": seeds,
                  "algos": args.algos, "total_env_steps": steps, "git_commit": git_commit()}
    (out / "logs" / "phase1_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    eval_rows, logs = [], []
    for ticker in tickers:
        splits = load_ticker(ticker, cfg.data)
        print(f"== {ticker}: train {len(splits['train'])} days, val {len(splits['val'])}, "
              f"test {[len(d) for d in splits['test']]}")
        rows, bh_traj = evaluate_all(buy_and_hold_policy, splits, cfg,
                                     {"method": "buy_and_hold", "ticker": ticker, "seed": -1})
        eval_rows += rows
        sample_traj = {"buy_and_hold": bh_traj}

        for seed in seeds:
            for algo in args.algos:
                run = f"{algo}_{ticker}_seed{seed}"
                agent, (e0, e1, ef), ext = make_agent(algo, splits, cfg, seed)
                train_env = TradingEnv(splits["train"], cfg.env, episode_len=cfg.env.train_episode_len)
                val_env = TradingEnv(splits["val"], cfg.env)
                t0 = time.time()
                log = train_agent(agent, train_env, total_steps=steps, eps_start=e0, eps_end=e1,
                                  eps_decay_frac=ef, seed=seed, val_env=val_env,
                                  eval_every=cfg.train.eval_every,
                                  keep_best=cfg.train.keep_best_on_val, run_name=run)
                log["algo"], log["ticker"], log["seed"] = algo, ticker, seed
                log.to_csv(out / "logs" / f"{run}.csv", index=False)
                logs.append(log)
                agent.save(out / "models" / f"{run}.{ext}")
                rows, traj = evaluate_all(lambda a=agent: (lambda o: a.act(o, explore=False)),
                                          splits, cfg, {"method": algo, "ticker": ticker, "seed": seed})
                eval_rows += rows
                if seed == seeds[0]:
                    sample_traj[algo] = traj
                test_mean = pd.DataFrame(rows).query("split == 'test'")["terminal_return"].mean()
                print(f"  {run:<24} {time.time() - t0:6.1f}s  best val {log.attrs['best_val_return']:+.3f}"
                      f"  mean test {test_mean:+.3f}")

        for split, window in sample_traj["buy_and_hold"]:
            plot_portfolio_values({m: t[(split, window)] for m, t in sample_traj.items()},
                                  out / "figures" / f"phase1_values_{ticker}_{split}{window}.png",
                                  title=f"{ticker} {split} {window} (seed {seeds[0]})")

    results = pd.DataFrame(eval_rows)
    for split in ("val", "test"):
        results.query("split == @split").to_csv(out / "tables" / f"phase1_{split}.csv", index=False)
    all_logs = pd.concat(logs, ignore_index=True)
    for ticker in tickers:
        plot_learning_curves(all_logs.query("ticker == @ticker"),
                             out / "figures" / f"phase1_learning_{ticker}.png",
                             title=f"{ticker}: validation return during training")
    print("\nTest windows, terminal return by method:")
    print(summarize(results.query("split == 'test'"), by=("method",)).round(3).to_string())


def run_phase2(args):
    """TODO: compute validation terminal returns and ensemble weights."""
    raise NotImplementedError


def run_phase3(args):
    """TODO: evaluate all methods and buy-and-hold on the test windows."""
    raise NotImplementedError


def main(argv=None):
    args = parse_args(argv)
    {1: run_phase1, 2: run_phase2, 3: run_phase3}[args.phase](args)


if __name__ == "__main__":
    main()
