"""Entry point for training and evaluation runs.

Usage::

    # Phase 1: train Q-learning, SARSA, and DQN, then evaluate each trained
    # policy and buy-and-hold on the validation year and the test windows.
    python experiments/run_experiments.py --phase 1 --tickers AAPL --seeds 0
    python experiments/run_experiments.py --phase 1                 # all tickers / seeds in config

    # Phase 2: weighted and equal-weight ensembles of the Phase 1 agents.
    python experiments/run_experiments.py --phase 2
    # Phase 3: final tables and figures for all six methods on the test windows.
    python experiments/run_experiments.py --phase 3

Phases 2 and 3 read the Phase 1 models and tables, so pass the same
``--tickers`` / ``--seeds`` to every phase.

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
import shutil
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
from src.ensemble import EnsemblePolicy, equal_weights, softmax_weights  # noqa: E402
from src.environment import RAW_DIR, TradingEnv, file_sha256, load_ticker  # noqa: E402
from src.evaluation import (buy_and_hold_policy, plot_action_frequencies,  # noqa: E402
                            plot_learning_curves, plot_learning_curves_by_ticker,
                            plot_loss_categories, plot_method_returns,
                            paired_comparison, plot_portfolio_values, result_row, run_policy,
                            summarize)
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


MANIFEST = "run_manifest.json"


def git_commit() -> str:
    """Short commit hash, with ``-dirty`` if there are uncommitted changes."""
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    import platform

    import gymnasium
    import matplotlib
    import numpy

    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pd.__version__, "gymnasium": gymnasium.__version__,
            "torch": torch.__version__, "matplotlib": matplotlib.__version__}


RESULT_ITEMS = {"models", "logs", "tables", "figures", MANIFEST, "README.md", ".gitkeep", ".DS_Store"}


def read_manifest(out: Path) -> dict:
    path = out / MANIFEST
    if not path.exists():
        sys.exit(f"{path} not found: run --phase 1 first.")
    return json.loads(path.read_text())


def config_dict(cfg: ExperimentConfig) -> dict:
    """The config as plain JSON types, so it compares equal to the saved manifest."""
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


def data_hashes(tickers) -> dict:
    """SHA-256 of the price files actually on disk."""
    return {t: file_sha256(RAW_DIR / f"{t}.csv") for t in tickers}


def check_results_dir(out: Path) -> None:
    """Refuse to clear a folder that does not look like a results folder."""
    if out.exists():
        unexpected = sorted(p.name for p in out.iterdir() if p.name not in RESULT_ITEMS)
        if unexpected:
            sys.exit(f"{out} contains {unexpected[:5]}, so it does not look like a results "
                     "folder. Refusing to clear it; choose an empty or results folder.")


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
    check_results_dir(out)
    # Load and validate every ticker first: a bad ticker or data file must not
    # destroy the previous run's results.
    all_splits = {t: load_ticker(t, cfg.data) for t in tickers}
    # Phase 1 starts a new experiment: clear the previous one so that old
    # models, logs and tables can never be mixed into this run's results.
    for sub in ("models", "logs", "tables", "figures"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir(parents=True)
    (out / MANIFEST).unlink(missing_ok=True)
    torch.set_num_threads(1)

    eval_rows, logs = [], []
    for ticker in tickers:
        splits = all_splits[ticker]
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
                print(f"  {run:<24} {time.time() - t0:6.1f}s  val {log.attrs['val_return']:+.3f}"
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
    # Written last, so it only exists when phase 1 finished. Phases 2 and 3
    # read it to use exactly the same tickers, seeds and runs.
    manifest = {"tickers": tickers, "seeds": seeds, "algos": args.algos,
                "total_env_steps": steps, "git_commit": git_commit(),
                "packages": package_versions(), "data_sha256": data_hashes(tickers),
                "runs": [f"{a}_{t}_seed{s}" for t in tickers for s in seeds for a in args.algos],
                "config": config_dict(cfg)}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print("\nTest windows, terminal return by method:")
    print(summarize(results.query("split == 'test'"), by=("method",)).round(3).to_string())


METHOD_ORDER = ["buy_and_hold", "qlearning", "sarsa", "dqn", "weighted_ens", "equal_ens"]


COMPARISONS = [("qlearning", "buy_and_hold"), ("sarsa", "buy_and_hold"), ("dqn", "buy_and_hold"),
               ("weighted_ens", "buy_and_hold"), ("equal_ens", "buy_and_hold"),
               ("qlearning", "dqn"), ("sarsa", "dqn"), ("qlearning", "sarsa"),
               ("weighted_ens", "equal_ens"), ("equal_ens", "qlearning")]


def _settings(args, out: Path, cfg: ExperimentConfig):
    """Settings of the phase 1 run in ``out``. Phases 2 and 3 must use exactly
    the same stocks, seeds, config and data, or they stop with an error."""
    m = read_manifest(out)
    if set(m["algos"]) != set(ALGORITHMS):
        sys.exit(f"Phase 1 trained only {m['algos']}; the ensembles need all of {ALGORITHMS}.")
    for name, given in (("tickers", args.tickers), ("seeds", args.seeds)):
        if given is not None and sorted(given) != sorted(m[name]):
            sys.exit(f"--{name} {given} differs from phase 1 ({m[name]}). Omit it, or re-run phase 1.")
    now = config_dict(cfg)
    changed = [f"{group}.{key}" for group in now for key in now[group]
               if m["config"].get(group, {}).get(key) != now[group][key]]
    if changed:
        sys.exit(f"src/config.py changed since phase 1 ({', '.join(changed)}). Re-run phase 1.")
    if data_hashes(m["tickers"]) != m["data_sha256"]:
        sys.exit("Price files in data/raw/ changed since phase 1. Re-run phase 1.")
    return m["tickers"], m["seeds"], m


def load_members(ticker: str, seed: int, splits: dict, cfg: ExperimentConfig, models_dir: Path):
    """Load the three Phase 1 agents trained with ``seed`` on ``ticker``."""
    members = {}
    for algo in ALGORITHMS:
        agent, _, ext = make_agent(algo, splits, cfg, seed)
        agent.load(models_dir / f"{algo}_{ticker}_seed{seed}.{ext}")
        members[algo] = agent
    return members


def run_phase2(args):
    """Weighted and equal-weight ensembles from the Phase 1 agents.

    For each ticker and seed, the ensemble members are the Q-learning, SARSA
    and DQN agents trained with that seed. Each member's weight uses its
    terminal return on the validation year (2022) only.
    """
    cfg = ExperimentConfig()
    out = args.results_dir
    tickers, seeds, _ = _settings(args, out, cfg)
    torch.set_num_threads(1)
    val1 = pd.read_csv(out / "tables" / "phase1_val.csv")
    eval_rows, weight_rows = [], []
    for ticker in tickers:
        splits = load_ticker(ticker, cfg.data)
        for seed in seeds:
            members = load_members(ticker, seed, splits, cfg, out / "models")
            val_ret = [val1.query("method == @a and ticker == @ticker and seed == @seed")
                       ["terminal_return"].item() for a in ALGORITHMS]
            ensembles = {
                "weighted_ens": softmax_weights(val_ret, cfg.ensemble.tau),
                "equal_ens": equal_weights(len(ALGORITHMS)),
            }
            for name, w in ensembles.items():
                weight_rows.append({"method": name, "ticker": ticker, "seed": seed,
                                    **{f"val_{a}": r for a, r in zip(ALGORITHMS, val_ret)},
                                    **{f"w_{a}": x for a, x in zip(ALGORITHMS, w)}})
                ens = EnsemblePolicy(list(members.values()), w, cfg.ensemble.tie_break)
                rows, _ = evaluate_all(lambda e=ens: (lambda o: e.act(o)), splits, cfg,
                                       {"method": name, "ticker": ticker, "seed": seed})
                eval_rows += rows
            w = ensembles["weighted_ens"]
            print(f"  {ticker} seed {seed}: weights "
                  + ", ".join(f"{a} {x:.2f}" for a, x in zip(ALGORITHMS, w)))
    results = pd.DataFrame(eval_rows)
    for split in ("val", "test"):
        results.query("split == @split").to_csv(out / "tables" / f"phase2_{split}.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "tables" / "phase2_weights.csv", index=False)
    print("\nTest windows, ensembles:")
    print(summarize(results.query("split == 'test'"), by=("method",)).round(3).to_string())


def run_phase3(args):
    """Final out-of-sample comparison of all six methods: tables and figures."""
    cfg = ExperimentConfig()
    out = args.results_dir
    tickers, seeds, manifest = _settings(args, out, cfg)
    torch.set_num_threads(1)
    if not (out / "tables" / "phase2_test.csv").exists():
        sys.exit("Phase 2 results not found: run --phase 2 first.")
    test = pd.concat([pd.read_csv(out / "tables" / f"phase{p}_test.csv") for p in (1, 2)],
                     ignore_index=True)
    expected = {(t, s) for t in tickers for s in seeds}
    for method, d in test.query("seed >= 0").groupby("method"):
        got = set(zip(d["ticker"], d["seed"]))
        if got != expected:
            sys.exit(f"{method}: results cover {sorted(got)}, expected {sorted(expected)}. "
                     "Re-run phases 1-3 together.")
    test.to_csv(out / "tables" / "phase3_test.csv", index=False)

    order = [m for m in METHOD_ORDER if m in set(test["method"])]
    by_method = summarize(test, by=("method",)).reindex(order)
    by_method.to_csv(out / "tables" / "phase3_summary_by_method.csv")
    summarize(test, by=("method", "ticker")).to_csv(out / "tables" / "phase3_summary_by_ticker.csv")
    summarize(test, by=("method", "window")).to_csv(out / "tables" / "phase3_summary_by_window.csv")
    # Seed stability: per (method, ticker), mean test return of each seed, then
    # the spread of those means across seeds.
    per_seed = (test.query("seed >= 0").groupby(["method", "ticker", "seed"])["terminal_return"]
                .mean().groupby(["method", "ticker"]).agg(["mean", "std", "min", "max"]))
    per_seed.to_csv(out / "tables" / "phase3_seed_stability.csv")
    paired = pd.DataFrame([paired_comparison(test, a, b) for a, b in COMPARISONS])
    paired.to_csv(out / "tables" / "phase3_paired_comparisons.csv", index=False)
    behavior = test.groupby("method")[[f"freq_a{a}" for a in range(9)] + ["n_trades", "total_fees"]].mean()
    behavior.reindex(order).to_csv(out / "tables" / "phase3_trading_behavior.csv")

    figs = out / "figures"
    plot_method_returns(test, order, figs / "phase3_terminal_return.png")
    plot_loss_categories(test, order, figs / "phase3_loss_categories.png")
    plot_action_frequencies(behavior.reindex(order), figs / "phase3_action_frequencies.png")
    logs = pd.concat([pd.read_csv(out / "logs" / f"{run}.csv") for run in manifest["runs"]],
                     ignore_index=True)
    val1 = pd.read_csv(out / "tables" / "phase1_val.csv")
    bh_val = val1.query("method == 'buy_and_hold'").set_index("ticker")["terminal_return"].to_dict()
    plot_learning_curves_by_ticker(logs[logs["ticker"].isin(tickers)],
                                   figs / "phase3_learning_curves.png", benchmarks=bh_val)

    # Portfolio-value curves for the first seed, every method, every test window.
    seed = seeds[0]
    for ticker in tickers:
        splits = load_ticker(ticker, cfg.data)
        members = load_members(ticker, seed, splits, cfg, out / "models")
        val_ret = [val1.query("method == @a and ticker == @ticker and seed == @seed")
                   ["terminal_return"].item() for a in ALGORITHMS]
        policies = {"buy_and_hold": buy_and_hold_policy}
        policies.update({a: (lambda m=m: (lambda o: m.act(o, explore=False))) for a, m in members.items()})
        for name, w in (("weighted_ens", softmax_weights(val_ret, cfg.ensemble.tau)),
                        ("equal_ens", equal_weights(len(ALGORITHMS)))):
            ens = EnsemblePolicy(list(members.values()), w, cfg.ensemble.tie_break)
            policies[name] = lambda e=ens: (lambda o: e.act(o))
        for (start, _), df in zip(cfg.data.test_windows, splits["test"]):
            traj = {n: run_policy(TradingEnv(df, cfg.env), make()) for n, make in policies.items()}
            plot_portfolio_values(traj, figs / f"phase3_values_{ticker}_{start[:4]}.png",
                                  title=f"{ticker} test {start[:4]} (seed {seed})")

    print("Test windows, all methods (seeds averaged within each ticker and year):")
    print(by_method.round(3).to_string())
    print("\nPaired comparisons over market windows (a - b, 95% bootstrap CI):")
    print(paired.round(3).to_string(index=False))
    print("\nSeed stability (mean test return per seed, spread across seeds):")
    print(per_seed.round(3).to_string())


def main(argv=None):
    args = parse_args(argv)
    {1: run_phase1, 2: run_phase2, 3: run_phase3}[args.phase](args)


if __name__ == "__main__":
    main()
