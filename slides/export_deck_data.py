"""Collect every number the slide deck shows into slides/deck_data.json.

Needs the committed results/ tables and logs (or a fresh run of the three
experiment phases):
    python slides/export_deck_data.py
    cd slides && npm install && npm run build
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ExperimentConfig  # noqa: E402

RES = ROOT / "results"
T = RES / "tables"
METHODS = ["buy_and_hold", "qlearning", "sarsa", "dqn", "weighted_ens", "equal_ens"]
AGENTS = ["qlearning", "sarsa", "dqn"]
LABELS = {"buy_and_hold": "Buy-and-hold", "qlearning": "Q-learning", "sarsa": "SARSA",
          "dqn": "DQN", "weighted_ens": "Weighted ens.", "equal_ens": "Equal-weight ens."}


def pct(x) -> float:
    return round(100 * float(x), 1)


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def summary_fields(summary, behavior, test) -> dict:
    bh = test.query("method == 'buy_and_hold'")
    return {
        "methods": [LABELS[m] for m in METHODS],
        "mean_return": [pct(summary.loc[m, "mean"]) for m in METHODS],
        "seed_std": [None if pd.isna(summary.loc[m, "seed_std"]) else pct(summary.loc[m, "seed_std"])
                     for m in METHODS],
        "loss_share": {f"cat{c}": [pct(summary.loc[m, f"share_cat{c}"]) for m in METHODS]
                       for c in (1, 2, 3, 4)},
        "large_losses": int((test["loss_category"] >= 3).sum()),
        "worst_loss": pct(test["terminal_loss"].max()),
        "falling_stock_years": [f"{r.ticker} {r.window}" for r in bh.itertuples() if r.terminal_return < 0],
        "trades": [round(behavior.loc[m, "n_trades"]) for m in METHODS],
        "fees": [round(behavior.loc[m, "total_fees"]) for m in METHODS],
        "avg_position": [pct(behavior.loc[m, "avg_position"]) for m in METHODS],
    }


def ensemble_fields(weights, val1, summary, sens) -> dict:
    w = weights.query("method == 'weighted_ens'")
    picks = w["top_member"].value_counts()
    return {
        "picks": {LABELS[a]: int(picks.get(a, 0)) for a in AGENTS},
        "n_runs": len(w),
        "follows_top": int(w["follows_top_member"].sum()),
        "min_top_weight": round(float(w["top_weight"].min()), 2),
        "val_by_ticker": {t: {LABELS[a]: pct(val1.query("method == @a and ticker == @t")
                                             ["terminal_return"].mean())
                              for a in ["buy_and_hold", *AGENTS]}
                          for t in sorted(val1["ticker"].unique())},
        "test_by_agent": {LABELS[a]: pct(summary.loc[a, "mean"]) for a in AGENTS},
        "tiebreak": {r: pct(sens.loc[r, "mean"]) for r in sens.index},
    }


def curves(manifest) -> dict:
    out = {}
    for t in manifest["tickers"]:
        logs = pd.concat([pd.read_csv(RES / "logs" / f"{a}_{t}_seed{s}.csv")
                          for a in AGENTS for s in manifest["seeds"]])
        g = logs.dropna(subset=["eval_return"]).groupby(["algo", "env_step"])["eval_return"].mean()
        g = g.unstack("algo")
        out[t] = {"steps": [int(x) for x in g.index], **{LABELS[a]: [pct(v) for v in g[a]] for a in AGENTS}}
    return out


def main() -> None:
    manifest = json.loads((RES / "run_manifest.json").read_text())
    summary = pd.read_csv(T / "phase3_summary_by_method.csv", index_col="method").reindex(METHODS)
    behavior = pd.read_csv(T / "phase3_trading_behavior.csv", index_col="method").reindex(METHODS)
    test = pd.read_csv(T / "phase3_test.csv")
    paired = pd.read_csv(T / "phase3_paired_comparisons.csv")
    cfg = ExperimentConfig()
    data = {
        "commit": manifest["git_commit"].replace("-dirty", ""), "tickers": manifest["tickers"],
        "seeds": manifest["seeds"], "steps": manifest["total_env_steps"],
        "n_stock_years": int(summary.loc["qlearning", "n_windows"]), "n_tests": count_tests(),
        "config": {"fee_pct": 100 * cfg.env.fee_rate, "cash": cfg.env.initial_cash,
                   "episode_len": cfg.env.train_episode_len, "gamma": cfg.train.gamma,
                   "temperature": cfg.ensemble.temperature, "hidden": list(cfg.dqn.hidden_sizes),
                   "n_states": int(pd.Series(cfg.tabular.n_bins).prod() * cfg.tabular.n_position_levels)},
        **summary_fields(summary, behavior, test),
        **ensemble_fields(pd.read_csv(T / "phase2_weights.csv"), pd.read_csv(T / "phase1_val.csv"),
                          summary, pd.read_csv(T / "phase2_tiebreak_sensitivity.csv", index_col="tie_break")),
        "paired": [{"a": LABELS[r.method_a], "b": LABELS[r.method_b], "diff": pct(r.mean_diff),
                    "wins": int(r.a_wins), "n": int(r.n_windows)} for r in paired.itertuples()],
        "curves": curves(manifest),
    }
    (ROOT / "slides" / "deck_data.json").write_text(json.dumps(data, indent=1))
    print(f"Wrote slides/deck_data.json ({data['n_tests']} tests, commit {data['commit']})")


if __name__ == "__main__":
    main()
