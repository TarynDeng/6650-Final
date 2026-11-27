"""Collect every number the slide deck shows into slides/deck_data.json.

Run after the three experiment phases:
    python slides/export_deck_data.py
    node slides/build_deck.js
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.environment import load_ticker  # noqa: E402

RES = ROOT / "results"
T = RES / "tables"
METHODS = ["buy_and_hold", "qlearning", "sarsa", "dqn", "weighted_ens", "equal_ens"]
LABELS = {"buy_and_hold": "Buy-and-hold", "qlearning": "Q-learning", "sarsa": "SARSA",
          "dqn": "DQN", "weighted_ens": "Weighted ens.", "equal_ens": "Equal-weight ens."}

manifest = json.loads((RES / "run_manifest.json").read_text())
summary = pd.read_csv(T / "phase3_summary_by_method.csv").set_index("method").reindex(METHODS)
by_ticker = pd.read_csv(T / "phase3_summary_by_ticker.csv").set_index(["method", "ticker"])
paired = pd.read_csv(T / "phase3_paired_comparisons.csv")
behavior = pd.read_csv(T / "phase3_trading_behavior.csv").set_index("method").reindex(METHODS)
weights = pd.read_csv(T / "phase2_weights.csv").query("method == 'weighted_ens'")
val1 = pd.read_csv(T / "phase1_val.csv")

tickers = manifest["tickers"]
data = {
    "commit": manifest["git_commit"], "tickers": tickers, "seeds": manifest["seeds"],
    "steps": manifest["total_env_steps"],
    "methods": [LABELS[m] for m in METHODS],
    "mean_return": [round(100 * summary.loc[m, "mean"], 1) for m in METHODS],
    "seed_std": [None if pd.isna(summary.loc[m, "seed_std"]) else round(100 * summary.loc[m, "seed_std"], 1)
                 for m in METHODS],
    "loss_share": {f"cat{c}": [round(100 * summary.loc[m, f"share_cat{c}"], 1) for m in METHODS]
                   for c in (1, 2, 3, 4)},
    "by_ticker": {t: [round(100 * by_ticker.loc[(m, t), "mean"], 1) for m in METHODS] for t in tickers},
    "trades": [round(behavior.loc[m, "n_trades"]) for m in METHODS],
    "fees": [round(behavior.loc[m, "total_fees"]) for m in METHODS],
    "hold_share": [round(100 * behavior.loc[m, "freq_a0"]) for m in METHODS],
    "paired": [{"a": LABELS[r.method_a], "b": LABELS[r.method_b], "diff": round(100 * r.mean_diff, 1),
                "wins": int(r.a_wins), "n": int(r.n_windows),
                "lo": round(100 * r.ci_low, 1), "hi": round(100 * r.ci_high, 1)}
               for r in paired.itertuples()],
    "avg_weights": {a: round(float(weights[f"w_{a}"].mean()), 2) for a in ("qlearning", "sarsa", "dqn")},
    "val_return": {LABELS[m]: round(100 * val1.query("method == @m")["terminal_return"].mean(), 1)
                   for m in ("buy_and_hold", "qlearning", "sarsa", "dqn")},
}

# Learning curves: mean validation return over seeds, per algorithm, for each ticker.
curves = {}
for t in tickers:
    logs = pd.concat([pd.read_csv(RES / "logs" / f"{a}_{t}_seed{s}.csv")
                      for a in ("qlearning", "sarsa", "dqn") for s in manifest["seeds"]])
    ev = logs.dropna(subset=["eval_return"])
    g = ev.groupby(["algo", "env_step"])["eval_return"].mean().unstack("algo")
    curves[t] = {"steps": [int(x) for x in g.index],
                 **{LABELS[a]: [round(100 * v, 1) for v in g[a]] for a in ("qlearning", "sarsa", "dqn")}}
data["curves"] = curves

# Example year: buy-and-hold price path of each ticker in each test year (for context).
yearly = {}
for t in tickers:
    splits = load_ticker(t)
    yearly[t] = [round(100 * (d["Close"].iloc[-1] / d["Close"].iloc[0] - 1), 1) for d in splits["test"]]
data["stock_year_returns"] = yearly

(ROOT / "slides" / "deck_data.json").write_text(json.dumps(data, indent=1))
print(json.dumps({k: data[k] for k in ("mean_return", "seed_std", "avg_weights", "val_return",
                                       "stock_year_returns", "trades", "fees", "hold_share")}, indent=1))
