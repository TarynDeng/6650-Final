"""Write the paper's result tables and figures from results/.

    python paper/make_assets.py

Creates paper/tables/main_results.tex, paper/tables/paired.tex and
paper/figures/learning_curves.png. Re-run after re-running the experiments.
"""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper"
NAMES = {"buy_and_hold": "Buy-and-hold", "qlearning": "Q-learning", "sarsa": "SARSA",
         "dqn": "DQN", "weighted_ens": "Weighted ensemble", "equal_ens": "Equal-weight ensemble"}


def pct(x: float, signed: bool = True) -> str:
    if pd.isna(x):
        return "--"
    return f"${100 * x:+.1f}$" if signed else f"{100 * x:.1f}"  # math mode: a real minus sign


def main_results_table() -> str:
    s = pd.read_csv(RES / "tables" / "phase3_summary_by_method.csv", index_col="method")
    b = pd.read_csv(RES / "tables" / "phase3_trading_behavior.csv", index_col="method")
    rows = []
    for m, name in NAMES.items():
        loss = 1 - s.loc[m, "share_cat1"]
        rows.append(f"{name} & {pct(s.loc[m, 'mean'])} & {pct(s.loc[m, 'median'])} & "
                    f"{pct(s.loc[m, 'seed_std'], False)} & {pct(loss, False)} & "
                    f"{b.loc[m, 'n_trades']:.0f} & {b.loc[m, 'total_fees']:.0f} & "
                    f"{pct(b.loc[m, 'avg_position'], False)} \\\\")
    n = int(s["n_windows"].iloc[0])
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Test years 2023--2025: terminal return in percent, mean and median over the "
        rf"{n} stock-years after averaging seeds; spread across seeds (std, percentage points); "
        r"share of episodes ending with a loss; trades and fees (\$) per year; average share "
        r"of capital held in stock.}",
        r"\label{tab:main}", r"\small",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & Mean & Median & Seed std & Loss share & Trades & Fees & In stock \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def paired_table() -> str:
    p = pd.read_csv(RES / "tables" / "phase3_paired_comparisons.csv")
    rows = [f"{NAMES[r.method_a]} $-$ {NAMES[r.method_b]} & {pct(r.mean_diff)} & "
            f"{r.a_wins}/{r.n_windows} & [{pct(r.ci_low)}, {pct(r.ci_high)}] & "
            f"[{pct(r.year_min)}, {pct(r.year_max)}] \\\\" for r in p.itertuples()]
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired differences in test terminal return (percentage points), stock-year by "
        r"stock-year. Wins: stock-years where A was better. The bootstrap interval treats "
        r"stock-years as independent and is therefore too narrow; the last column gives the range "
        r"of the three yearly mean differences.}",
        r"\label{tab:paired}", r"\small",
        r"\begin{tabular}{lrrcc}", r"\toprule",
        r"A $-$ B & Mean & Wins & 95\% bootstrap & Yearly range \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main() -> None:
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables" / "main_results.tex").write_text(main_results_table())
    (OUT / "tables" / "paired.tex").write_text(paired_table())
    shutil.copy(RES / "figures" / "phase3_learning_curves.png", OUT / "figures" / "learning_curves.png")
    print("Wrote paper/tables/main_results.tex, paper/tables/paired.tex, "
          "paper/figures/learning_curves.png")


if __name__ == "__main__":
    main()
