# 6650-Final

**Comparative Study of Shallow and Deep Reinforcement Learning Algorithms in Stock Trading with Performance-Based Weighted Ensemble**

DSAN 6650 (Reinforcement Learning), Final Project. Presentation due **12/2**.

> **Status:** the full pipeline works end to end: data loading, the trading environment, Q-learning, SARSA, DQN, the weighted and equal-weight ensembles, buy-and-hold, evaluation tables and figures. All three phases run with one command each. Every setting the team has not decided yet has a default in [`src/config.py`](src/config.py); see [Open Decisions](#open-decisions). The full proposal, including all equations, is in [`docs/proposal.md`](docs/proposal.md).

## Overview

We use a historical stock-trading simulation as a common environment to compare **shallow (tabular) RL** and **deep RL** algorithms. The focus is the RL algorithms, not a sophisticated trading agent. Each algorithm learns on its own **when to buy, when to sell, how much to trade, and when to hold**. After training, we compare the algorithms on held-out historical data, and test whether a **validation-weighted vote over the trained policies** beats the individual algorithms.

All experiments use historical data and simulated capital. No real trading is involved.

## Research Questions

- **RQ1, Algorithm Performance:** How does shallow RL differ from deep RL in trading performance in the same market environment?
- **RQ2, Terminal Loss Distribution:** How do the algorithms differ in the frequency and severity of terminal investment losses?
- **RQ3, Learning Efficiency and Stability:** How do they differ in learning efficiency, training stability, and results across random seeds?
- **RQ4, Weighted Ensemble:** Does a weighted combination built from validation performance improve return, terminal-loss outcomes, and stability?

## Methods at a Glance

| Category | Algorithm | Notes |
|----------|-----------|-------|
| Shallow RL | Q-learning | Tabular, off-policy |
| Shallow RL | SARSA | Tabular, on-policy |
| Deep RL | DQN | Neural network, off-policy; experience replay + target network |
| Ensemble | Performance-based weighted vote | Softmax weights from validation terminal return |
| Ensemble baseline | Equal-weight vote | Control for the weighted ensemble |
| Benchmark | Buy-and-hold | Non-RL baseline |

**Environment.** One stock per episode, daily bars, no leverage, no short selling, transaction costs included. We prefer an existing Gymnasium-compatible trading environment (e.g., Gym Trading Env) plus a thin action adapter if needed.

**State.** Market features (e.g., 5-day return, 20-day return, 20-day volatility) and portfolio features (cash fraction, position fraction). Discretized for the tabular methods, continuous for DQN. Discretization and normalization are fit on the training set only.

**Actions (tentative): 9-level discrete trade sizing.**

| Action | Operation | Action | Operation |
|:-:|---|:-:|---|
| 0 | Hold | | |
| 1 | Buy with 25% of available cash | 5 | Sell 25% of holdings |
| 2 | Buy with 50% of available cash | 6 | Sell 50% of holdings |
| 3 | Buy with 75% of available cash | 7 | Sell 75% of holdings |
| 4 | Buy with 100% of available cash | 8 | Sell 100% of holdings |

**Reward.** Per-step log return of portfolio value, $r_t = \ln(V_t / V_{t-1})$, with fees reflected in $V_t$. No drawdown penalty, risk-aversion term, or stop-loss rule.

## Data Split (tentative)

| Split | Period | Purpose |
|-------|--------|---------|
| Training | 2015–2021 | Train the RL algorithms |
| Validation | 2022 | Limited hyperparameter selection; ensemble weights |
| Testing | 2023–2025 | Out-of-sample comparison, as three 12-month windows |

About 3 stocks and about 5 training seeds per algorithm (both tentative). The test set is never used for training, tuning, or ensemble weights.

## Evaluation

- **Terminal return** $R_T = (V_T - V_0)/V_0$: mean, median, per window.
- **Terminal loss** categories at the end of each test window: (1) profit or break-even, (2) loss in (0%, 20%], (3) loss in (20%, 40%], (4) loss > 40%. **This is not maximum drawdown.**
- **Learning efficiency:** return vs. environment steps.
- **Stability:** mean and standard deviation across seeds, stocks, and test years.
- **Trading behavior:** buy/sell/hold frequencies, action-size distribution, trade count and fees, portfolio-value curves.

## Repository Layout

```
6650-Final/
├── data/
│   └── README.md               # data layout and sources (data files are git-ignored)
├── docs/
│   ├── proposal.md             # full project proposal
│   └── presentation.md         # slide-by-slide outline for 12/2
├── experiments/
│   └── run_experiments.py      # entry point for training / evaluation runs
├── paper/
│   ├── main.tex                # research paper skeleton (LaTeX, 4–6 pages)
│   ├── references.bib
│   └── figures/                # final figures used in the paper
├── results/
│   └── README.md               # output layout and naming (outputs are git-ignored)
├── src/
│   ├── __init__.py
│   ├── config.py               # all default settings (team to confirm)
│   ├── environment.py          # data loading, features, trading env
│   ├── tabular_rl.py           # state discretization, Q-learning, SARSA
│   ├── deep_rl.py              # DQN (replay buffer, target network)
│   ├── ensemble.py             # softmax weights, weighted / equal-weight voting
│   ├── evaluation.py           # terminal return / loss, rollouts, buy-and-hold, plots
│   └── training.py             # one training loop shared by all algorithms
├── tests/                      # accounting and agent tests (pytest)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests, then Phase 1. The first run downloads daily prices from yfinance into `data/raw/` (git-ignored); later runs read the cached CSVs.

```bash
python -m pytest -q tests
python experiments/run_experiments.py --phase 1 --tickers AAPL --seeds 0   # one stock, one seed (~1 min)

# Full experiment, all tickers and seeds in src/config.py (~20 min on a laptop CPU)
python experiments/run_experiments.py --phase 1   # train Q-learning, SARSA, DQN; evaluate them and buy-and-hold
python experiments/run_experiments.py --phase 2   # weighted and equal-weight ensembles (weights from 2022 only)
python experiments/run_experiments.py --phase 3   # final tables and figures for all six methods
```

Phases 2 and 3 read the Phase 1 models and tables, so pass the same `--tickers` / `--seeds` to every phase.

Outputs go to `results/` (see [`results/README.md`](results/README.md)):
- `tables/phase{1,2}_{val,test}.csv`: one row per method, ticker, seed, and window; `phase2_weights.csv`: ensemble weights;
- `tables/phase3_*.csv`: final summaries by method, ticker, test year, seed stability, and trading behavior;
- `logs/{algo}_{TICKER}_seed{n}.csv`: learning curves;
- `models/`: trained Q-tables and DQN weights;
- `figures/phase3_*.png`: terminal return, loss categories, action frequencies, learning curves, and portfolio-value curves.

Build the paper (needs a TeX distribution with `latexmk`):

```bash
cd paper && latexmk -pdf main.tex
```

## Implementation Plan

1. Minimal working experiment: verify data, environment, action execution, accounting, and reward; run all three algorithms on one stock.
2. Train and compare the individual algorithms; save models and learning curves; finish validation.
3. Weighted ensemble from validation results only, plus the equal-weight baseline.
4. Scale up: more stocks, fixed yearly test windows, multiple seeds.
5. Analysis and reporting: metrics, figures, paper, presentation.
6. Optional: DQN on the discrete state encoding as a control experiment.

## Open Decisions

Every item below has a default in [`src/config.py`](src/config.py) so the code runs today. The defaults are placeholders: confirm or change them, and record the decision here.

| Decision | Current default |
|---|---|
| Data source, tickers, dates | yfinance, adjusted prices; AAPL, JPM, XOM; 2015-2025 |
| Environment | Small custom Gymnasium env (Gym Trading Env uses target-position actions, not cash-fraction orders) |
| Execution and costs | Fill at day t's close, mark at day t+1's close; 0.1% fee per trade; fractional shares |
| Training episodes | Random 252-day windows inside 2015-2021 |
| Tabular state | ret5, ret20, vol20 in 3 quantile bins each (fit on train) x 5 position levels = 135 states |
| DQN | HW-2-style PyTorch DQN: MLP 64-64, replay, soft target update tau = 0.005 |
| Budget and selection | 100k env steps per run, gamma = 0.99, 5 seeds; the snapshot with the best validation return is kept for every algorithm |
| Ensemble | tau = 0.05; ties go to the top-weighted policy's action, then Hold |

- [ ] Data source, final stock list, and date range
- [ ] Trading environment, and how it supports discrete trade sizing
- [ ] Keep the 9 actions? Integer vs. fractional shares, commission, execution-price rule
- [ ] State features, discretization levels, tabular state-space size
- [ ] DQN from the HW-2 PyTorch implementation or from RLlib
- [ ] Training budget, number of seeds, hyperparameter ranges
- [ ] Ensemble temperature $\tau$ and vote tie-breaking rule
- [ ] Discount factor $\gamma$ (shared across all algorithms) *(added during repo setup)*
- [ ] Team roles and GitHub workflow
- [ ] Paper template required by the course (the skeleton uses the plain `article` class) and slide software *(added during repo setup)*
- [ ] Repository visibility: the deliverables call for a private repository; this one is currently public *(added during repo setup)*
- [ ] Confirm the equations marked *reconstructed* in [`docs/proposal.md`](docs/proposal.md) *(added during repo setup)*

## Deliverables

1. Original, reproducible code (training, testing, ensemble, evaluation)
2. Research paper, 4–6 pages
3. Presentation, 10–15 minutes (12/2)
4. Private GitHub repository with commit history
