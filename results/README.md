# Results

The **tables, learning-curve logs and run manifest** of the reported run are committed, so every number in the README, slides and paper can be checked from a fresh clone. Trained models and figures are not committed (re-create them with the commands in the main README).

```
results/
├── run_manifest.json   # stocks, seeds, settings, git commit, code and data checksums, package versions (committed)
├── tables/             # per-episode results and summaries (committed)
├── logs/               # one learning curve per run (committed)
├── models/             # trained Q-tables (.pkl) and DQN weights (.pt)
└── figures/            # plots for the paper and slides
```

Phase 1 first checks the price data, then deletes the previous run's `models/`, `logs/`, `tables/`, `figures/` and `run_manifest.json`, so results from different runs are never mixed. It refuses to clear a folder that contains other files. To keep a run, use `--results-dir` with another folder.

## Naming

- A **run** is `{algo}_{TICKER}_seed{n}`, for example `dqn_AAPL_seed0`. Ensembles use the method names `weighted_ens` and `equal_ens`; buy-and-hold is `buy_and_hold` with seed `-1`.
- A **stock-year** is one stock in one test year; the column `window` holds the year.

## Files

- `logs/{run}.csv`: `env_step, episode, train_return, eval_return, epsilon, complete, wall_time, run, algo, ticker, seed`. `eval_return` is the 2022 validation return, recorded every 5,000 steps; `complete` is False for an episode the step budget cut short.
- `tables/phase{1,2}_{val,test}.csv`: one row per method, stock, seed and year: terminal return and loss, loss category, trades, fees, `avg_position` (average share of capital in stock), final value, and the share of steps on each action (`freq_a0` … `freq_a8`).
- `tables/phase2_weights.csv`: each ensemble's weights, the validation returns they came from, the top-weighted member and whether it holds more than half the weight.
- `tables/phase2_tiebreak_sensitivity.csv`: the equal-weight ensemble's test return under alternative tie-break rules (a robustness check; the reported rule was fixed in advance).
- `tables/phase3_*.csv`: all test rows (`phase3_test.csv`) and summaries by method, stock and year, seed stability, paired comparisons, and trading behavior. The main README, Section 7, explains every column.
