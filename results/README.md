# Results

Experiment outputs are **not committed** to git (see `.gitignore`); re-create them with the commands in the main README. Final figures for the paper are copied into `paper/figures/`, which is tracked.

```
results/
├── run_manifest.json   # stocks, seeds, settings, git commit, package versions, data checksums
├── models/             # trained Q-tables (.pkl) and DQN weights (.pt)
├── logs/               # one learning curve per run
├── tables/             # per-episode results and summaries
└── figures/            # plots for the paper and slides
```

Phase 1 deletes the previous contents of `results/` before it starts, so results from different runs are never mixed. To keep an older run, use `--results-dir` with another folder.

## Naming

Each training run is `{algo}_{TICKER}_seed{n}`, for example `dqn_AAPL_seed0`. The ensembles use the method names `weighted_ens` and `equal_ens`; buy-and-hold is `buy_and_hold` with seed `-1`.

- `logs/{run}.csv`: `env_step, episode, train_return, eval_return, epsilon, wall_time` (`eval_return` is the 2022 validation return, recorded every 5,000 steps).
- `tables/phase{1,2}_{val,test}.csv`: one row per method, ticker, seed and year, with terminal return, loss category, trade count, fees and action frequencies.
- `tables/phase3_*.csv`: final summaries; see "Output files" in the main README.
