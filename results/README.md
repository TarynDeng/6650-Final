# Results

Experiment outputs are **not committed** to git (see `.gitignore`). Only this file is tracked. Final figures for the paper are copied into `paper/figures/`, which is tracked.

## Layout

```
results/
├── README.md        # this file (tracked)
├── models/          # trained Q-tables and DQN weights
├── logs/            # per-run learning curves (return vs. environment steps)
├── tables/          # aggregated metrics (terminal return, loss categories, stability)
└── figures/         # plots for the paper and slides
```

## Naming

Each training run is identified by algorithm, ticker, and seed:

```
{algo}_{TICKER}_seed{n}        e.g. dqn_XXX_seed0
```

- `models/{run}.pkl` (tabular Q-table) or `models/{run}.pt` (DQN weights)
- `logs/{run}.csv` with at least: `env_step, episode, train_return, eval_return, wall_time`
- `tables/{phase}_{split}.csv`, e.g. `phase3_test.csv`, one row per (method, ticker, test window, seed)
- Ensembles use `weighted_ens` and `equal_ens` as the algorithm name.

Every run should also save its config (hyperparameters, seed, data split, git commit hash) so results can be traced back to code.
