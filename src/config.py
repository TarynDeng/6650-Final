"""Default experiment settings, in one place.

Every value here is a placeholder default chosen to get the pipeline running;
the team should confirm or change them (see "Decisions and their status" in the
README). Code reads settings from these dataclasses instead of hard-coding
constants, so changing a decision means editing this file and re-running all
three phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    # Three large, liquid stocks from different sectors (tech, banking, energy).
    tickers: tuple[str, ...] = ("AAPL", "JPM", "XOM")
    # Download starts before 2015 so that 20-day features exist on 2015-01-02.
    download_start: str = "2014-10-01"
    download_end: str = "2026-01-01"  # exclusive
    train: tuple[str, str] = ("2015-01-01", "2021-12-31")
    val: tuple[str, str] = ("2022-01-01", "2022-12-31")
    test_windows: tuple[tuple[str, str], ...] = (
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
    )
    # Prices are split- and dividend-adjusted (yfinance auto_adjust=True), so
    # stock splits do not show up as fake crashes.
    adjusted: bool = True


@dataclass(frozen=True)
class EnvConfig:
    initial_cash: float = 10_000.0
    # Proportional commission on the traded value, charged on buys and sells.
    fee_rate: float = 0.001  # 10 basis points
    # Fractional shares keep the 25/50/75/100% actions exact. With integer
    # shares, small trades would often round to zero.
    fractional_shares: bool = True
    # Training episodes are random 252-trading-day (about 12-month) windows
    # inside the training period, matching the 12-month test windows.
    train_episode_len: int = 252
    # Timing rule: the agent observes features computed from closes up to and
    # including day t, the order fills at day t's close, and the portfolio is
    # marked to market at day t+1's close. No future data is used.


@dataclass(frozen=True)
class TabularConfig:
    # Quantile bins per market feature, fit on the training split only.
    n_bins: tuple[int, int, int] = (3, 3, 3)  # ret5, ret20, vol20
    # Position fraction is bucketed into 5 levels: ~0, 25, 50, 75, ~100%.
    n_position_levels: int = 5  # -> 3 * 3 * 3 * 5 = 135 states
    alpha: float = 0.05
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.5  # fraction of training steps used to decay epsilon


@dataclass(frozen=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 5e-4
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 1_000
    target_update_rate: float = 0.005  # soft target-network update (often called tau), as in HW-2
    grad_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    gamma: float = 0.99  # shared by all algorithms
    total_env_steps: int = 100_000  # same step budget for every algorithm
    eval_every: int = 5_000  # validation check interval, in env steps (monitoring only)
    # False: every algorithm is evaluated with its final model after the full
    # step budget. The validation year is then used for one thing only, the
    # ensemble weights. (True would keep the snapshot with the best validation
    # return, but then the same year both selects checkpoints and sets the
    # weights, which inflates the noisiest learner's validation score.)
    keep_best_on_val: bool = False
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class EnsembleConfig:
    # Softmax temperature on each member's 2022 validation terminal return.
    # With returns that differ by 0.1-0.3, tau = 0.05 puts more than half the
    # weight on one member, so the weighted vote then follows that member.
    temperature: float = 0.05
    # Tie-break, fixed before any results were seen: among tied actions, take
    # the one chosen by the single highest-weight member; otherwise (e.g. equal
    # weights) take Hold (0) if it is tied, else the smallest action index.
    tie_break: str = "top_weight_then_hold"
    # Alternative rules evaluated in phase 2 as a sensitivity check only.
    tie_break_sensitivity: tuple[str, ...] = ("top_weight_then_hold", "hold", "largest_action")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
