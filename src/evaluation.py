"""Evaluation metrics, stability analysis, and plots.

See ``docs/proposal.md`` Section 8.

Metrics:
    * Terminal return (reconstructed): ``R_T = (V_T - V_0) / V_0``.
    * Terminal loss (reconstructed): ``L_T = max(0, -R_T)``, classified at
      the end of each test window into
        Category 1: profit or break-even (L_T == 0)
        Category 2: 0% < L_T <= 20%
        Category 3: 20% < L_T <= 40%
        Category 4: L_T > 40%
      Terminal loss is NOT maximum drawdown; do not compute drawdown as a
      substitute. Report zero counts honestly and also show the continuous
      return distribution.
    * Learning efficiency: return vs. environment steps.
    * Stability: mean / std across seeds, stocks, and test years. Results
      from different seeds on the same market window are not independent
      market samples.
    * Trading behavior: buy / sell / hold frequencies, action-size
      distribution, number of trades, total fees, portfolio-value curves.

Baselines evaluated alongside the RL methods: buy-and-hold.
"""

from __future__ import annotations

# Terminal-loss category upper bounds (fraction of initial capital).
LOSS_CATEGORY_BOUNDS = (0.0, 0.20, 0.40)


def terminal_return(portfolio_values):
    """Return ``(V_T - V_0) / V_0`` for one episode's portfolio-value path."""
    raise NotImplementedError


def terminal_loss(portfolio_values):
    """Return ``max(0, -R_T)`` for one episode."""
    raise NotImplementedError


def loss_category(loss):
    """Map a terminal loss to category 1-4 (see module docstring)."""
    raise NotImplementedError


def run_policy(env, policy):
    """Roll out a greedy policy on one test window and record the trajectory.

    TODO: record portfolio values, actions, trades, and fees per step.
    """
    raise NotImplementedError


def buy_and_hold(prices, initial_cash, fee_model):
    """Buy-and-hold benchmark portfolio-value path."""
    raise NotImplementedError


def summarize(results):
    """Aggregate per-run results into mean / median / std tables by method,
    stock, test window, and seed."""
    raise NotImplementedError


def plot_learning_curves(logs, out_path):
    raise NotImplementedError


def plot_portfolio_values(trajectories, out_path):
    raise NotImplementedError
