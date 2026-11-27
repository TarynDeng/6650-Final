"""Performance-based weighted ensemble and equal-weight ensemble.

See ``docs/proposal.md`` Section 6, Phase 2. Default settings live in
``src/config.py`` (``EnsembleConfig``).

Weights (reconstructed formula, team to confirm)::

    w_i = exp(R_bar_i / tau) / sum_j exp(R_bar_j / tau)

where ``R_bar_i`` is policy i's mean terminal return on the validation set
and ``tau`` is a temperature fixed in advance.

Voting at each step::

    S_t(a) = sum_i w_i * 1[pi_i(s_t) == a]
    a_t    = argmax_a S_t(a)

Rules the implementation respects:
    * Weights come from the validation set only, never the test set.
    * At every step, each member policy recomputes its greedy action from the
      ensemble's own current observation (market features and the ensemble's
      own position). Actions from each member's separate backtest are never
      spliced together.
    * Ties use a fixed rule decided before looking at results
      ("top_weight_then_hold"): among the tied actions, take the action of the
      single highest-weight policy if it is one of them; otherwise (for
      example, equal weights) take Hold (0) if it is tied, else the smallest
      tied action index.
    * The equal-weight ensemble (w_i = 1/N) is the control.

With three members, the weighted vote follows the top-weighted policy unless
the other two agree and together outweigh it.
"""

from __future__ import annotations

import numpy as np

from .environment import N_ACTIONS


def softmax_weights(validation_returns, tau: float) -> np.ndarray:
    r = np.asarray(validation_returns, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")
    z = (r - r.max()) / tau  # subtract the max for numerical stability
    w = np.exp(z)
    return w / w.sum()


def equal_weights(n_policies: int) -> np.ndarray:
    return np.full(n_policies, 1.0 / n_policies)


def weighted_vote(actions, weights, n_actions: int = N_ACTIONS,
                  tie_break: str = "top_weight_then_hold") -> int:
    """Return argmax_a S(a) for the members' ``actions`` and ``weights``."""
    if tie_break != "top_weight_then_hold":
        raise ValueError(f"Unknown tie_break rule: {tie_break}")
    actions = np.asarray(actions, dtype=int)
    weights = np.asarray(weights, dtype=float)
    scores = np.bincount(actions, weights=weights, minlength=n_actions)
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    if len(tied) == 1:
        return int(tied[0])
    top = np.flatnonzero(np.isclose(weights, weights.max()))
    if len(top) == 1 and actions[top[0]] in tied:
        return int(actions[top[0]])
    if 0 in tied:
        return 0
    return int(tied.min())


class EnsemblePolicy:
    """Combine member agents by weighted voting on their greedy actions."""

    def __init__(self, members, weights, tie_break: str = "top_weight_then_hold"):
        if len(members) != len(weights):
            raise ValueError("one weight per member")
        self.members = list(members)
        self.weights = np.asarray(weights, dtype=float)
        self.tie_break = tie_break

    def member_actions(self, obs) -> list[int]:
        return [m.act(obs, explore=False) for m in self.members]

    def act(self, obs, explore: bool = False) -> int:
        return weighted_vote(self.member_actions(obs), self.weights, N_ACTIONS, self.tie_break)
