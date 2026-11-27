"""Performance-based weighted ensemble and equal-weight ensemble.

See ``docs/proposal.md`` Section 6, Phase 2, and ``EnsembleConfig`` in
``src/config.py``.

Weights (formula reconstructed from the outline)::

    w_i = exp(R_i / T) / sum_j exp(R_j / T)

where ``R_i`` is member i's terminal return on the validation year and ``T``
is the temperature. Voting at each step::

    S_t(a) = sum_i w_i * 1[pi_i(s_t) == a]
    a_t    = argmax_a S_t(a)

Rules the implementation respects:
    * Weights come from the validation year only, never the test years.
    * At every step, each member recomputes its greedy action from the
      ensemble's own observation (market features and the ensemble's own
      position). Actions from each member's separate backtest are never
      spliced together.
    * Ties use a rule fixed before looking at results ("top_weight_then_hold"):
      take the action of the single highest-weight member if it is tied;
      otherwise take Hold (0) if it is tied, else the smallest action index.
      Two alternative rules ("hold", "largest_action") exist only for a
      sensitivity check.
    * The equal-weight ensemble (w_i = 1/N) is the control.

Note: a member whose weight exceeds 0.5 always wins the vote, so a weighted
ensemble with a dominant member behaves exactly like that member.
"""

from __future__ import annotations

import numpy as np

from .environment import N_ACTIONS


def softmax_weights(validation_returns, temperature: float) -> np.ndarray:
    r = np.asarray(validation_returns, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (r - r.max()) / temperature  # subtract the max for numerical stability
    w = np.exp(z)
    return w / w.sum()


def equal_weights(n_policies: int) -> np.ndarray:
    return np.full(n_policies, 1.0 / n_policies)


TIE_BREAKS = ("top_weight_then_hold", "hold", "largest_action")


def weighted_vote(actions, weights, n_actions: int = N_ACTIONS,
                  tie_break: str = "top_weight_then_hold") -> int:
    """Return argmax_a S(a) for the members' ``actions`` and ``weights``."""
    if tie_break not in TIE_BREAKS:
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
    if tie_break == "hold":
        return 0
    if tie_break == "largest_action":
        return int(tied.max())
    return 0 if 0 in tied else int(tied.min())


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
