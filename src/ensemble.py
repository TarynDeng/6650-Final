"""Performance-based weighted ensemble and equal-weight ensemble.

See ``docs/proposal.md`` Section 6, Phase 2.

Weights (reconstructed formula, team to confirm)::

    w_i = exp(R_bar_i / tau) / sum_j exp(R_bar_j / tau)

where ``R_bar_i`` is policy i's mean terminal return on the validation set
and ``tau`` is a temperature fixed in advance or chosen on validation only.

Voting at each step::

    S_t(a) = sum_i w_i * 1[pi_i(s_t) == a]
    a_t    = argmax_a S_t(a)

Rules the implementation must respect:
    * Weights come from the validation set only. Never use the test set.
    * At every step, each member policy recomputes its greedy action from the
      ensemble's own current account state (cash / position). Do not splice
      together actions from each policy's independent backtest.
    * Ties are broken by a fixed, reproducible rule decided in advance.
    * The equal-weight ensemble (w_i = 1/N) is the control.

Still open: the value of ``tau`` and the tie-breaking rule.
"""

from __future__ import annotations


def softmax_weights(validation_returns, tau):
    """Return softmax weights from mean validation terminal returns.

    TODO: implement in a numerically stable way (subtract the max).
    """
    raise NotImplementedError


def equal_weights(n_policies):
    """Return uniform weights 1/N."""
    raise NotImplementedError


def weighted_vote(actions, weights, n_actions, tie_break):
    """Combine member actions into one action by weighted voting.

    TODO: score each action as the sum of the weights of policies that chose
    it, take the argmax, and resolve ties with ``tie_break`` (rule TBD).
    """
    raise NotImplementedError


class EnsemblePolicy:
    """Policy that queries every member on the current state and votes.

    TODO: ``act(observation)`` asks each member for its greedy action on the
    same observation (built from the ensemble's own portfolio), then calls
    ``weighted_vote``.
    """

    def __init__(self, policies, weights, tie_break):
        raise NotImplementedError

    def act(self, observation):
        raise NotImplementedError
