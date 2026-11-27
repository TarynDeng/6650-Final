"""Ensemble weights, the vote, tie-break rules, and acting on the ensemble's own account."""

import numpy as np
import pytest

from src.ensemble import EnsemblePolicy, equal_weights, softmax_weights, weighted_vote
from src.environment import TradingEnv
from src.evaluation import run_policy
from tests.helpers import make_features


def test_softmax_weights():
    w = softmax_weights([0.10, -0.05, 0.02], temperature=0.05)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[2] > w[1]
    assert np.allclose(softmax_weights([0.1, -0.1, 0.0], temperature=1e6), 1 / 3, atol=1e-6)
    assert np.allclose(softmax_weights([0.2, 0.2, 0.2], temperature=0.05), 1 / 3)
    with pytest.raises(ValueError):
        softmax_weights([0.1], temperature=0)


def test_vote_majority_and_weights():
    assert weighted_vote([2, 2, 5], equal_weights(3)) == 2
    assert weighted_vote([4, 8, 8], [0.6, 0.2, 0.2]) == 4  # one member above 0.5 always wins
    assert weighted_vote([4, 8, 8], [0.4, 0.3, 0.3]) == 8  # two agreeing members outweigh it


def test_tie_break_rules():
    assert weighted_vote([3, 6, 6], [0.5, 0.25, 0.25]) == 3  # single top-weight member
    assert weighted_vote([0, 4, 8], equal_weights(3)) == 0  # Hold is tied
    assert weighted_vote([7, 2, 5], equal_weights(3)) == 2  # smallest index
    assert weighted_vote([7, 2, 5], equal_weights(3), tie_break="hold") == 0
    assert weighted_vote([7, 2, 5], equal_weights(3), tie_break="largest_action") == 7
    with pytest.raises(ValueError):
        weighted_vote([1, 2, 3], equal_weights(3), tie_break="coin_flip")


class PositionAgent:
    """Buys all when flat, sells all when fully invested; refuses to explore."""

    def act(self, obs, explore=True):
        assert explore is False, "ensemble members must act greedily"
        return 4 if obs[-1] < 0.5 else 8


def test_members_act_on_the_ensembles_own_position():
    # Every member reads the ensemble's position: it alternates buy-all / sell-all.
    ens = EnsemblePolicy([PositionAgent(), PositionAgent(), PositionAgent()], equal_weights(3))
    res = run_policy(TradingEnv(make_features(n=30)), ens.act)
    assert list(res["actions"][:4]) == [4, 8, 4, 8]
