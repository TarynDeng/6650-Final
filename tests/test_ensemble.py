import numpy as np
import pytest

from src.ensemble import EnsemblePolicy, equal_weights, softmax_weights, weighted_vote


def test_softmax_weights_sum_to_one_and_order():
    w = softmax_weights([0.10, -0.05, 0.02], tau=0.05)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[2] > w[1]
    # Large tau -> nearly equal; equal returns -> exactly equal.
    assert np.allclose(softmax_weights([0.1, -0.1, 0.0], tau=1e6), 1 / 3, atol=1e-6)
    assert np.allclose(softmax_weights([0.2, 0.2, 0.2], tau=0.05), 1 / 3)
    with pytest.raises(ValueError):
        softmax_weights([0.1], tau=0)


def test_weighted_vote_majority_and_weights():
    assert weighted_vote([2, 2, 5], equal_weights(3)) == 2
    # One heavy policy beats two light ones that agree with each other.
    assert weighted_vote([4, 8, 8], [0.6, 0.2, 0.2]) == 4
    # Two agreeing policies outweigh the top one.
    assert weighted_vote([4, 8, 8], [0.4, 0.3, 0.3]) == 8


def test_tie_break_rules():
    # Two-way score tie: the single top-weight policy wins if its action is tied.
    assert weighted_vote([3, 6, 6], [0.5, 0.25, 0.25]) == 3
    # Equal weights, all different, Hold among them -> Hold.
    assert weighted_vote([0, 4, 8], equal_weights(3)) == 0
    # Equal weights, all different, no Hold -> smallest index.
    assert weighted_vote([7, 2, 5], equal_weights(3)) == 2


class Fixed:
    def __init__(self, a):
        self.a = a

    def act(self, obs, explore=False):
        return self.a


def test_ensemble_policy_uses_member_actions():
    ens = EnsemblePolicy([Fixed(1), Fixed(1), Fixed(8)], equal_weights(3))
    assert ens.act(np.zeros(4)) == 1
    assert ens.member_actions(np.zeros(4)) == [1, 1, 8]
