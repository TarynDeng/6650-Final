"""Tabular RL: state discretization, Q-learning, and SARSA.

See ``docs/proposal.md`` Sections 4 and 5.3. Builds on HW-1.

Rules the implementation must respect:
    * Discretization boundaries (bins) are fit on the training split only,
      then reused unchanged for validation and test.
    * Use epsilon-greedy exploration during training and greedy actions for
      evaluation.
    * Q-learning is off-policy (max over next actions); SARSA is on-policy
      (uses the next action actually taken).
    * The discount factor gamma must be the same for every algorithm.
    * Record the random seed and the number of environment steps for every
      run, so learning curves are comparable with DQN.

Still open: which features to discretize, the number of bins per feature,
and the resulting size of the state space.
"""

from __future__ import annotations


class StateDiscretizer:
    """Map continuous state features to a discrete state index.

    TODO: ``fit`` learns bin edges from training data only; ``transform``
    maps an observation to an integer state.
    """

    def fit(self, train_features):
        raise NotImplementedError

    def transform(self, observation):
        raise NotImplementedError


class QLearningAgent:
    """Tabular Q-learning (off-policy TD control).

    TODO: Q-table, epsilon-greedy ``act``, ``update`` with the max over next
    actions, ``train`` loop, ``greedy_action`` for evaluation, save / load.
    """

    def __init__(self, n_states, n_actions, *, alpha, gamma, epsilon, seed=None):
        raise NotImplementedError

    def act(self, state, explore=True):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError


class SarsaAgent:
    """Tabular SARSA (on-policy TD control).

    TODO: same interface as ``QLearningAgent``, but ``update`` uses the next
    action chosen by the behavior policy.
    """

    def __init__(self, n_states, n_actions, *, alpha, gamma, epsilon, seed=None):
        raise NotImplementedError

    def act(self, state, explore=True):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError
