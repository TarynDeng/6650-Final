"""Deep RL: DQN with experience replay and a target network.

See ``docs/proposal.md`` Section 4. Builds on HW-2 (PyTorch DQN on
Gymnasium; Ray RLlib as an alternative).

Scope: the basic DQN taught in the course, i.e. an MLP Q-network, an
experience replay buffer, a periodically updated target network, and
epsilon-greedy exploration. No Double / Dueling / Rainbow extensions,
PPO, or SAC.

Rules the implementation must respect:
    * Input is the continuous feature vector from ``environment.py``,
      normalized with training-split statistics only.
    * Optional control experiment: feed the same discrete state encoding the
      tabular agents use (one-hot of the ``StateDiscretizer`` index).
    * The discount factor gamma must match the tabular agents.
    * Log return vs. environment steps for learning curves, plus wall-clock
      training time.

Still open: HW-2 PyTorch implementation vs. RLlib, network size, training
budget, and hyperparameter ranges.
"""

from __future__ import annotations


class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    TODO: ``push`` transitions and ``sample`` uniform minibatches.
    """

    def __init__(self, capacity, seed=None):
        raise NotImplementedError

    def push(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError


class QNetwork:
    """MLP mapping a state vector to one Q-value per action.

    TODO: implement as a ``torch.nn.Module``. Hidden sizes are an open
    decision; pass them from the experiment config.
    """

    def __init__(self, state_dim, n_actions, hidden_sizes):
        raise NotImplementedError


class DQNAgent:
    """DQN agent with an online network and a target network.

    TODO: epsilon-greedy ``act``, ``update`` from replay minibatches,
    periodic target-network sync, ``train`` loop, ``greedy_action`` for
    evaluation, save / load of model weights under ``results/``.
    """

    def __init__(self, state_dim, n_actions, *, gamma, lr, seed=None, **kwargs):
        raise NotImplementedError

    def act(self, state, explore=True):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
