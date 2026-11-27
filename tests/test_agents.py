"""Agent tests: exact update rules, training loop, determinism, save/load."""

import numpy as np
import pytest
import torch

from src.config import DQNConfig, EnvConfig, TabularConfig
from src.deep_rl import DQNAgent, Standardizer
from src.environment import N_ACTIONS, TradingEnv
from src.evaluation import run_policy
from src.tabular_rl import QLearningAgent, SarsaAgent, StateDiscretizer
from src.training import train_agent
from tests.helpers import make_features

GAMMA = 0.9


def make_agent(name, df, seed=0):
    if name == "dqn":
        return DQNAgent(Standardizer().fit(df), gamma=0.99,
                        cfg=DQNConfig(learning_starts=100, batch_size=32), seed=seed)
    cls = QLearningAgent if name == "qlearning" else SarsaAgent
    return cls(StateDiscretizer().fit(df), gamma=0.99, seed=seed)


class FixedDiscretizer:
    """Maps observation[0] straight to a state index (for exact update tests)."""

    n_states = 2

    def transform(self, obs):
        return int(obs[0])


def tabular(cls):
    agent = cls(FixedDiscretizer(), gamma=GAMMA, cfg=TabularConfig(alpha=0.5), seed=0)
    agent.q[1] = [1.0, 5.0, 2.0, 0, 0, 0, 0, 0, 0]  # max over next actions is 5.0
    return agent


def test_q_learning_and_sarsa_targets_differ_exactly():
    s, s2 = np.array([0.0]), np.array([1.0])
    q = tabular(QLearningAgent)
    q.update(s, 3, 0.1, s2, next_action=2, terminated=False)
    assert q.q[0, 3] == pytest.approx(0.5 * (0.1 + GAMMA * 5.0))  # max over next actions
    sarsa = tabular(SarsaAgent)
    sarsa.update(s, 3, 0.1, s2, next_action=2, terminated=False)
    assert sarsa.q[0, 3] == pytest.approx(0.5 * (0.1 + GAMMA * 2.0))  # the action actually taken


@pytest.mark.parametrize("cls", [QLearningAgent, SarsaAgent])
def test_tabular_termination_stops_bootstrapping(cls):
    agent = tabular(cls)
    agent.update(np.array([0.0]), 3, 0.1, np.array([1.0]), next_action=1, terminated=True)
    assert agent.q[0, 3] == pytest.approx(0.5 * 0.1)


def test_dqn_target_and_soft_update():
    df = make_features(n=200)
    agent = DQNAgent(Standardizer().fit(df), gamma=GAMMA,
                     cfg=DQNConfig(learning_starts=1, batch_size=1, lr=0.0, target_update_rate=0.25), seed=0)
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.add_(1.0)  # make target != online
    before = [p.clone() for p in agent.target_net.parameters()]
    online = [p.clone() for p in agent.policy_net.parameters()]
    obs = TradingEnv(df).reset(seed=0)[0]
    agent.update(obs, 2, 0.1, obs, None, terminated=False)
    for t0, o, t1 in zip(before, online, agent.target_net.parameters(), strict=True):
        assert torch.allclose(t1, t0 + 0.25 * (o - t0), atol=1e-6)  # lr=0: online unchanged


@pytest.mark.parametrize("terminated", [False, True])
def test_dqn_target_bootstraps_only_when_not_terminated(terminated):
    df = make_features(n=200)
    agent = DQNAgent(Standardizer().fit(df), gamma=GAMMA,
                     cfg=DQNConfig(learning_starts=1, batch_size=1), seed=0)
    seen = {}
    loss_fn = agent.loss_fn

    def capture(pred, target):
        seen["target"] = target.item()
        return loss_fn(pred, target)

    agent.loss_fn = capture
    obs = TradingEnv(df).reset(seed=0)[0]
    with torch.no_grad():
        next_q = agent.target_net(torch.as_tensor(agent.standardizer.transform(obs))).max().item()
    agent.update(obs, 2, 0.3, obs, None, terminated=terminated)
    expected = 0.3 if terminated else 0.3 + GAMMA * next_q
    assert seen["target"] == pytest.approx(expected, rel=1e-5)


class RecordingAgent:
    """Records every update call; acts randomly."""

    def __init__(self):
        self.epsilon, self.calls, self.rng = 1.0, [], np.random.default_rng(0)

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, obs, explore=True):
        return int(self.rng.integers(N_ACTIONS))

    def update(self, *args):
        self.calls.append(args)


def test_training_loop_treats_window_end_as_time_limit():
    df = make_features(n=60)
    agent = RecordingAgent()
    log = train_agent(agent, TradingEnv(df, episode_len=20), total_steps=50, eps_start=1.0,
                      eps_end=0.05, eps_decay_frac=0.5, seed=0)
    assert len(agent.calls) == 50
    assert not any(call[-1] for call in agent.calls)  # terminated is never True
    assert all(call[4] is not None for call in agent.calls)  # SARSA always gets a next action
    episodes = log.dropna(subset=["train_return"])
    assert list(episodes["complete"]) == [True, True, False]  # 19 + 19 + 12 steps
    assert agent.epsilon == pytest.approx(0.05)


@pytest.mark.parametrize("name", ["qlearning", "sarsa", "dqn"])
def test_agent_trains_and_save_load_restores_everything(name, tmp_path):
    df = make_features(n=200)
    agent = make_agent(name, df)
    log = train_agent(agent, TradingEnv(df, EnvConfig(), episode_len=60), total_steps=600,
                      eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5, seed=0,
                      val_env=TradingEnv(df), eval_every=200)
    assert log["eval_return"].notna().sum() == 3
    assert log.attrs["val_return"] == log["eval_return"].dropna().iloc[-1]  # final model, not best
    res = run_policy(TradingEnv(df), lambda o: agent.act(o, explore=False))
    assert len(res["actions"]) == len(df) - 1 and 0.0 <= res["avg_position"] <= 1.0
    path = tmp_path / f"{name}.bin"
    agent.save(path)
    other = make_agent(name, make_features(n=200, seed=7) * 3.0, seed=1)  # fit on other data
    other.load(path)
    env = TradingEnv(df)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        a = agent.act(obs, explore=False)
        assert other.act(obs, explore=False) == a
        obs, *_ = env.step(a)


@pytest.mark.parametrize("name", ["qlearning", "sarsa", "dqn"])
def test_training_is_deterministic(name):
    df = make_features(n=200)

    def train_once():
        agent = make_agent(name, df, seed=3)
        log = train_agent(agent, TradingEnv(df, EnvConfig(), episode_len=60), total_steps=500,
                          eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5, seed=3,
                          val_env=TradingEnv(df), eval_every=250)
        res = run_policy(TradingEnv(df), lambda o: agent.act(o, explore=False))
        return log.drop(columns="wall_time"), res["values"]

    (log1, v1), (log2, v2) = train_once(), train_once()
    assert log1.equals(log2)
    assert np.array_equal(v1, v2)


def test_discretizer_state_count_and_range():
    df = make_features(n=200)
    disc = StateDiscretizer((3, 3, 3), 5).fit(df)
    assert disc.n_states == 135
    obs, _ = TradingEnv(df).reset(seed=0)
    assert 0 <= disc.transform(obs) < disc.n_states
