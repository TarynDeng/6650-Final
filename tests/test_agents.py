"""Smoke tests: each agent trains briefly on synthetic data and acts legally."""

import numpy as np
import pytest

from src.config import DQNConfig, EnvConfig
from src.deep_rl import DQNAgent, Standardizer
from src.environment import N_ACTIONS, TradingEnv
from src.evaluation import buy_and_hold_policy, loss_category, run_policy, terminal_loss
from src.tabular_rl import QLearningAgent, SarsaAgent, StateDiscretizer
from src.training import train_agent
from tests.test_environment import make_features


def make_agent(name, df, seed=0):
    if name == "dqn":
        return DQNAgent(Standardizer().fit(df), gamma=0.99,
                        cfg=DQNConfig(learning_starts=100, batch_size=32), seed=seed)
    cls = QLearningAgent if name == "qlearning" else SarsaAgent
    return cls(StateDiscretizer().fit(df), gamma=0.99, seed=seed)


@pytest.mark.parametrize("name", ["qlearning", "sarsa", "dqn"])
def test_agent_trains_and_acts(name, tmp_path):
    df = make_features(n=200)
    agent = make_agent(name, df)
    log = train_agent(agent, TradingEnv(df, EnvConfig(), episode_len=60), total_steps=600,
                      eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5, seed=0,
                      val_env=TradingEnv(df), eval_every=200)
    assert log["eval_return"].notna().sum() == 3
    res = run_policy(TradingEnv(df), lambda o: agent.act(o, explore=False))
    assert len(res["actions"]) == len(df) - 1
    assert all(0 <= a < N_ACTIONS for a in res["actions"])
    # Save / load round trip gives the same greedy actions.
    path = tmp_path / f"{name}.bin"
    agent.save(path)
    other = make_agent(name, df, seed=1)
    other.load(path)
    obs = TradingEnv(df).reset(seed=0)[0]
    assert other.act(obs, explore=False) == agent.act(obs, explore=False)


def test_discretizer_state_count_and_range():
    df = make_features(n=200)
    disc = StateDiscretizer((3, 3, 3), 5).fit(df)
    assert disc.n_states == 135
    env = TradingEnv(df)
    obs, _ = env.reset(seed=0)
    assert 0 <= disc.transform(obs) < disc.n_states


def test_loss_categories():
    assert [loss_category(x) for x in (0.0, 0.1, 0.2, 0.3, 0.5)] == [1, 2, 2, 3, 4]
    assert terminal_loss([100, 120]) == 0.0
    assert terminal_loss([100, 70]) == pytest.approx(0.3)


def test_buy_and_hold_trades_once():
    res = run_policy(TradingEnv(make_features()), buy_and_hold_policy())
    assert res["n_trades"] == 1
