"""Accounting tests for the trading environment (synthetic prices, no download)."""

import numpy as np
import pandas as pd
import pytest

from src.config import EnvConfig
from src.data import compute_market_features
from src.environment import N_ACTIONS, OBS_NAMES, TradingEnv
from tests.helpers import make_features


def run_actions(env, actions):
    obs, info = env.reset(seed=0)
    v0 = info["portfolio_value"]
    rewards, infos = [], []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        infos.append(info)
        assert not term
        if trunc:
            break
    return v0, np.array(rewards), infos


def test_features_have_no_nans_and_use_only_past_prices():
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))),
                      index=pd.bdate_range("2020-01-01", periods=120))
    base = compute_market_features(close.to_frame("Close"))
    assert not base.isna().any().any()
    t = 70
    changed = close.copy()
    changed.iloc[t + 1:] *= 1.5  # change every price after day t
    after = compute_market_features(changed.to_frame("Close"))
    day = close.index[t]
    pd.testing.assert_frame_equal(base.loc[:day], after.loc[:day])


def test_rewards_telescope_to_log_terminal_value():
    env = TradingEnv(make_features(), EnvConfig(fee_rate=0.001))
    rng = np.random.default_rng(1)
    v0, rewards, infos = run_actions(env, rng.integers(0, N_ACTIONS, 200))
    vT = infos[-1]["portfolio_value"]
    assert np.isclose(rewards.sum(), np.log(vT / v0))


def test_episode_length_and_truncation():
    df = make_features(n=50)
    env = TradingEnv(df)
    _, rewards, infos = run_actions(env, [0] * 100)
    assert len(rewards) == len(df) - 1  # N days -> N - 1 steps
    with pytest.raises(RuntimeError):
        env.step(0)


def test_hold_with_all_cash_keeps_value_constant():
    env = TradingEnv(make_features())
    v0, rewards, infos = run_actions(env, [0] * 100)
    assert np.allclose(rewards, 0.0)
    assert infos[-1]["portfolio_value"] == pytest.approx(v0)


def test_buy_and_hold_matches_price_ratio_minus_one_fee():
    df = make_features()
    fee = 0.001
    env = TradingEnv(df, EnvConfig(fee_rate=fee))
    v0, _, infos = run_actions(env, [4] + [0] * 100)
    expected = v0 / (1 + fee) * df["Close"].iloc[-1] / df["Close"].iloc[0]
    assert infos[-1]["portfolio_value"] == pytest.approx(expected)
    assert infos[-1]["cash"] == pytest.approx(0.0, abs=1e-9)


def test_round_trip_costs_two_fees():
    df = make_features()
    fee = 0.001
    env = TradingEnv(df, EnvConfig(fee_rate=fee))
    obs, info = env.reset(seed=0)
    v0 = info["portfolio_value"]
    env.step(4)  # buy all at close 0
    # sell all at close 1
    _, _, _, _, info = env.step(8)
    price_ratio = df["Close"].iloc[1] / df["Close"].iloc[0]
    assert info["shares"] == 0
    assert info["cash"] == pytest.approx(v0 / (1 + fee) * price_ratio * (1 - fee))


def test_no_leverage_no_shorting_and_partial_orders():
    env = TradingEnv(make_features(), EnvConfig(fee_rate=0.001))
    rng = np.random.default_rng(2)
    env.reset(seed=0)
    for a in rng.integers(0, N_ACTIONS, 70):
        _, _, _, trunc, info = env.step(a)
        assert info["cash"] >= 0 and info["shares"] >= 0
        if trunc:
            break
    # Buy 50% of cash spends half the cash (fee included).
    env.reset(seed=0)
    cash0 = env.cash
    env.step(2)
    assert env.cash == pytest.approx(cash0 / 2)
    # Sell 25% of holdings.
    shares = env.shares
    env.step(5)
    assert env.shares == pytest.approx(0.75 * shares)


def test_observation_shape_and_position_fraction():
    env = TradingEnv(make_features())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_NAMES),) and obs[-1] == 0.0
    obs, *_ = env.step(4)
    assert obs[-1] == pytest.approx(1.0)


def test_random_training_windows_have_fixed_length():
    df = make_features(n=80)
    env = TradingEnv(df, episode_len=30)
    starts = set()
    for s in range(10):
        _, info = env.reset(seed=s)
        starts.add(info["date"])
        steps = 0
        trunc = False
        while not trunc:
            *_, trunc, _ = env.step(0)
            steps += 1
        assert steps == 29
    assert len(starts) > 1


def test_tiny_leftover_trades_are_skipped():
    env = TradingEnv(make_features(), EnvConfig(fee_rate=0.001))
    env.reset(seed=0)
    env.cash = 0.004  # less than one cent to spend
    _, _, _, _, info = env.step(4)
    assert info["traded_shares"] == 0 and info["fee"] == 0
    env.shares = 0.004 / env.close[env._t]  # holdings worth less than one cent
    _, _, _, _, info = env.step(8)
    assert info["traded_shares"] == 0


def test_integer_shares_mode_keeps_whole_shares():
    env = TradingEnv(make_features(), EnvConfig(fee_rate=0.001, fractional_shares=False))
    rng = np.random.default_rng(4)
    obs, info = env.reset(seed=0)
    v0, total = info["portfolio_value"], 0.0
    for a in rng.integers(0, N_ACTIONS, 70):
        _, r, _, trunc, info = env.step(a)
        total += r
        assert float(info["shares"]).is_integer() and info["cash"] >= 0
        if trunc:
            break
    assert np.isclose(total, np.log(info["portfolio_value"] / v0))
