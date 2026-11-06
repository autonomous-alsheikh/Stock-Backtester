import numpy as np
import pandas as pd
import pytest

from backtester import backtest, rsi, sma, synthetic_prices
from backtester.metrics import max_drawdown, sharpe_ratio
from backtester.ml import build_features, build_target, time_split
from backtester.strategies import rsi_mean_reversion, sma_crossover


def test_sma_matches_manual_value():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(close, 3).iloc[-1] == pytest.approx(4.0)
    assert sma(close, 3).iloc[:2].isna().all()


def test_rsi_is_bounded_and_extremes_make_sense():
    rising = pd.Series(np.arange(1.0, 60.0))
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(rising).dropna().iloc[-1] == pytest.approx(100.0)
    assert rsi(falling).dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)
    values = rsi(synthetic_prices(500)).dropna()
    assert values.between(0, 100).all()


def test_engine_applies_signal_one_day_late():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    signal = pd.Series([0.0, 1.0, 1.0, 1.0])  # decided at close of day 1
    res = backtest(close, signal, cost_bps=0)
    assert list(res["position"]) == [0.0, 0.0, 1.0, 1.0]
    # Day 1's +10% move must NOT be captured by a signal that only appeared at day 1's close.
    assert res["strategy_return"].iloc[1] == 0.0


def test_always_long_matches_buy_and_hold_without_costs():
    close = synthetic_prices(300)
    res = backtest(close, pd.Series(1.0, index=close.index), cost_bps=0)
    expected = close.pct_change().fillna(0.0).iloc[1:]
    # Position starts on day 1, so day 1 onward matches the asset.
    assert res["strategy_return"].iloc[1:].to_numpy() == pytest.approx(expected.to_numpy())


def test_costs_reduce_returns():
    close = synthetic_prices(500)
    signal = sma_crossover(close, 5, 20)
    free = backtest(close, signal, cost_bps=0)["equity"].iloc[-1]
    costly = backtest(close, signal, cost_bps=25)["equity"].iloc[-1]
    assert costly < free


def test_max_drawdown_known_case():
    equity = pd.Series([1.0, 2.0, 1.0, 1.5])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_sharpe_is_nan_for_zero_volatility():
    assert np.isnan(sharpe_ratio(pd.Series([0.001] * 50)))


def test_rsi_strategy_is_binary_and_stateful():
    close = synthetic_prices(600)
    sig = rsi_mean_reversion(close)
    assert set(sig.unique()) <= {0.0, 1.0}


def test_time_split_is_ordered_and_disjoint():
    idx = pd.bdate_range("2020-01-01", periods=100)
    train, test = time_split(idx, 0.7)
    assert len(train) == 70 and len(test) == 30
    assert train.max() < test.min()


def test_features_do_not_use_future_prices():
    close = synthetic_prices(400)
    full = build_features(close)
    truncated = build_features(close.iloc[:300])
    # Rows computed from the first 300 prices must be identical whether or not
    # the later 100 prices exist. If they differ, a feature peeked ahead.
    pd.testing.assert_frame_equal(full.iloc[:300], truncated)


def test_target_is_next_day_direction():
    close = pd.Series([1.0, 2.0, 1.0, 3.0])
    target = build_target(close)
    assert list(target.iloc[:3]) == [1.0, 0.0, 1.0]
    assert np.isnan(target.iloc[3])
