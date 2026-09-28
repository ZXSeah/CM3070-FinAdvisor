from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_engine import (
    _monthly_rebalance,
    backtest_long_term_strategies,
    calculate_performance_metrics,
    prepare_long_term_features,
)


def test_long_term_features_are_bounded(synthetic_prices):
    data = prepare_long_term_features(synthetic_prices)
    assert {"sma_200", "momentum_126d", "momentum_252d", "trend_score", "regime", "systematic_target"}.issubset(data.columns)
    assert data["trend_score"].between(0, 1).all()
    assert data["systematic_target"].between(0, 1).all()


def test_monthly_rebalance_holds_target_between_month_ends():
    dates = pd.bdate_range("2024-01-02", "2024-03-15")
    target = pd.Series(np.linspace(0, 1, len(dates)))
    held = _monthly_rebalance(target, pd.Series(dates))
    # First target seeds strategy; within January it remains unchanged until Jan month-end.
    jan = pd.Series(dates).dt.month == 1
    jan_values_before_last = held[jan].iloc[:-1]
    assert jan_values_before_last.nunique() == 1
    # New January month-end target is held through February until February month-end.
    feb = pd.Series(dates).dt.month == 2
    assert held[feb].iloc[:-1].nunique() == 1


def test_long_term_backtest_exposure_and_timing(synthetic_prices):
    start = synthetic_prices["Date"].iloc[500].date().isoformat()
    bt, metrics, advice = backtest_long_term_strategies(synthetic_prices, start_date=start)
    for col in ["systematic_exposure", "dynamic_exposure", "systematic_executed_exposure", "dynamic_executed_exposure"]:
        assert bt[col].between(0, 1).all()
    assert bt["systematic_executed_exposure"].iloc[0] == 0
    assert bt["dynamic_executed_exposure"].iloc[0] == 0
    assert bt["buy_hold_return"].iloc[0] == 0
    assert np.isclose(bt["buy_hold_equity"].iloc[0], 1.0)
    assert set(metrics) == {"Buy & Hold", "Trend + Momentum", "FinAdvisor Dynamic"}
    assert 0 <= advice.dynamic_allocation <= 1


def test_long_term_transaction_cost_matches_executed_turnover(synthetic_prices):
    start = synthetic_prices["Date"].iloc[500].date().isoformat()
    cost_rate = 0.0025
    bt, _, _ = backtest_long_term_strategies(synthetic_prices, start_date=start, transaction_cost=cost_rate)
    assert np.allclose(bt["dynamic_cost"], bt["dynamic_turnover"] * cost_rate)
    assert np.allclose(bt["systematic_cost"], bt["systematic_turnover"] * cost_rate)


def test_performance_metrics_known_simple_series():
    returns = pd.Series([0.0, 0.10, -0.05])
    equity = (1 + returns).cumprod()
    turnover = pd.Series([0.0, 1.0, 0.5])
    costs = turnover * 0.001
    exposure = pd.Series([0.0, 1.0, 0.5])
    m = calculate_performance_metrics(returns, equity, turnover, costs, exposure)
    assert np.isclose(m["total_return"], 1.10 * 0.95 - 1)
    assert np.isclose(m["max_drawdown"], -0.05)
    assert np.isclose(m["turnover"], 1.5)
    assert np.isclose(m["transaction_costs"], 0.0015)
    assert np.isclose(m["average_exposure"], 0.5)
