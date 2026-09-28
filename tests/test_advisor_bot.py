from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advisor_bot import (
    FEATURE_COLUMNS,
    backtest_strategy,
    engineer_features,
    engineer_predictor_features,
    full_advisor_run,
    walk_forward_evaluate,
)


def test_predictor_features_keep_latest_available_observation(synthetic_prices):
    predictors = engineer_predictor_features(synthetic_prices)
    assert predictors["Date"].iloc[-1] == synthetic_prices["Date"].iloc[-1]
    assert predictors[FEATURE_COLUMNS].notna().all().all()


def test_labelled_features_drop_unknown_future_horizon(synthetic_prices):
    featured = engineer_features(synthetic_prices, horizon=5)
    assert featured["future_return"].notna().all()
    assert featured["target"].isin([0, 1]).all()
    assert featured["Date"].max() == synthetic_prices["Date"].iloc[-6]


@pytest.mark.parametrize("model_name", ["Random Forest", "Logistic Regression"])
def test_walk_forward_produces_ordered_nonoverlapping_future_folds(synthetic_prices, model_name):
    featured = engineer_features(synthetic_prices)
    metrics, oos, folds = walk_forward_evaluate(featured, model_name=model_name, n_splits=5)
    assert metrics["folds"] == 5
    assert len(folds) == 5
    assert oos["Date"].is_monotonic_increasing
    assert oos["Date"].is_unique
    assert oos["probability_up"].between(0, 1).all()
    for row in folds.itertuples():
        assert pd.Timestamp(row.train_end) < pd.Timestamp(row.test_start)


def test_walk_forward_rejects_too_little_data(synthetic_prices):
    featured = engineer_features(synthetic_prices).iloc[:250]
    with pytest.raises(ValueError, match="At least 300"):
        walk_forward_evaluate(featured)


def test_full_run_uses_latest_price_for_current_recommendation(synthetic_prices):
    result = full_advisor_run("SYNTH", raw_data=synthetic_prices, model_name="Logistic Regression")
    assert result["predictor_data"]["Date"].iloc[-1] == synthetic_prices["Date"].iloc[-1]
    advisor = result["advisor_result"]
    assert advisor.recommendation in {"BUY", "HOLD", "SELL"}
    assert 0 <= advisor.probability_up <= 1
    assert len(advisor.explanation) >= 4


def test_short_horizon_backtest_charges_cost_when_position_is_executed():
    dates = pd.bdate_range("2024-01-01", periods=4)
    test = pd.DataFrame({
        "Date": dates,
        "Adj Close": [100.0, 101.0, 102.0, 103.0],
        "probability_up": [0.70, 0.70, 0.30, 0.30],
    })
    bt, metrics = backtest_strategy(test, transaction_cost=0.01)
    # Day 0 signal is not executable until day 1, so no day-0 turnover/cost.
    assert bt.loc[0, "executed_position"] == 0
    assert bt.loc[0, "trade"] == 0
    assert bt.loc[1, "executed_position"] == 1
    assert bt.loc[1, "trade"] == 1
    # Sell signal at day 2 executes on day 3.
    assert bt.loc[2, "executed_position"] == 1
    assert bt.loc[3, "executed_position"] == 0
    assert bt.loc[3, "trade"] == 1
    assert metrics["number_of_trades"] == 2
