"""Long-horizon strategy, regime detection, and portfolio evaluation for FinAdvisor AI.

The module deliberately separates long-term allocation from the short-horizon ML
classifier in advisor_bot.py. Signals use only information available at or before
the decision date and portfolio exposure is changed at month-end to reduce
turnover. This is an academic prototype, not financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class LongTermAdvice:
    regime: str
    trend_score: float
    systematic_allocation: float
    dynamic_allocation: float
    ml_probability_up: Optional[float]
    summary: str


def prepare_long_term_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create slow-moving trend, momentum, volatility, and drawdown features."""
    data = df.copy().sort_values("Date").reset_index(drop=True)
    price = data["Adj Close"].astype(float)

    data["asset_return"] = price.pct_change().fillna(0.0)
    data["sma_50"] = price.rolling(50).mean()
    data["sma_200"] = price.rolling(200).mean()
    data["momentum_63d"] = price.pct_change(63)
    data["momentum_126d"] = price.pct_change(126)
    data["momentum_252d"] = price.pct_change(252)
    data["volatility_20d"] = data["asset_return"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    data["volatility_63d"] = data["asset_return"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    data["volatility_reference"] = data["volatility_20d"].rolling(252, min_periods=60).median()
    data["running_peak"] = price.cummax()
    data["drawdown"] = price / data["running_peak"] - 1.0

    # Four transparent, pre-defined trend/momentum conditions.
    conditions = pd.DataFrame(
        {
            "above_200d": price > data["sma_200"],
            "fast_above_slow": data["sma_50"] > data["sma_200"],
            "positive_6m": data["momentum_126d"] > 0,
            "positive_12m": data["momentum_252d"] > 0,
        }
    ).astype(float)
    data["trend_score"] = conditions.mean(axis=1)
    data["high_volatility"] = data["volatility_20d"] > data["volatility_reference"]

    bull = (price > data["sma_200"]) & (data["momentum_126d"] > 0)
    bear = (price < data["sma_200"]) & (data["momentum_126d"] < 0)
    data["regime"] = "Neutral"
    data.loc[bull & ~data["high_volatility"], "regime"] = "Bull / Lower Volatility"
    data.loc[bull & data["high_volatility"], "regime"] = "Bull / Higher Volatility"
    data.loc[bear & ~data["high_volatility"], "regime"] = "Bear / Lower Volatility"
    data.loc[bear & data["high_volatility"], "regime"] = "Bear / Higher Volatility"

    # Convert the transparent score into a long/cash allocation. High volatility
    # reduces, but does not reverse, the strategic allocation.
    base_allocation = data["trend_score"].clip(0.0, 1.0)
    volatility_multiplier = np.where(data["high_volatility"], 0.75, 1.0)
    data["systematic_target"] = (base_allocation * volatility_multiplier).clip(0.0, 1.0)

    return data


def _monthly_rebalance(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Hold a target allocation constant between month-end decisions."""
    frame = pd.DataFrame({"Date": pd.to_datetime(dates), "target": series.astype(float)})
    month = frame["Date"].dt.to_period("M")
    rebalance = month.ne(month.shift(-1))
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.loc[rebalance] = frame.loc[rebalance, "target"]

    # Allow the first available target to seed the strategy, while subsequent
    # changes occur only at month-end. shift(1) in the backtest prevents same-day
    # look-ahead in return application.
    first_valid = frame["target"].first_valid_index()
    if first_valid is not None:
        out.loc[first_valid] = frame.loc[first_valid, "target"]
    return out.ffill().fillna(0.0)


def calculate_performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    turnover: Optional[pd.Series] = None,
    transaction_costs: Optional[pd.Series] = None,
    exposure: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Return a consistent set of risk/return metrics for strategy comparison."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    e = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n == 0 or len(e) == 0:
        return {k: 0.0 for k in [
            "total_return", "cagr", "annual_volatility", "sharpe", "sortino",
            "max_drawdown", "calmar", "turnover", "transaction_costs", "average_exposure"
        ]}

    total_return = float(e.iloc[-1] - 1.0)
    years = max(n / TRADING_DAYS, 1 / TRADING_DAYS)
    final_equity = float(e.iloc[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    annual_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() / r.std(ddof=0)) * np.sqrt(TRADING_DAYS)) if r.std(ddof=0) > 0 else 0.0
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    sortino = float((r.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS))) if downside_std > 0 else 0.0
    peak = e.cummax()
    max_drawdown = float((e / peak - 1.0).min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": float(cagr),
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover": float(turnover.sum()) if turnover is not None else 0.0,
        "transaction_costs": float(transaction_costs.sum()) if transaction_costs is not None else 0.0,
        "average_exposure": float(exposure.mean()) if exposure is not None else 1.0,
    }


def backtest_long_term_strategies(
    raw_data: pd.DataFrame,
    start_date: str,
    ml_test_data: Optional[pd.DataFrame] = None,
    transaction_cost: float = 0.001,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], LongTermAdvice]:
    """Backtest buy-and-hold, systematic trend/momentum, and FinAdvisor Dynamic.

    FinAdvisor Dynamic uses the slow systematic allocation as the primary risk
    budget and the out-of-sample ML probability only as a conservative
    confirmation multiplier. This prevents a five-day classifier from overriding
    the long-horizon regime model.
    """
    data = prepare_long_term_features(raw_data)

    if ml_test_data is not None and "probability_up" in ml_test_data.columns:
        ml = ml_test_data[["Date", "probability_up"]].copy()
        ml["Date"] = pd.to_datetime(ml["Date"])
        data = data.merge(ml, on="Date", how="left")
    else:
        data["probability_up"] = np.nan

    # Preserve only dates in the common out-of-sample evaluation period.
    data = data[data["Date"] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No observations remain in the requested long-term evaluation period.")

    # Restart return measurement at 0 on the first evaluation observation so all
    # strategies begin from exactly the same $1 capital base. Slow indicators are
    # still computed on the complete pre-test history above.
    data["asset_return"] = data["Adj Close"].pct_change().fillna(0.0)

    data["ml_probability_filled"] = data["probability_up"].ffill()
    ml_multiplier = np.select(
        [data["ml_probability_filled"] >= 0.55, data["ml_probability_filled"] <= 0.45],
        [1.0, 0.50],
        default=0.75,
    )
    # If ML is unavailable, dynamic defaults to the systematic strategy rather
    # than treating missing data as a negative prediction.
    ml_multiplier = np.where(data["ml_probability_filled"].isna(), 1.0, ml_multiplier)
    data["dynamic_target"] = (data["systematic_target"] * ml_multiplier).clip(0.0, 1.0)

    data["systematic_exposure"] = _monthly_rebalance(data["systematic_target"], data["Date"])
    data["dynamic_exposure"] = _monthly_rebalance(data["dynamic_target"], data["Date"])

    for name in ["systematic", "dynamic"]:
        target_exposure = data[f"{name}_exposure"]
        # A target observed at today's close is executed for the next trading day.
        # Turnover/costs are therefore calculated on the executed (shifted)
        # exposure, keeping timing consistent with the return application.
        executed_exposure = target_exposure.shift(1).fillna(0.0)
        turnover = executed_exposure.diff().abs().fillna(executed_exposure.abs())
        costs = turnover * float(transaction_cost)
        strategy_return = executed_exposure * data["asset_return"] - costs
        data[f"{name}_executed_exposure"] = executed_exposure
        data[f"{name}_turnover"] = turnover
        data[f"{name}_cost"] = costs
        data[f"{name}_return"] = strategy_return
        data[f"{name}_equity"] = (1.0 + strategy_return).cumprod()

    data["buy_hold_return"] = data["asset_return"]
    data["buy_hold_equity"] = (1.0 + data["buy_hold_return"]).cumprod()

    metrics = {
        "Buy & Hold": calculate_performance_metrics(
            data["buy_hold_return"], data["buy_hold_equity"], exposure=pd.Series(1.0, index=data.index)
        ),
        "Trend + Momentum": calculate_performance_metrics(
            data["systematic_return"],
            data["systematic_equity"],
            turnover=data["systematic_turnover"],
            transaction_costs=data["systematic_cost"],
            exposure=data["systematic_executed_exposure"],
        ),
        "FinAdvisor Dynamic": calculate_performance_metrics(
            data["dynamic_return"],
            data["dynamic_equity"],
            turnover=data["dynamic_turnover"],
            transaction_costs=data["dynamic_cost"],
            exposure=data["dynamic_executed_exposure"],
        ),
    }

    latest = data.iloc[-1]
    current_ml = None if pd.isna(latest["ml_probability_filled"]) else float(latest["ml_probability_filled"])
    advice = LongTermAdvice(
        regime=str(latest["regime"]),
        trend_score=float(latest["trend_score"]),
        systematic_allocation=float(latest["systematic_exposure"]),
        dynamic_allocation=float(latest["dynamic_exposure"]),
        ml_probability_up=current_ml,
        summary=(
            f"Current regime: {latest['regime']}. The long-horizon trend/momentum score is "
            f"{float(latest['trend_score']):.0%}; the systematic model therefore holds "
            f"{float(latest['systematic_exposure']):.0%} exposure, while the FinAdvisor Dynamic "
            f"allocation is {float(latest['dynamic_exposure']):.0%}."
        ),
    )
    return data, metrics, advice


def metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert nested strategy metrics to a presentation-friendly DataFrame."""
    rows = []
    for strategy, values in metrics.items():
        row = {"Strategy": strategy, **values}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")
