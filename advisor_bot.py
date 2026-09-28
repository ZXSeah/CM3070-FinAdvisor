"""Financial Advisor Bot - verified-data, walk-forward advisor logic.

Academic decision-support prototype only. It does not provide personalised
financial advice and does not execute trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asset_universe import RESEARCH_ASSETS
from data_source import research_ticker_frame
from strategy_engine import backtest_long_term_strategies


FEATURE_COLUMNS = [
    "daily_return",
    "return_5d",
    "return_20d",
    "return_63d",
    "sma_ratio_20_50",
    "sma_ratio_50_200",
    "rsi_14",
    "volatility_20d",
    "drawdown",
]


@dataclass
class AdvisorResult:
    ticker: str
    recommendation: str
    confidence: float
    probability_up: float
    explanation: List[str]
    risk_warning: str
    latest_features: Dict[str, float]


def _normalise_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
        if "Date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    if "Adj Close" not in out.columns:
        if "Close" not in out.columns:
            raise ValueError("Price data require Close or Adj Close.")
        out["Adj Close"] = out["Close"]
    out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    out = out[["Date", "Adj Close"]].dropna().sort_values("Date").drop_duplicates("Date")
    if (out["Adj Close"] <= 0).any():
        raise ValueError("Adjusted close must be positive.")
    out["Close"] = out["Adj Close"]
    return out.reset_index(drop=True)


def load_market_data(ticker: str, start: str = "2018-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load a ticker from the formal frozen research sources or yfinance for other custom symbols."""
    ticker = ticker.upper().strip()
    if ticker in RESEARCH_ASSETS:
        frame, _ = research_ticker_frame(ticker, start=start, end=end)
        return frame

    # Custom ticker mode is deliberately separate from the frozen research panel.
    try:
        import yfinance as yf  # type: ignore
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            return _normalise_price_frame(df.reset_index())
    except Exception as exc:
        raise FileNotFoundError(f"Could not download custom ticker {ticker}: {exc}") from exc
    raise FileNotFoundError(f"No verified or yfinance data available for {ticker}.")


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def engineer_predictor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-derived predictors without requiring a future label.

    This is used for the current recommendation so the newest available market
    observation is not discarded merely because its future return is unknown.
    """
    data = _normalise_price_frame(df).copy()
    price = data["Adj Close"]
    data["daily_return"] = price.pct_change()
    data["return_5d"] = price.pct_change(5)
    data["return_20d"] = price.pct_change(20)
    data["return_63d"] = price.pct_change(63)
    data["sma_20"] = price.rolling(20).mean()
    data["sma_50"] = price.rolling(50).mean()
    data["sma_200"] = price.rolling(200).mean()
    data["sma_ratio_20_50"] = data["sma_20"] / data["sma_50"] - 1
    data["sma_ratio_50_200"] = data["sma_50"] / data["sma_200"] - 1
    data["rsi_14"] = calculate_rsi(price, 14)
    data["volatility_20d"] = data["daily_return"].rolling(20).std() * np.sqrt(252)
    data["running_peak"] = price.cummax()
    data["drawdown"] = price / data["running_peak"] - 1
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create predictors plus a leakage-safe future-direction target."""
    data = _normalise_price_frame(df).copy()
    price = data["Adj Close"]
    data["daily_return"] = price.pct_change()
    data["return_5d"] = price.pct_change(5)
    data["return_20d"] = price.pct_change(20)
    data["return_63d"] = price.pct_change(63)
    data["sma_20"] = price.rolling(20).mean()
    data["sma_50"] = price.rolling(50).mean()
    data["sma_200"] = price.rolling(200).mean()
    data["sma_ratio_20_50"] = data["sma_20"] / data["sma_50"] - 1
    data["sma_ratio_50_200"] = data["sma_50"] / data["sma_200"] - 1
    data["rsi_14"] = calculate_rsi(price, 14)
    data["volatility_20d"] = data["daily_return"].rolling(20).std() * np.sqrt(252)
    data["running_peak"] = price.cummax()
    data["drawdown"] = price / data["running_peak"] - 1
    data["future_return"] = price.shift(-horizon) / price - 1
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLUMNS + ["future_return"]).copy()
    data["target"] = (data["future_return"] > 0).astype(int)
    return data.reset_index(drop=True)


def _build_model(model_name: str):
    if model_name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
    if model_name != "Random Forest":
        raise ValueError(f"Unknown model: {model_name}")
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced_subsample",
    )


def fit_model(data: pd.DataFrame, model_name: str):
    model = _build_model(model_name)
    model.fit(data[FEATURE_COLUMNS], data["target"])
    return model


def _classification_metrics(y_true, pred) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def walk_forward_evaluate(
    data: pd.DataFrame,
    model_name: str = "Random Forest",
    initial_train_ratio: float = 0.50,
    n_splits: int = 5,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward evaluation with contiguous future folds."""
    if len(data) < 300:
        raise ValueError("At least 300 labelled rows are required for walk-forward evaluation.")
    if not (0.3 <= initial_train_ratio < 0.9):
        raise ValueError("initial_train_ratio must be between 0.3 and 0.9.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    initial_end = int(len(data) * initial_train_ratio)
    remaining = len(data) - initial_end
    if remaining < n_splits * 20:
        raise ValueError("Not enough observations for the requested walk-forward folds.")

    boundaries = np.linspace(initial_end, len(data), n_splits + 1, dtype=int)
    predictions = []
    fold_rows = []

    for fold in range(n_splits):
        test_start_idx, test_end_idx = boundaries[fold], boundaries[fold + 1]
        train = data.iloc[:test_start_idx].copy()
        test = data.iloc[test_start_idx:test_end_idx].copy()
        if test.empty:
            continue
        model = fit_model(train, model_name)
        pred = model.predict(test[FEATURE_COLUMNS])
        prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        scored = test.copy()
        scored["predicted_up"] = pred
        scored["probability_up"] = prob
        scored["fold"] = fold + 1
        predictions.append(scored)
        fm = _classification_metrics(test["target"], pred)
        fold_rows.append({
            "fold": fold + 1,
            "train_start": str(train["Date"].min().date()),
            "train_end": str(train["Date"].max().date()),
            "test_start": str(test["Date"].min().date()),
            "test_end": str(test["Date"].max().date()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            **{k: fm[k] for k in ["accuracy", "precision", "recall", "f1"]},
        })

    oos = pd.concat(predictions, ignore_index=True).sort_values("Date").reset_index(drop=True)
    aggregate = _classification_metrics(oos["target"], oos["predicted_up"])
    aggregate.update({
        "evaluation": "expanding-window walk-forward",
        "folds": int(len(fold_rows)),
        "initial_train_ratio": float(initial_train_ratio),
        "train_rows_initial": int(initial_end),
        "test_rows": int(len(oos)),
        "test_start": str(oos["Date"].min().date()),
        "test_end": str(oos["Date"].max().date()),
    })
    return aggregate, oos, pd.DataFrame(fold_rows)


def generate_recommendation(
    ticker: str,
    model: object,
    predictor_data: pd.DataFrame,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
) -> AdvisorResult:
    latest = predictor_data.iloc[-1]
    probability_up = float(model.predict_proba(latest[FEATURE_COLUMNS].to_frame().T)[0][1])
    if probability_up >= buy_threshold:
        recommendation, confidence = "BUY", probability_up
    elif probability_up <= sell_threshold:
        recommendation, confidence = "SELL", 1 - probability_up
    else:
        recommendation = "HOLD"
        confidence = 1 - abs(probability_up - 0.5) * 2
    return AdvisorResult(
        ticker=ticker.upper(),
        recommendation=recommendation,
        confidence=float(confidence),
        probability_up=probability_up,
        explanation=build_explanation(latest, probability_up, recommendation),
        risk_warning=build_risk_warning(latest),
        latest_features={col: float(latest[col]) for col in FEATURE_COLUMNS},
    )


def build_explanation(latest: pd.Series, probability_up: float, recommendation: str) -> List[str]:
    statements: List[str] = []
    statements.append(
        "The 20-day moving average is above the 50-day moving average." if latest["sma_ratio_20_50"] > 0
        else "The 20-day moving average is below the 50-day moving average."
    )
    statements.append(
        "The 50-day moving average is above the 200-day moving average, supporting the longer trend." if latest["sma_ratio_50_200"] > 0
        else "The 50-day moving average is below the 200-day moving average, indicating longer-term trend weakness."
    )
    statements.append(
        "The asset has positive three-month momentum." if latest["return_63d"] > 0
        else "The asset has negative three-month momentum."
    )
    if latest["rsi_14"] > 70:
        statements.append("RSI is above 70, indicating an extended recent move.")
    elif latest["rsi_14"] < 30:
        statements.append("RSI is below 30, indicating unusually weak recent momentum.")
    else:
        statements.append("RSI is in its neutral range.")
    statements.append(
        f"The final model estimates a {probability_up:.1%} probability of a positive five-day return, producing {recommendation}."
    )
    return statements


def build_risk_warning(latest: pd.Series) -> str:
    if latest["drawdown"] < -0.20:
        return "Risk warning: drawdown exceeds 20%. Historical model output is not a guarantee of future performance."
    if latest["volatility_20d"] > 0.45:
        return "Risk warning: recent annualised volatility is high. Historical model output is not financial advice."
    return "Risk warning: this prototype uses historical market prices and cannot guarantee future performance."


def _annualised_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(252)) if len(r) and sd > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity)
    return float((e / e.cummax() - 1).min())


def backtest_strategy(
    test_data: pd.DataFrame,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
    transaction_cost: float = 0.001,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bt = test_data.copy().sort_values("Date").reset_index(drop=True)
    bt["position"] = np.nan
    bt.loc[bt["probability_up"] >= buy_threshold, "position"] = 1.0
    bt.loc[bt["probability_up"] <= sell_threshold, "position"] = 0.0
    bt["position"] = bt["position"].ffill().fillna(0.0)
    bt["asset_return"] = bt["Adj Close"].pct_change().fillna(0.0)
    # A signal observed at the close becomes the next trading day's executed
    # position. Turnover and transaction costs are charged when that executed
    # position actually changes, keeping execution timing internally consistent.
    bt["executed_position"] = bt["position"].shift(1).fillna(0.0)
    bt["trade"] = bt["executed_position"].diff().abs().fillna(bt["executed_position"].abs())
    bt["strategy_return"] = bt["executed_position"] * bt["asset_return"] - bt["trade"] * transaction_cost
    bt["strategy_equity"] = (1 + bt["strategy_return"]).cumprod()
    bt["buy_hold_equity"] = (1 + bt["asset_return"]).cumprod()
    return bt, {
        "strategy_total_return": float(bt["strategy_equity"].iloc[-1] - 1),
        "buy_hold_total_return": float(bt["buy_hold_equity"].iloc[-1] - 1),
        "strategy_sharpe": _annualised_sharpe(bt["strategy_return"]),
        "buy_hold_sharpe": _annualised_sharpe(bt["asset_return"]),
        "strategy_max_drawdown": _max_drawdown(bt["strategy_equity"]),
        "buy_hold_max_drawdown": _max_drawdown(bt["buy_hold_equity"]),
        "number_of_trades": int(round(bt["trade"].sum())),
    }


def full_advisor_run(
    ticker: str,
    model_name: str = "Random Forest",
    start: str = "2018-01-01",
    transaction_cost: float = 0.001,
    raw_data: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    raw = _normalise_price_frame(raw_data) if raw_data is not None else load_market_data(ticker, start=start)
    labelled = engineer_features(raw)
    predictor_data = engineer_predictor_features(raw)
    model_metrics, oos, folds = walk_forward_evaluate(labelled, model_name=model_name)
    final_model = fit_model(labelled, model_name)
    advisor_result = generate_recommendation(ticker, final_model, predictor_data)
    backtest, backtest_metrics = backtest_strategy(oos, transaction_cost=transaction_cost)
    long_term_backtest, long_term_metrics, long_term_advice = backtest_long_term_strategies(
        raw_data=raw,
        start_date=model_metrics["test_start"],
        ml_test_data=oos,
        transaction_cost=transaction_cost,
    )
    return {
        "raw_data": raw,
        "featured_data": labelled,
        "predictor_data": predictor_data,
        "model": final_model,
        "model_metrics": model_metrics,
        "walk_forward_folds": folds,
        "advisor_result": advisor_result,
        "backtest": backtest,
        "backtest_metrics": backtest_metrics,
        "long_term_backtest": long_term_backtest,
        "long_term_metrics": long_term_metrics,
        "long_term_advice": long_term_advice,
    }
