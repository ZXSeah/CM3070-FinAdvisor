"""Aggregate FinAdvisor's archived 15-asset research output into report-ready CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else float("nan")


def summarise(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "status" not in results.columns:
        raise ValueError("Results file is missing status column.")
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("Results file contains no successful research runs.")

    ok["dynamic_minus_buy_hold_cagr"] = ok["dynamic_cagr"] - ok["buy_hold_cagr"]
    ok["dynamic_minus_buy_hold_sharpe"] = ok["dynamic_sharpe"] - ok["buy_hold_sharpe"]
    # Drawdown values are negative. A greater value means a shallower drawdown.
    ok["dynamic_drawdown_improvement"] = ok["dynamic_max_drawdown"] - ok["buy_hold_max_drawdown"]
    ok["dynamic_beats_cagr"] = ok["dynamic_minus_buy_hold_cagr"] > 0
    ok["dynamic_beats_sharpe"] = ok["dynamic_minus_buy_hold_sharpe"] > 0
    ok["dynamic_lower_drawdown"] = ok["dynamic_drawdown_improvement"] > 0

    model_rows = []
    for model, g in ok.groupby("model", sort=True):
        model_rows.append({
            "model": model,
            "successful_assets": int(g["ticker"].nunique()),
            "mean_accuracy": _safe_mean(g["accuracy"]),
            "mean_f1": _safe_mean(g["f1"]),
            "mean_buy_hold_cagr": _safe_mean(g["buy_hold_cagr"]),
            "mean_dynamic_cagr": _safe_mean(g["dynamic_cagr"]),
            "mean_dynamic_minus_buy_hold_cagr": _safe_mean(g["dynamic_minus_buy_hold_cagr"]),
            "fraction_dynamic_beats_buy_hold_cagr": float(g["dynamic_beats_cagr"].mean()),
            "fraction_dynamic_beats_buy_hold_sharpe": float(g["dynamic_beats_sharpe"].mean()),
            "fraction_dynamic_lower_drawdown": float(g["dynamic_lower_drawdown"].mean()),
            "mean_dynamic_average_exposure": _safe_mean(g["dynamic_average_exposure"]),
        })

    asset_rows = []
    for ticker, g in ok.groupby("ticker", sort=True):
        asset_rows.append({
            "ticker": ticker,
            "successful_models": int(g["model"].nunique()),
            "mean_accuracy": _safe_mean(g["accuracy"]),
            "mean_f1": _safe_mean(g["f1"]),
            "buy_hold_cagr": _safe_mean(g["buy_hold_cagr"]),
            "mean_dynamic_cagr": _safe_mean(g["dynamic_cagr"]),
            "mean_dynamic_minus_buy_hold_cagr": _safe_mean(g["dynamic_minus_buy_hold_cagr"]),
            "buy_hold_max_drawdown": _safe_mean(g["buy_hold_max_drawdown"]),
            "mean_dynamic_max_drawdown": _safe_mean(g["dynamic_max_drawdown"]),
            "mean_dynamic_drawdown_improvement": _safe_mean(g["dynamic_drawdown_improvement"]),
            "mean_dynamic_average_exposure": _safe_mean(g["dynamic_average_exposure"]),
        })

    return pd.DataFrame(model_rows), pd.DataFrame(asset_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-asset summary tables from an archived FinAdvisor research run.")
    parser.add_argument("--input", default="results/multi_asset_evaluation.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    results = pd.read_csv(args.input)
    model_summary, asset_summary = summarise(results)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_summary.to_csv(out / "model_summary.csv", index=False)
    asset_summary.to_csv(out / "asset_summary.csv", index=False)
    print(model_summary.to_string(index=False))
    print(f"\nSaved {out / 'model_summary.csv'} and {out / 'asset_summary.csv'}")


if __name__ == "__main__":
    main()
