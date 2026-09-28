"""Batch evaluation for the 12-ETF core plus AAPL/MSFT/NVDA extension."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from advisor_bot import full_advisor_run
from asset_universe import RESEARCH_ASSETS
from data_source import load_research_panel, save_provenance


def _row(ticker: str, model_name: str, result: dict) -> dict:
    m = result["model_metrics"]
    lt = result["long_term_metrics"]
    advice = result["long_term_advice"]
    return {
        "ticker": ticker,
        "model": model_name,
        "evaluation": m["evaluation"],
        "folds": m["folds"],
        "test_start": m["test_start"],
        "test_end": m["test_end"],
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "buy_hold_total_return": lt["Buy & Hold"]["total_return"],
        "buy_hold_cagr": lt["Buy & Hold"]["cagr"],
        "buy_hold_sharpe": lt["Buy & Hold"]["sharpe"],
        "buy_hold_max_drawdown": lt["Buy & Hold"]["max_drawdown"],
        "systematic_total_return": lt["Trend + Momentum"]["total_return"],
        "systematic_cagr": lt["Trend + Momentum"]["cagr"],
        "systematic_sharpe": lt["Trend + Momentum"]["sharpe"],
        "systematic_max_drawdown": lt["Trend + Momentum"]["max_drawdown"],
        "dynamic_total_return": lt["FinAdvisor Dynamic"]["total_return"],
        "dynamic_cagr": lt["FinAdvisor Dynamic"]["cagr"],
        "dynamic_sharpe": lt["FinAdvisor Dynamic"]["sharpe"],
        "dynamic_max_drawdown": lt["FinAdvisor Dynamic"]["max_drawdown"],
        "dynamic_average_exposure": lt["FinAdvisor Dynamic"]["average_exposure"],
        "current_regime": advice.regime,
        "current_dynamic_allocation": advice.dynamic_allocation,
        "status": "ok",
        "error": "",
    }


def run_batch(assets, models, start: str, transaction_cost: float, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel, _, etf_prov, equity_prov = load_research_panel()
        save_provenance(etf_prov)
        save_provenance(equity_prov)
    rows = []
    for ticker in assets:
        if ticker not in panel.columns:
            rows.extend({"ticker": ticker, "model": m, "status": "failed", "error": "ticker absent from frozen research panel"} for m in models)
            continue
        raw = panel[["Date", ticker]].rename(columns={ticker: "Adj Close"}).dropna().copy()
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw[raw["Date"] >= pd.Timestamp(start)].reset_index(drop=True)
        raw["Close"] = raw["Adj Close"]
        for model_name in models:
            try:
                result = full_advisor_run(
                    ticker=ticker,
                    model_name=model_name,
                    start=start,
                    transaction_cost=transaction_cost,
                    raw_data=raw,
                )
                rows.append(_row(ticker, model_name, result))
                print(f"OK   {ticker:5s} | {model_name}")
            except Exception as exc:
                rows.append({"ticker": ticker, "model": model_name, "status": "failed", "error": str(exc)})
                print(f"FAIL {ticker:5s} | {model_name}: {exc}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinAdvisor across the 12-ETF core and three-equity robustness extension.")
    parser.add_argument("--assets", nargs="+", default=RESEARCH_ASSETS)
    parser.add_argument("--models", nargs="+", default=["Random Forest", "Logistic Regression"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--output", default="results/multi_asset_evaluation.csv")
    args = parser.parse_args()

    panel, _, etf_prov, equity_prov = load_research_panel()
    save_provenance(etf_prov)
    save_provenance(equity_prov)
    frame = run_batch(args.assets, args.models, args.start, args.transaction_cost, panel=panel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nSaved {len(frame)} rows to {output}")
    ok = frame[frame["status"] == "ok"] if "status" in frame else frame.iloc[0:0]
    print(f"Successful runs: {len(ok)}/{len(frame)}")


if __name__ == "__main__":
    main()
