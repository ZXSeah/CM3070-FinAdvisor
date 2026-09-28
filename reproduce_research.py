"""One-command reproduction of the 15-asset FinAdvisor research experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asset_universe import RESEARCH_ASSETS, ETF_RESEARCH_ASSETS, EQUITY_EXTENSION_ASSETS
from batch_evaluate import run_batch
from data_source import load_research_panel, save_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download both the ETF source and AAPL/MSFT/NVDA extension before evaluation.",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    args = parser.parse_args()

    panel, research_prov, etf_prov, equity_prov = load_research_panel(
        refresh_etf=args.refresh_data,
        refresh_equities=args.refresh_data,
    )
    save_provenance(etf_prov)
    save_provenance(equity_prov)

    results = run_batch(
        RESEARCH_ASSETS,
        ["Random Forest", "Logistic Regression"],
        start=args.start,
        transaction_cost=args.transaction_cost,
        panel=panel,
    )
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/multi_asset_evaluation.csv", index=False)

    ok = results[results["status"] == "ok"].copy()
    summary = {
        "successful_runs": int(len(ok)),
        "requested_runs": int(len(results)),
        "assets": RESEARCH_ASSETS,
        "etf_core_assets": ETF_RESEARCH_ASSETS,
        "individual_equity_extension": EQUITY_EXTENSION_ASSETS,
        "models": ["Random Forest", "Logistic Regression"],
        "etf_data_sha256": research_prov.etf_sha256,
        "equity_data_sha256": research_prov.equity_sha256,
        "combined_research_sha256": research_prov.combined_sha256,
        "data_first_date": research_prov.first_date,
        "data_last_date": research_prov.last_date,
        "mean_accuracy": float(ok["accuracy"].mean()) if len(ok) else None,
        "mean_dynamic_minus_buy_hold_cagr": float((ok["dynamic_cagr"] - ok["buy_hold_cagr"]).mean()) if len(ok) else None,
        "dynamic_lower_drawdown_fraction": float((ok["dynamic_max_drawdown"] > ok["buy_hold_max_drawdown"]).mean()) if len(ok) else None,
    }
    Path("results/reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
