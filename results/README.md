# FinAdvisor empirical results

`multi_asset_evaluation.csv` contains the completed 30-run experiment: 15 assets x 2 models.

The 12-ETF core remains preserved under `archived_12_etf_core/`. The active result adds AAPL, MSFT and NVDA as the individual-equity robustness extension.

Generated summary files:
- `model_summary.csv`
- `asset_summary.csv`

Important reproducibility note: the AAPL/MSFT/NVDA run was produced from a separately frozen equity snapshot on the research machine. Before final submission, copy that machine's `data/verified_equity_adjclose.csv` and `data/verified_equity_adjclose.provenance.json` into this project so the exact stock data and SHA-256 used for the six equity runs are archived alongside these results.
