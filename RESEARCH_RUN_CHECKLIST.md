# Research Run Checklist — 15 Assets

1. Use Python 3.10 and install `requirements.txt`.
2. Run `python -m pytest -q` and confirm **21 passed**.
3. With internet access, run `python reproduce_research.py --refresh-data` once to freeze the ETF panel plus AAPL/MSFT/NVDA.
4. Confirm both provenance files exist and contain SHA-256 hashes.
5. Confirm `results/multi_asset_evaluation.csv` contains **30 requested rows (15 assets × 2 models)** or document every failure.
6. Run `python generate_results_summary.py`.
7. Archive the two frozen data CSVs, two provenance JSON files and all result CSV/JSON files together.
8. Do not mix the archived 24-run ETF-only results with the new 30-run experiment.
9. Update Chapter 5 of the report with measured AAPL/MSFT/NVDA results only after the frozen extension run completes.
