# Prototype Submission Notes

The formal research design now contains a 12-ETF core plus AAPL, MSFT and NVDA as an individual-equity robustness extension.

Before treating the three stocks as empirical final-report evidence:

1. run `python reproduce_research.py --refresh-data` on a connected machine;
2. archive `data/verified_equity_adjclose.csv` and its provenance JSON;
3. confirm the batch contains 30 requested asset/model rows;
4. regenerate the summary files; and
5. update the report with the measured stock-extension results.

Do not invent or copy old demonstration-data returns for AAPL/MSFT/NVDA. The current package deliberately fails loudly if the formal equity extension cannot be downloaded or validated.
