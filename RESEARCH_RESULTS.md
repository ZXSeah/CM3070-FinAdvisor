# Completed 15-Asset Research Results

The active `results/multi_asset_evaluation.csv` contains 30 successful asset/model runs: the preserved 12-ETF core plus AAPL, MSFT and NVDA, each evaluated with Random Forest and Logistic Regression.

Headline combined results:
- Mean classification accuracy across all 30 runs: 49.98%.
- Random Forest: 50.95% mean accuracy, 0.538 mean F1.
- Logistic Regression: 49.01% mean accuracy, 0.476 mean F1.
- Mean buy-and-hold CAGR across 15 assets: 19.33%.
- Mean Dynamic CAGR: 8.34% (Random Forest), 7.83% (Logistic Regression).
- Dynamic CAGR exceeds buy-and-hold on 1/15 assets under each model.
- Dynamic maximum drawdown is shallower than buy-and-hold on 15/15 assets under each model.
- Mean Dynamic exposure: 48.5% (Random Forest), 47.6% (Logistic Regression).

Individual-equity extension, Random Forest:
- AAPL: 52.0% accuracy, F1 0.579; B&H CAGR 19.3%, Dynamic CAGR 8.3%; max DD -33.4% vs -16.1%.
- MSFT: 56.0% accuracy, F1 0.660; B&H CAGR 19.6%, Dynamic CAGR 6.1%; max DD -34.5% vs -18.1%.
- NVDA: 48.9% accuracy, F1 0.559; B&H CAGR 102.9%, Dynamic CAGR 47.9%; max DD -36.9% vs -21.7%.

Interpretation: the combined experiment does not support a persistent market-beating return edge. The consistent result is drawdown reduction produced alongside lower average market exposure, which creates substantial opportunity cost during strong trends.
