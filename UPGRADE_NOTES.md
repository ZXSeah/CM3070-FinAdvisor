# Upgrade Notes — 15-Asset Equity Extension

Changes from the previous verified v3 build:

- Added AAPL, MSFT and NVDA as a formal individual-equity robustness extension.
- Preserved the original 12 ETFs as the primary cross-asset core panel.
- Added a second frozen data cache and provenance JSON for the three stocks.
- Added strict adjusted-close detection rather than silently substituting raw Close.
- Equity data are clipped to the ETF snapshot end date for a comparable research horizon.
- Added split/corruption guard checks and end-date coverage validation.
- Combined the two source hashes in the reproduction summary.
- Updated the Streamlit selector from 12 to 15 formal research assets.
- Updated the batch experiment from 24 to 30 requested runs.
- Expanded the deterministic software suite to 21 passing tests.
- Archived the old 24-run ETF-only outputs so they cannot be mistaken for the new 15-asset result.
