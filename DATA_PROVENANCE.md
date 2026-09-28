# Data Provenance and Integrity Protocol

## 1. Primary ETF research source

The 12-ETF core panel is downloaded from the public `manisahni/marketdata` static adjusted-close mirror:

`https://raw.githubusercontent.com/manisahni/marketdata/main/daily_adjclose.csv`

The upstream repository documents the file as a static mirror of dividend/split-adjusted closes exported from a yfinance cache and publishes market-fact integrity checks.

FinAdvisor stores:

- `data/verified_daily_adjclose.csv`
- `data/verified_daily_adjclose.provenance.json`

The provenance file records the URL, date range, row/column count and SHA-256 hash.

## 2. Individual-equity robustness extension

The formal extension contains **AAPL, MSFT and NVDA**. The public source repositories are maintained from yfinance history and their updater uses `yf.download(..., auto_adjust=False)`, preserving an `Adj Close` field.

FinAdvisor source URLs are defined in `data_source.py` and point to:

- Apple / AAPL — `kalilurrahman/APPLEStockdata`
- Microsoft / MSFT — `kalilurrahman/MSFTStockdata`
- NVIDIA / NVDA — `kalilurrahman/NVidiaStockdata`

On the first connected reproduction run, FinAdvisor:

1. downloads each stock-history CSV;
2. refuses the file if a genuine adjusted-close field cannot be identified;
3. retains only Date + Adjusted Close;
4. clips all three series to the final date of the frozen ETF snapshot;
5. merges them into `data/verified_equity_adjclose.csv`;
6. validates structure and price integrity; and
7. records a separate SHA-256 in `data/verified_equity_adjclose.provenance.json`.

The upstream repositories may continue to update, but an already archived FinAdvisor experiment does not change unless `--refresh-data` is explicitly used.

## 3. Equity integrity checks

`validate_equity_panel()` requires:

- Date exists, is sorted and unique;
- AAPL, MSFT and NVDA are present;
- each series contains at least 500 positive finite observations;
- no one-day adjusted-close move exceeds 50%, which provides a guard against obvious unadjusted split artefacts or corruption; and
- each equity reaches within 10 calendar days of the ETF research end date.

These checks do not prove every quote is perfect. They are intended to prevent common data-pipeline errors and to make failures explicit rather than silently substituting data.

## 4. Why the sources remain separate

The ETF and individual-equity datasets are not described as one original source. They are frozen and hashed independently, then merged only at the research interface. The reproduction summary records:

- ETF SHA-256;
- equity-extension SHA-256; and
- a combined research SHA-256 derived from those two hashes.

This makes it possible to reproduce the exact 15-asset experiment while retaining honest source provenance.

## 5. Consistent feature scope

The stock source files contain OHLCV, but the ETF source provides adjusted closes only. To preserve a fair common feature set across all 15 assets, the formal experiment intentionally retains **adjusted close only** for AAPL, MSFT and NVDA.

Consequently:

- price-derived features are common to every asset;
- stocks do not receive extra volume/OHLC information unavailable to the ETF panel;
- ORB is still not backtested; and
- a future intraday study needs a separately specified, timestamped OHLCV protocol.

## 6. Reproduction command

On a connected machine:

```bash
python reproduce_research.py --refresh-data
python generate_results_summary.py
```

Archive both data CSVs, both provenance JSON files, `results/multi_asset_evaluation.csv`, and `results/reproduction_summary.json` together before quoting 15-asset results in the report.
