# Financial Advisor Bot — 15-Asset Verified Research Build

Financial Advisor Bot is an explainable financial decision-support prototype for the CM3070 final project. It combines:

- a five-trading-day Random Forest / Logistic Regression classifier;
- a long-horizon trend, momentum and volatility regime engine;
- a bounded ML confirmation layer for portfolio exposure;
- five-fold purged expanding-window walk-forward validation;
- next-day execution and transaction-cost-aware backtesting;
- a **12-ETF core research panel**;
- a formal **AAPL, MSFT and NVDA individual-equity robustness extension**; and
- a Streamlit interface for non-technical users.

> Academic prototype only. Not financial advice. Historical backtests do not guarantee future performance.

## Formal research universe

The project now contains **15 formal assets** and evaluates two models, giving **30 requested asset/model runs** when the complete research snapshot has been frozen.

### ETF core panel

| Ticker | Research role |
|---|---|
| SPY | US broad equity |
| QQQ | US growth equity |
| IWM | US small-cap equity |
| EFA | Developed ex-US equity |
| EEM | Emerging-market equity |
| VNQ | US real estate |
| GLD | Gold |
| DBC | Broad commodities |
| IEF | Intermediate Treasuries |
| TLT | Long Treasuries |
| LQD | Investment-grade credit |
| HYG | High-yield credit |

### Individual-equity robustness extension

| Ticker | Company | Research purpose |
|---|---|---|
| AAPL | Apple Inc. | Liquid mega-cap technology / consumer hardware |
| MSFT | Microsoft Corp. | Software and cloud mega-cap technology |
| NVDA | NVIDIA Corp. | High-growth semiconductor / AI computing stress test |

The equities are deliberately reported as a separate extension rather than treated as equivalent to diversified ETFs. This reconnects the final empirical design with the liquid-stock examples in the preliminary project while preserving the broader cross-asset ETF replication panel.

## Data provenance

The ETF core uses the verified `manisahni/marketdata` adjusted-close snapshot. AAPL, MSFT and NVDA are downloaded from Yahoo Finance via yfinance, reduced to **Adjusted Close only**, clipped to the ETF panel's final date, frozen locally and hashed separately.

The first connected run creates:

```text
data/verified_daily_adjclose.csv
data/verified_daily_adjclose.provenance.json
data/verified_equity_adjclose.csv
data/verified_equity_adjclose.provenance.json
```

The two SHA-256 hashes are combined in the reproduction summary. This keeps the sources auditable without pretending that the ETF and stock data came from the same original file.

## Reproduce the 15-asset experiment

Python 3.10 is recommended.

```bash
pyenv local 3.10.13
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the **first connected run**, freeze both datasets and evaluate all 15 assets:

```bash
python reproduce_research.py --refresh-data
python generate_results_summary.py
```

Expected current-run outputs:

```text
results/multi_asset_evaluation.csv       # up to 30 rows = 15 assets × 2 models
results/reproduction_summary.json
results/model_summary.csv
results/asset_summary.csv
```

After the first successful run, omit `--refresh-data` to reuse the exact archived files and hashes:

```bash
python reproduce_research.py
```

## Important result-version note

The supplied `results/archived_12_etf_core/` directory contains the earlier completed **24-run ETF-only experiment**. It is retained for audit history only.

Once AAPL, MSFT and NVDA have been frozen on a connected machine, `python reproduce_research.py --refresh-data` regenerates the root `results/` files for the full 30-run experiment.

## Run tests

```bash
python -m pytest -q
```

Expected result in this package:

```text
21 passed
```

The test suite is offline and deterministic. Synthetic fixtures are used only for software correctness and are never treated as empirical market evidence.

## Run the interface

```bash
streamlit run app.py
```

The 15 formal research assets appear directly in the selector. Other symbols remain available through Custom ticker and are exploratory only unless separately frozen and documented.

## Research architecture

```text
12-ETF verified panel -----------+
                                 |
AAPL/MSFT/NVDA frozen extension -+--> common 15-asset research interface
                                      |
                                      +--> price features --> RF / Logistic Regression
                                      |                       |
                                      |                       +--> purged 5-day OOS probabilities
                                      |
                                      +--> trend + momentum + volatility regime
                                                               |
                                                               +--> monthly systematic risk budget
                                                                            |
5-day OOS probability -------------------------------------------------------+
                                                                            v
                                                                 FinAdvisor Dynamic
                                                                            |
                                                         common-period benchmark backtest
```

## Main files

```text
advisor_bot.py              walk-forward ML, recommendation and short-horizon backtest
strategy_engine.py          regime, monthly allocation and long-term backtest
data_source.py              ETF/equity freeze, integrity checks and provenance
asset_universe.py           12-ETF core + AAPL/MSFT/NVDA extension
batch_evaluate.py           15 assets × model batch runner
reproduce_research.py       one-command 30-run reproducibility workflow
generate_results_summary.py aggregate cross-asset tables
app.py                      Streamlit interface
tests/                      21 deterministic offline tests
DATA_PROVENANCE.md          two-source data/provenance protocol
```

## ORB research extension

Opening Range Breakout remains a literature-backed future intraday module. The formal 15-asset study intentionally uses adjusted daily closes for a consistent feature set. Even though the public individual-equity source files contain OHLCV, the formal experiment retains only adjusted close and does **not** use those extra fields to give the stocks an informational advantage over the ETF panel.
