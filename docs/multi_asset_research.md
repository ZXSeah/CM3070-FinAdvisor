# Multi-Asset Research Protocol

The final research universe has two deliberately separated components:

- **Core replication panel:** 12 heterogeneous ETFs from one verified adjusted-close source.
- **Individual-equity robustness panel:** AAPL, MSFT and NVDA, frozen and hashed separately from public yfinance-history mirrors.

The same modelling rules are applied to all 15 assets. The stocks do not receive OHLC or volume features even though those fields are available upstream; only adjusted close is retained so the feature set is consistent.

Protocol:

1. Freeze and validate both data sources.
2. Use the same research start date and the ETF panel's final date.
3. Build the same price-derived predictors for every asset.
4. Use the same five-day target and five-observation purge.
5. Run five expanding future folds.
6. Apply the same Random Forest and Logistic Regression settings.
7. Apply the same long-horizon trend/momentum/volatility allocation rules.
8. Preserve every losing and failed run.
9. Run **15 assets × 2 models = 30 requested experiments**.
10. Report the 12-ETF core and the three-stock extension separately as well as in the combined summary.

This design uses AAPL/MSFT/NVDA as a robustness extension rather than a search for high-performing stocks.
