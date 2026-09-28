import pandas as pd
import streamlit as st

from advisor_bot import full_advisor_run
from asset_universe import RESEARCH_ASSETS, asset_table_rows
from data_source import VERIFIED_DATA_URL
from strategy_engine import metrics_table

st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈", layout="wide")

st.title("📈 Financial Advisor Bot")
st.caption(
    "Verified-data financial decision-support prototype combining expanding-window machine learning "
    "with long-horizon trend, momentum, regime detection and portfolio backtesting. "
    "Academic demonstration only — not financial advice."
)

with st.sidebar:
    st.header("Advisor Settings")
    asset_choice = st.selectbox(
        "Verified research asset",
        RESEARCH_ASSETS + ["Custom ticker"],
        help="The formal research universe contains a 12-ETF core plus frozen AAPL, MSFT and NVDA adjusted-close extensions.",
    )
    ticker = (
        st.text_input("Custom yfinance ticker", value="").upper().strip()
        if asset_choice == "Custom ticker" else asset_choice
    )
    model_name = st.selectbox("ML model", ["Random Forest", "Logistic Regression"])
    start_date = st.text_input("Research start date", value="2018-01-01")
    transaction_cost_bps = st.slider("Transaction cost (bps per allocation turnover)", 0, 50, 10, 1)
    run = st.button("Run FinAdvisor", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Research data: verified 12-ETF panel plus independently frozen AAPL/MSFT/NVDA adjusted-close extension.")
    st.caption("Custom symbols beyond the 15 formal assets use yfinance separately and are not part of the fixed experiment.")

if not ticker:
    st.warning("Enter a ticker to begin.")
    st.stop()

if "has_run" not in st.session_state:
    st.session_state.has_run = True

if run or st.session_state.has_run:
    try:
        result = full_advisor_run(
            ticker=ticker,
            model_name=model_name,
            start=start_date,
            transaction_cost=transaction_cost_bps / 10_000,
        )
        advisor = result["advisor_result"]
        raw = result["raw_data"]
        model_metrics = result["model_metrics"]
        folds = result["walk_forward_folds"]
        backtest = result["backtest"]
        backtest_metrics = result["backtest_metrics"]
        long_term_backtest = result["long_term_backtest"]
        long_term_metrics = result["long_term_metrics"]
        long_term_advice = result["long_term_advice"]
    except Exception as exc:
        st.error(f"Could not run FinAdvisor: {exc}")
        st.info("For the formal research universe, connect once and run `python reproduce_research.py --refresh-data` to freeze both the ETF panel and AAPL/MSFT/NVDA extension.")
        st.stop()

    overview, long_term_tab, ml_tab, methods_tab = st.tabs(
        ["Advisor Overview", "Long-Term Strategy", "Walk-Forward ML", "Methodology & Limits"]
    )

    with overview:
        st.subheader(f"Current assessment for {advisor.ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("5-Day ML Signal", advisor.recommendation)
        c2.metric("Probability Up", f"{advisor.probability_up:.1%}")
        c3.metric("Market Regime", long_term_advice.regime)
        c4.metric("Dynamic Allocation", f"{long_term_advice.dynamic_allocation:.0%}")
        st.info(long_term_advice.summary)
        st.warning(advisor.risk_warning)
        st.markdown("### Adjusted Price History")
        st.line_chart(raw.set_index("Date")[["Adj Close"]])
        st.markdown("### Explanation")
        for item in advisor.explanation:
            st.write(f"- {item}")

    with long_term_tab:
        st.subheader("Walk-Forward Out-of-Sample Portfolio Comparison")
        st.caption(
            f"Evaluation period {model_metrics['test_start']} to {model_metrics['test_end']}. "
            f"ML confirmation probabilities are generated only from expanding-window future folds. "
            f"Transaction-cost assumption: {transaction_cost_bps} bps per unit of turnover."
        )
        equity = long_term_backtest.set_index("Date")[["buy_hold_equity", "systematic_equity", "dynamic_equity"]].rename(
            columns={"buy_hold_equity": "Buy & Hold", "systematic_equity": "Trend + Momentum", "dynamic_equity": "FinAdvisor Dynamic"}
        )
        st.line_chart(equity)
        table = metrics_table(long_term_metrics)
        display = pd.DataFrame(index=table.index)
        display["CAGR"] = table["cagr"].map(lambda x: f"{x:.2%}")
        display["Total Return"] = table["total_return"].map(lambda x: f"{x:.2%}")
        display["Volatility"] = table["annual_volatility"].map(lambda x: f"{x:.2%}")
        display["Sharpe"] = table["sharpe"].map(lambda x: f"{x:.2f}")
        display["Sortino"] = table["sortino"].map(lambda x: f"{x:.2f}")
        display["Max Drawdown"] = table["max_drawdown"].map(lambda x: f"{x:.2%}")
        display["Calmar"] = table["calmar"].map(lambda x: f"{x:.2f}")
        display["Turnover"] = table["turnover"].map(lambda x: f"{x:.2f}x")
        display["Average Exposure"] = table["average_exposure"].map(lambda x: f"{x:.0%}")
        st.dataframe(display, use_container_width=True)

        bnh, dyn = long_term_metrics["Buy & Hold"], long_term_metrics["FinAdvisor Dynamic"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Dynamic CAGR", f"{dyn['cagr']:.2%}", delta=f"{dyn['cagr'] - bnh['cagr']:+.2%} vs B&H")
        d2.metric("Dynamic Sharpe", f"{dyn['sharpe']:.2f}", delta=f"{dyn['sharpe'] - bnh['sharpe']:+.2f}")
        d3.metric("Dynamic Max DD", f"{dyn['max_drawdown']:.2%}", delta=f"{dyn['max_drawdown'] - bnh['max_drawdown']:+.2%}")
        d4.metric("Average Exposure", f"{dyn['average_exposure']:.0%}")

        st.markdown("### Allocation history")
        st.line_chart(long_term_backtest.set_index("Date")[["systematic_exposure", "dynamic_exposure"]].rename(
            columns={"systematic_exposure": "Trend + Momentum Target", "dynamic_exposure": "Dynamic Target"}
        ))

    with ml_tab:
        st.subheader("Expanding-Window Walk-Forward Evaluation")
        st.caption(
            "The earliest half of labelled observations forms the initial training sample. "
            "Five contiguous future folds are then predicted sequentially; each fold is trained only on data available before that fold."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{model_metrics['precision']:.3f}")
        m3.metric("Recall", f"{model_metrics['recall']:.3f}")
        m4.metric("F1", f"{model_metrics['f1']:.3f}")
        st.caption(
            f"{model_metrics['folds']} folds | {model_metrics['test_rows']} total OOS rows | "
            f"{model_metrics['test_start']} to {model_metrics['test_end']}"
        )
        st.dataframe(folds, use_container_width=True, hide_index=True)
        cm = pd.DataFrame(model_metrics["confusion_matrix"], index=["Actual Down/Flat", "Actual Up"], columns=["Pred Down/Flat", "Pred Up"])
        st.dataframe(cm, use_container_width=True)

        st.markdown("### ML long/cash diagnostic backtest")
        st.line_chart(backtest.set_index("Date")[["strategy_equity", "buy_hold_equity"]].rename(
            columns={"strategy_equity": "ML Strategy", "buy_hold_equity": "Buy & Hold"}
        ))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("ML Strategy Return", f"{backtest_metrics['strategy_total_return']:.1%}")
        b2.metric("Buy/Hold Return", f"{backtest_metrics['buy_hold_total_return']:.1%}")
        b3.metric("ML Sharpe", f"{backtest_metrics['strategy_sharpe']:.2f}")
        b4.metric("ML Max Drawdown", f"{backtest_metrics['strategy_max_drawdown']:.1%}")

    with methods_tab:
        st.subheader("Methodology")
        st.markdown(
            f"""
**Verified research data.** The primary 12-ETF panel uses dividend/split-adjusted daily closes from a static public market-data mirror whose provenance is a yfinance download cache and whose repository publishes integrity checks. AAPL, MSFT and NVDA form a separate individual-equity robustness extension sourced from public yfinance-history mirrors, clipped to the ETF snapshot end date, frozen locally and hashed independently. The two hashes are combined for the formal 15-asset experiment. ETF source: `{VERIFIED_DATA_URL}`.

**Price-only ML features.** Because the verified source intentionally contains adjusted closes rather than fabricated OHLCV fields, the classifier uses nine price-derived predictors: 1-, 5-, 20- and 63-day returns; 20/50 and 50/200 moving-average ratios; RSI; 20-day volatility; and drawdown.

**Walk-forward validation.** The model is not evaluated with a single random split. An expanding-window protocol trains on the past and predicts five sequential future folds, producing one concatenated out-of-sample prediction series.

**Long-horizon allocation.** Price relative to the 200-day average, 50/200 trend, six- and twelve-month momentum and volatility determine a monthly long/cash risk budget. The ML probability can only reduce or confirm this allocation.

**Timing.** Portfolio targets are shifted by one trading day before returns and transaction costs are applied. All benchmark and active equity curves restart from the same $1 capital base at the beginning of the walk-forward evaluation period.
            """
        )
        st.markdown("### Fixed research universe")
        st.dataframe(pd.DataFrame(asset_table_rows()), use_container_width=True, hide_index=True)
        st.markdown("### Limits")
        st.markdown(
            """
- The formal experiment uses adjusted closes only. The individual-equity source files contain OHLCV, but FinAdvisor intentionally retains only adjusted close so all 15 assets use the same price-only feature set; ORB still requires a separately designed intraday dataset.
- The strategy still uses fixed hand-designed trend/regime thresholds and simplified transaction costs.
- Taxes, spreads, slippage and market impact are not fully modelled.
- The system evaluates generic market signals and does not perform personal suitability assessment.
- Historical results do not guarantee future performance.
            """
        )

    st.markdown("---")
    st.caption("Financial Advisor Bot academic prototype — historical simulation, not financial advice.")
