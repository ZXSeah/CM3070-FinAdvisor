from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_universe import (
    ASSET_UNIVERSE,
    ETF_RESEARCH_ASSETS,
    EQUITY_EXTENSION_ASSETS,
    RESEARCH_ASSETS,
)
from data_source import validate_verified_panel, validate_equity_panel, merge_research_panels


def test_research_universe_has_15_unique_assets_with_three_equities():
    assert len(RESEARCH_ASSETS) == 15
    assert len(set(RESEARCH_ASSETS)) == 15
    assert len(ETF_RESEARCH_ASSETS) == 12
    assert EQUITY_EXTENSION_ASSETS == ["AAPL", "MSFT", "NVDA"]
    assert set(ETF_RESEARCH_ASSETS) == {"SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "IEF", "TLT", "LQD", "HYG"}
    assert set(RESEARCH_ASSETS) == set(ASSET_UNIVERSE)


def test_verified_panel_accepts_valid_fixture(synthetic_verified_panel):
    validate_verified_panel(synthetic_verified_panel, strict_ground_truth=False)


def test_verified_panel_rejects_duplicate_dates(synthetic_verified_panel):
    bad = pd.concat([synthetic_verified_panel, synthetic_verified_panel.iloc[[-1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated|unsorted"):
        validate_verified_panel(bad, strict_ground_truth=False)


def test_verified_panel_rejects_nonpositive_prices(synthetic_verified_panel):
    bad = synthetic_verified_panel.copy()
    bad.loc[100, "SPY"] = 0.0
    with pytest.raises(ValueError, match="Invalid adjusted-close"):
        validate_verified_panel(bad, strict_ground_truth=False)


def test_equity_panel_accepts_valid_fixture(synthetic_equity_panel):
    validate_equity_panel(synthetic_equity_panel, required_end=synthetic_equity_panel["Date"].max())


def test_equity_panel_rejects_split_like_jump(synthetic_equity_panel):
    bad = synthetic_equity_panel.copy()
    bad.loc[200, "NVDA"] = bad.loc[199, "NVDA"] * 2.0
    with pytest.raises(ValueError, match="split|corrupt"):
        validate_equity_panel(bad)


def test_research_panel_merge_contains_core_and_equities(synthetic_verified_panel, synthetic_equity_panel):
    merged = merge_research_panels(synthetic_verified_panel, synthetic_equity_panel)
    assert set(RESEARCH_ASSETS).issubset(merged.columns)
    assert merged["Date"].is_monotonic_increasing


def test_strict_integrity_checks_can_detect_published_ground_truth():
    dates = pd.bdate_range("2008-01-02", "2024-01-05")
    dates = dates[dates != pd.Timestamp("2011-09-05")]
    holidays_2023 = pd.to_datetime([
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    ])
    dates = dates[~dates.isin(holidays_2023)]
    panel = pd.DataFrame({"Date": dates})
    t = np.arange(len(dates), dtype=float)
    for i, ticker in enumerate(ETF_RESEARCH_ASSETS):
        panel[ticker] = 100 + i + 0.02 * t
    panel["^VIX"] = 20.0

    idx = panel.set_index("Date")
    idx.loc[pd.Timestamp("2008-11-20"), "^VIX"] = 80.86
    idx.loc[pd.Timestamp("2018-02-05"), "^VIX"] = 37.32
    idx.loc[pd.Timestamp("2020-03-16"), "^VIX"] = 82.69

    prior = idx.index[idx.index.get_loc(pd.Timestamp("2020-03-16")) - 1]
    idx.loc[prior, "SPY"] = 200.0
    idx.loc[pd.Timestamp("2020-03-16"), "SPY"] = 100.0
    after = idx.index[idx.index.get_loc(pd.Timestamp("2020-03-16")) + 1]
    idx.loc[after, "SPY"] = 101.0

    idx.loc[pd.Timestamp("2020-02-19"), "SPY"] = 150.0
    idx.loc[pd.Timestamp("2020-03-23"), "SPY"] = 150.0 * (1 - 0.337)
    panel = idx.reset_index()

    validate_verified_panel(panel, strict_ground_truth=True)


def test_equity_source_normalizer_requires_adjusted_close():
    from data_source import _extract_equity_adjusted_close
    dates = pd.bdate_range("2024-01-02", periods=3)
    good = pd.DataFrame({"Date": dates, "Close": [100, 101, 102], "Adj Close": [99, 100, 101]})
    out = _extract_equity_adjusted_close(good, "AAPL")
    assert list(out.columns) == ["Date", "AAPL"]
    assert out["AAPL"].tolist() == [99, 100, 101]

    bad = pd.DataFrame({"Date": dates, "Close": [100, 101, 102]})
    with pytest.raises(ValueError, match="adjusted-close"):
        _extract_equity_adjusted_close(bad, "AAPL")
