from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic synthetic prices used only for software unit tests."""
    n = 900
    dates = pd.bdate_range("2018-01-02", periods=n)
    t = np.arange(n, dtype=float)
    drift = np.where((t // 150) % 2 == 0, 0.0008, -0.00045)
    cyc = 0.006 * np.sin(t / 9.0) + 0.003 * np.sin(t / 31.0)
    log_price = np.log(100.0) + np.cumsum(drift + cyc)
    price = np.exp(log_price)
    return pd.DataFrame({"Date": dates, "Adj Close": price, "Close": price})


@pytest.fixture
def synthetic_verified_panel() -> pd.DataFrame:
    from asset_universe import ETF_RESEARCH_ASSETS

    n = 650
    dates = pd.bdate_range("2015-01-02", periods=n)
    t = np.arange(n, dtype=float)
    panel = pd.DataFrame({"Date": dates})
    for i, ticker in enumerate(ETF_RESEARCH_ASSETS):
        panel[ticker] = 50 + i * 5 + 0.03 * t + 2 * np.sin(t / (20 + i))
    return panel


@pytest.fixture
def synthetic_equity_panel() -> pd.DataFrame:
    from asset_universe import EQUITY_EXTENSION_ASSETS

    n = 650
    dates = pd.bdate_range("2015-01-02", periods=n)
    t = np.arange(n, dtype=float)
    panel = pd.DataFrame({"Date": dates})
    for i, ticker in enumerate(EQUITY_EXTENSION_ASSETS):
        # Smooth positive series with no split-like discontinuities.
        panel[ticker] = 80 + i * 20 + 0.04 * t + 1.5 * np.sin(t / (23 + i))
    return panel
