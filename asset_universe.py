"""Curated FinAdvisor research universe.

The primary replication panel contains 12 ETFs from one verified adjusted-close
source. A separate individual-equity robustness extension adds AAPL, MSFT and
NVDA. The equity extension is frozen and hashed independently so the ETF panel
and stock panel keep distinct provenance while sharing the same modelling and
backtesting protocol.
"""
from __future__ import annotations

from typing import Dict, List

ETF_ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "group": "US broad equity", "panel": "ETF core"},
    "QQQ": {"name": "Invesco QQQ Trust", "group": "US growth equity", "panel": "ETF core"},
    "IWM": {"name": "iShares Russell 2000 ETF", "group": "US small-cap equity", "panel": "ETF core"},
    "EFA": {"name": "iShares MSCI EAFE ETF", "group": "Developed ex-US equity", "panel": "ETF core"},
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "group": "Emerging-market equity", "panel": "ETF core"},
    "VNQ": {"name": "Vanguard Real Estate ETF", "group": "US real estate", "panel": "ETF core"},
    "GLD": {"name": "SPDR Gold Shares", "group": "Gold", "panel": "ETF core"},
    "DBC": {"name": "Invesco DB Commodity Index Tracking Fund", "group": "Broad commodities", "panel": "ETF core"},
    "IEF": {"name": "iShares 7-10 Year Treasury Bond ETF", "group": "Intermediate Treasuries", "panel": "ETF core"},
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "group": "Long Treasuries", "panel": "ETF core"},
    "LQD": {"name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "group": "Investment-grade credit", "panel": "ETF core"},
    "HYG": {"name": "iShares iBoxx $ High Yield Corporate Bond ETF", "group": "High-yield credit", "panel": "ETF core"},
}

EQUITY_ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.", "group": "Technology / consumer hardware", "panel": "Individual-equity extension"},
    "MSFT": {"name": "Microsoft Corp.", "group": "Technology / software & cloud", "panel": "Individual-equity extension"},
    "NVDA": {"name": "NVIDIA Corp.", "group": "Semiconductors / AI computing", "panel": "Individual-equity extension"},
}

ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {**ETF_ASSET_UNIVERSE, **EQUITY_ASSET_UNIVERSE}
ETF_RESEARCH_ASSETS: List[str] = list(ETF_ASSET_UNIVERSE)
EQUITY_EXTENSION_ASSETS: List[str] = list(EQUITY_ASSET_UNIVERSE)
RESEARCH_ASSETS: List[str] = ETF_RESEARCH_ASSETS + EQUITY_EXTENSION_ASSETS


def asset_table_rows():
    return [
        {
            "Ticker": ticker,
            "Asset": meta["name"],
            "Research role": meta["group"],
            "Panel": meta["panel"],
        }
        for ticker, meta in ASSET_UNIVERSE.items()
    ]
