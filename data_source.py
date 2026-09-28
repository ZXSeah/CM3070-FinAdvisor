"""Verified market-data access and provenance for FinAdvisor.

Primary ETF research source
---------------------------
A static public mirror of dividend/split-adjusted ETF closes maintained at:
https://github.com/manisahni/marketdata

Individual-equity robustness extension
--------------------------------------
AAPL, MSFT and NVDA are downloaded directly from Yahoo Finance through
``yfinance`` with ``auto_adjust=False`` so that the explicit ``Adj Close`` field
is retained. The first successful FinAdvisor retrieval is clipped to the ETF
panel's calendar coverage, then cached and hashed locally. This prevents later
provider updates from changing an already archived experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from asset_universe import ETF_RESEARCH_ASSETS, EQUITY_EXTENSION_ASSETS, RESEARCH_ASSETS

VERIFIED_DATA_URL = "https://raw.githubusercontent.com/manisahni/marketdata/main/daily_adjclose.csv"
EQUITY_SOURCE_URLS: Dict[str, str] = {
    "AAPL": "https://finance.yahoo.com/quote/AAPL/history/",
    "MSFT": "https://finance.yahoo.com/quote/MSFT/history/",
    "NVDA": "https://finance.yahoo.com/quote/NVDA/history/",
}

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_DIR / "data" / "verified_daily_adjclose.csv"
DEFAULT_PROVENANCE = PROJECT_DIR / "data" / "verified_daily_adjclose.provenance.json"
DEFAULT_EQUITY_CACHE = PROJECT_DIR / "data" / "verified_equity_adjclose.csv"
DEFAULT_EQUITY_PROVENANCE = PROJECT_DIR / "data" / "verified_equity_adjclose.provenance.json"


@dataclass(frozen=True)
class DataProvenance:
    source: str
    source_url: str
    cache_path: str
    sha256: str
    first_date: str
    last_date: str
    rows: int
    columns: int
    integrity_checks: str


@dataclass(frozen=True)
class EquityProvenance:
    source: str
    source_urls: Dict[str, str]
    cache_path: str
    sha256: str
    first_date: str
    last_date: str
    rows: int
    columns: int
    assets: list[str]
    integrity_checks: str


@dataclass(frozen=True)
class ResearchProvenance:
    etf_sha256: str
    equity_sha256: str
    combined_sha256: str
    first_date: str
    last_date: str
    assets: list[str]


def _hash_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined_hash(*parts: str) -> str:
    h = sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def validate_verified_panel(panel: pd.DataFrame, strict_ground_truth: bool = True) -> None:
    """Validate the primary ETF panel's schema and published market-fact checks."""
    if "Date" not in panel.columns:
        raise ValueError("Verified panel is missing Date.")
    missing = [ticker for ticker in ETF_RESEARCH_ASSETS if ticker not in panel.columns]
    if missing:
        raise ValueError(f"Verified panel missing ETF research assets: {missing}")

    dates = pd.to_datetime(panel["Date"])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Date index is invalid, duplicated or unsorted.")

    for ticker in ETF_RESEARCH_ASSETS:
        s = pd.to_numeric(panel[ticker], errors="coerce").dropna()
        if len(s) < 500:
            raise ValueError(f"Insufficient history for {ticker}: {len(s)} rows")
        if (s <= 0).any() or not np.isfinite(s).all():
            raise ValueError(f"Invalid adjusted-close values detected for {ticker}.")

    if strict_ground_truth:
        indexed = panel.copy()
        indexed["Date"] = dates
        indexed = indexed.set_index("Date")
        if "^VIX" not in indexed.columns:
            raise ValueError("Ground-truth VIX column is missing from verified panel.")
        checks = [
            ("2020-03-16", "^VIX", 82.69),
            ("2018-02-05", "^VIX", 37.32),
            ("2008-11-20", "^VIX", 80.86),
        ]
        for date, col, expected in checks:
            if pd.Timestamp(date) not in indexed.index:
                raise ValueError(f"Integrity-check date {date} is missing.")
            actual = float(indexed.loc[pd.Timestamp(date), col])
            if round(actual, 2) != expected:
                raise ValueError(f"Integrity check failed for {col} on {date}: {actual} != {expected}")
        spy_return = pd.to_numeric(indexed["SPY"], errors="coerce").pct_change(fill_method=None)
        if spy_return.idxmin().date().isoformat() != "2020-03-16":
            raise ValueError("SPY worst-day integrity check failed.")
        covid_dd = float(
            indexed.loc[pd.Timestamp("2020-03-23"), "SPY"]
            / indexed.loc[pd.Timestamp("2020-02-19"), "SPY"]
            - 1.0
        )
        if round(covid_dd, 3) != -0.337:
            raise ValueError("SPY COVID drawdown integrity check failed.")
        if pd.Timestamp("2011-09-05") in indexed.index:
            raise ValueError("NYSE holiday appears in verified daily panel.")
        rows_2023 = int((indexed.index.year == 2023).sum())
        if not (250 <= rows_2023 <= 253):
            raise ValueError(f"Unexpected 2023 trading-day count: {rows_2023}")


def validate_equity_panel(
    panel: pd.DataFrame,
    required_end: Optional[pd.Timestamp] = None,
) -> None:
    """Validate the frozen AAPL/MSFT/NVDA adjusted-close extension.

    The check intentionally focuses on structural integrity and split adjustment.
    A large split artefact would create an implausibly large one-day adjusted-close
    return, so absolute daily returns above 50% stop the formal research run.
    """
    if "Date" not in panel.columns:
        raise ValueError("Equity panel is missing Date.")
    missing = [ticker for ticker in EQUITY_EXTENSION_ASSETS if ticker not in panel.columns]
    if missing:
        raise ValueError(f"Equity panel missing research assets: {missing}")

    dates = pd.to_datetime(panel["Date"])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Equity Date index is invalid, duplicated or unsorted.")

    indexed = panel.copy()
    indexed["Date"] = dates
    indexed = indexed.set_index("Date")
    for ticker in EQUITY_EXTENSION_ASSETS:
        s = pd.to_numeric(indexed[ticker], errors="coerce").dropna()
        if len(s) < 500:
            raise ValueError(f"Insufficient equity history for {ticker}: {len(s)} rows")
        if (s <= 0).any() or not np.isfinite(s).all():
            raise ValueError(f"Invalid adjusted-close values detected for {ticker}.")
        if s.pct_change(fill_method=None).abs().max() > 0.50:
            raise ValueError(f"Possible unadjusted split/corrupt daily return detected for {ticker}.")
        if required_end is not None:
            last = s.index.max()
            if last < required_end - pd.Timedelta(days=10):
                raise ValueError(f"{ticker} does not cover the ETF research end date closely enough: {last.date()}")


def load_verified_panel(
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, DataProvenance]:
    """Load the verified ETF panel from cache or its public source."""
    cache = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)

    if refresh or not cache.exists():
        try:
            panel = pd.read_csv(VERIFIED_DATA_URL)
        except Exception as exc:
            raise FileNotFoundError(
                "Verified ETF research data are not cached and could not be downloaded. "
                "Connect to the internet and run `python reproduce_research.py --refresh-data`."
            ) from exc
        panel.to_csv(cache, index=False)
    else:
        panel = pd.read_csv(cache)

    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values("Date").reset_index(drop=True)
    validate_verified_panel(panel, strict_ground_truth=True)

    provenance = DataProvenance(
        source="manisahni/marketdata verified ETF adjusted-close mirror (yfinance cache)",
        source_url=VERIFIED_DATA_URL,
        cache_path=_relative_or_absolute(cache),
        sha256=_hash_file(cache),
        first_date=str(panel["Date"].min().date()),
        last_date=str(panel["Date"].max().date()),
        rows=int(len(panel)),
        columns=int(len(panel.columns)),
        integrity_checks="VIX published checks; SPY worst day/COVID drawdown; 2023 trading-day count; NYSE holiday exclusion",
    )
    return panel, provenance


def _extract_equity_adjusted_close(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize one public stock-history CSV to Date + adjusted close."""
    df = raw.copy()
    date_candidates = [c for c in df.columns if str(c).strip().lower() in {"date", "datetime", "timestamp"}]
    if not date_candidates:
        raise ValueError(f"Could not identify a date column in {ticker} source data.")
    date_col = date_candidates[0]

    normalized = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    adj_col = None
    for key in ("adj close", "adjusted close", "adjclose"):
        if key in normalized:
            adj_col = normalized[key]
            break
    if adj_col is None:
        raise ValueError(
            f"{ticker} source does not contain an adjusted-close field. "
            "Formal research refuses to substitute raw Close because stock splits/dividends would distort returns."
        )

    out = df[[date_col, adj_col]].copy()
    out.columns = ["Date", ticker]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    out[ticker] = pd.to_numeric(out[ticker], errors="coerce")
    return out.dropna().sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def _download_yfinance_adjusted_close(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download Date + Adjusted Close directly through yfinance.

    ``yf.download`` treats ``end`` as exclusive, so one calendar day is added
    by the caller. The helper supports both the flat and MultiIndex column
    layouts returned by different yfinance versions.
    """
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError(
            "yfinance is required for the AAPL/MSFT/NVDA research extension. "
            "Install project requirements with `python -m pip install -r requirements.txt`."
        ) from exc

    raw = yf.download(
        ticker,
        start=pd.Timestamp(start).date().isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        candidates = [
            c for c in raw.columns
            if str(c[0]).strip().lower().replace("_", " ") in {"adj close", "adjusted close", "adjclose"}
        ]
        if not candidates:
            raise ValueError(f"yfinance did not return Adj Close for {ticker}.")
        series = raw[candidates[0]]
    else:
        normalized = {str(c).strip().lower().replace("_", " "): c for c in raw.columns}
        adj_col = next(
            (normalized[k] for k in ("adj close", "adjusted close", "adjclose") if k in normalized),
            None,
        )
        if adj_col is None:
            raise ValueError(f"yfinance did not return Adj Close for {ticker}.")
        series = raw[adj_col]

    out = pd.DataFrame({
        "Date": pd.to_datetime(raw.index, errors="coerce"),
        ticker: pd.to_numeric(series, errors="coerce").to_numpy(),
    })
    out["Date"] = out["Date"].dt.tz_localize(None)
    return out.dropna().sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def load_equity_extension(
    cache_path: Optional[Path] = None,
    refresh: bool = False,
    primary_panel: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, EquityProvenance]:
    """Load/freeze AAPL, MSFT and NVDA adjusted closes.

    On the first connected run the three public yfinance-history mirrors are
    downloaded, normalized to adjusted close, clipped to the ETF snapshot's last
    date, merged into one wide CSV, validated and hashed. Later runs reuse the
    exact cached extension unless ``refresh=True`` is explicitly requested.
    """
    cache = Path(cache_path) if cache_path is not None else DEFAULT_EQUITY_CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)

    if primary_panel is None:
        primary_panel, _ = load_verified_panel()
    primary_dates = pd.to_datetime(primary_panel["Date"])
    research_start = primary_dates.min()
    research_end = primary_dates.max()

    if refresh or not cache.exists():
        merged: Optional[pd.DataFrame] = None
        errors: list[str] = []
        for ticker in EQUITY_EXTENSION_ASSETS:
            try:
                one = _download_yfinance_adjusted_close(ticker, research_start, research_end)
                # Keep the extension on exactly the same calendar coverage as the
                # primary ETF research panel. Missing non-trading dates are fine;
                # formal evaluation later uses available trading observations.
                one = one[(one["Date"] >= research_start) & (one["Date"] <= research_end)].copy()
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
                continue
            merged = one if merged is None else merged.merge(one, on="Date", how="outer")

        if errors or merged is None:
            detail = "; ".join(errors) if errors else "no data returned"
            raise FileNotFoundError(
                "The individual-equity research extension could not be frozen. "
                f"Details: {detail}. Connect to the internet and rerun `python reproduce_research.py --refresh-data`."
            )
        panel = merged.sort_values("Date").reset_index(drop=True)
        panel.to_csv(cache, index=False)
    else:
        panel = pd.read_csv(cache)
        panel["Date"] = pd.to_datetime(panel["Date"])
        panel = panel[(panel["Date"] >= research_start) & (panel["Date"] <= research_end)].sort_values("Date").reset_index(drop=True)

    validate_equity_panel(panel, required_end=research_end)
    provenance = EquityProvenance(
        source="AAPL/MSFT/NVDA Yahoo Finance adjusted closes downloaded through yfinance and frozen by FinAdvisor",
        source_urls=EQUITY_SOURCE_URLS,
        cache_path=_relative_or_absolute(cache),
        sha256=_hash_file(cache),
        first_date=str(pd.to_datetime(panel["Date"]).min().date()),
        last_date=str(pd.to_datetime(panel["Date"]).max().date()),
        rows=int(len(panel)),
        columns=int(len(panel.columns)),
        assets=EQUITY_EXTENSION_ASSETS,
        integrity_checks="clipped to ETF panel coverage; sorted unique dates; positive finite adjusted closes; >=500 observations; no >50% one-day split artefacts; coverage near ETF end date",
    )
    return panel, provenance


def merge_research_panels(etf_panel: pd.DataFrame, equity_panel: pd.DataFrame) -> pd.DataFrame:
    """Merge the independently verified panels while retaining their source provenance."""
    left = etf_panel.copy()
    right = equity_panel.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    merged = left.merge(right, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    return merged


def load_research_panel(
    refresh_etf: bool = False,
    refresh_equities: bool = False,
) -> Tuple[pd.DataFrame, ResearchProvenance, DataProvenance, EquityProvenance]:
    etf_panel, etf_prov = load_verified_panel(refresh=refresh_etf)
    equity_panel, eq_prov = load_equity_extension(refresh=refresh_equities, primary_panel=etf_panel)
    panel = merge_research_panels(etf_panel, equity_panel)
    combined = ResearchProvenance(
        etf_sha256=etf_prov.sha256,
        equity_sha256=eq_prov.sha256,
        combined_sha256=_combined_hash(etf_prov.sha256, eq_prov.sha256),
        first_date=str(pd.to_datetime(panel["Date"]).min().date()),
        last_date=str(pd.to_datetime(etf_panel["Date"]).max().date()),
        assets=RESEARCH_ASSETS,
    )
    return panel, combined, etf_prov, eq_prov


def save_provenance(provenance, path: Optional[Path] = None) -> Path:
    if isinstance(provenance, EquityProvenance):
        target = Path(path) if path is not None else DEFAULT_EQUITY_PROVENANCE
    else:
        target = Path(path) if path is not None else DEFAULT_PROVENANCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(provenance), indent=2), encoding="utf-8")
    return target


def research_ticker_frame(
    ticker: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, object]:
    ticker = ticker.upper().strip()
    if ticker not in RESEARCH_ASSETS:
        raise KeyError(f"{ticker} is not in the fixed formal research universe.")

    if ticker in ETF_RESEARCH_ASSETS:
        panel, provenance = load_verified_panel()
    else:
        primary, _ = load_verified_panel()
        panel, provenance = load_equity_extension(primary_panel=primary)

    frame = panel[["Date", ticker]].rename(columns={ticker: "Adj Close"}).dropna().copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame[frame["Date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["Date"] < pd.Timestamp(end)]
    frame["Close"] = frame["Adj Close"]
    return frame.reset_index(drop=True), provenance
