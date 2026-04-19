"""
preprocessing.py — Pure data transformation functions.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "flatten_multiindex_columns", "rename_adj_close",
    "validate_ohlcv_columns", "clean_ohlcv",
    "compute_log_returns", "compute_single_log_returns",
    "parse_weights", "normalise_weights", "equal_weights",
    "build_portfolio_returns",
    "align_date_ranges", "resolve_date_range",
]


_MIN_OBS: int = 30


# ---------------------------------------------------------------------------
# Structural cleaning
# ---------------------------------------------------------------------------

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df.copy()


def rename_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    return df


def validate_ohlcv_columns(
    df: pd.DataFrame,
    required: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
) -> list[str]:
    return [c for c in required if c not in df.columns]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_multiindex_columns(df)
    df = rename_adj_close(df)
    missing = validate_ohlcv_columns(df)
    if missing:
        raise ValueError(f"OHLCV DataFrame is missing columns: {missing}")
    required = ("Open", "High", "Low", "Close", "Volume")
    return df[list(required)].dropna().sort_index().copy()


# ---------------------------------------------------------------------------
# Log-return computation
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.dropna(axis=1, how="all").dropna(how="all")
    if len(clean) < 2:
        raise ValueError(
            f"prices has only {len(clean)} row(s) after NaN removal; need ≥ 2."
        )
    return np.log(clean / clean.shift(1)).dropna()


def compute_single_log_returns(price_series: pd.Series) -> pd.Series:
    clean = price_series.dropna()
    if len(clean) < 2:
        raise ValueError(
            f"price_series has only {len(clean)} non-NaN observation(s); need ≥ 2."
        )
    return np.log(clean / clean.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

def parse_weights(raw: str, n_tickers: int) -> Optional[list[float]]:
    if not raw or not raw.strip():
        return None
    try:
        values = [float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None
    if len(values) != n_tickers:
        return None
    if any(v < 0 for v in values):
        return None
    if sum(values) == 0:
        return None
    return values


def normalise_weights(raw_weights: list[float]) -> np.ndarray:
    w = np.asarray(raw_weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    total = w.sum()
    if total == 0:
        raise ValueError("Weights sum to zero; cannot normalise.")
    return w / total


def equal_weights(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}.")
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Portfolio return construction
# ---------------------------------------------------------------------------

def build_portfolio_returns(
    prices: pd.DataFrame,
    weights: np.ndarray,
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict:
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date}).")

    # Filter to date range
    idx = prices.index
    mask = (idx.date >= start_date) & (idx.date <= end_date)
    sliced = prices.loc[mask].dropna(axis=1, how="all")

    if sliced.empty:
        raise ValueError(f"No price data in range {start_date} → {end_date}.")

    log_returns = compute_log_returns(sliced)

    if len(log_returns) < _MIN_OBS:
        raise ValueError(
            f"Only {len(log_returns)} trading day(s) in range "
            f"({start_date} → {end_date}). Minimum required: {_MIN_OBS}."
        )

    n = log_returns.shape[1]
    if weights.size != n:
        raise ValueError(
            f"weights length ({weights.size}) ≠ number of usable tickers ({n})."
        )

    port_rets = log_returns.dot(weights)
    port_rets.name = "portfolio"

    return {
        "prices":       sliced.loc[log_returns.index[0]:],
        "log_returns":  log_returns,
        "port_rets":    port_rets,
        "weights":      weights,
        "start_date":   log_returns.index[0].date(),
        "end_date":     log_returns.index[-1].date(),
        "tickers":      list(log_returns.columns),
    }


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def align_date_ranges(*series: pd.Series) -> tuple:
    if len(series) < 2:
        raise ValueError("At least 2 Series required.")
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    if common.empty:
        raise ValueError("No common dates found across all input Series.")
    return tuple(s.loc[common] for s in series)


def resolve_date_range(
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    *,
    fallback_years: int = 5,
) -> tuple[datetime.date, datetime.date]:
    today = datetime.date.today()

    # Normalise datetime → date
    if hasattr(end_date, "date"):
        end_date = end_date.date()
    if hasattr(start_date, "date"):
        start_date = start_date.date()

    if end_date is None:
        end_date = today
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365 * fallback_years)

    if start_date >= end_date:
        raise ValueError(
            f"start_date ({start_date}) must be strictly before end_date ({end_date})."
        )
    return start_date, end_date
