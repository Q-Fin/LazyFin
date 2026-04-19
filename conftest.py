"""
conftest.py
===========
Shared pytest fixtures for the aleph_toolkit test suite.

All fixtures are session-scoped: synthetic data is generated once and reused
across the entire test session, keeping the suite fast (< 15 s total).

No network calls are made.  All data is generated from a fixed NumPy RNG seed.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def price_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=600, freq="B")


@pytest.fixture(scope="session")
def prices(rng: np.random.Generator, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """3-asset adjusted-close price DataFrame (600 business days)."""
    return pd.DataFrame(
        {
            "AAPL":  150  * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 600))),
            "MSFT":  300  * np.exp(np.cumsum(rng.normal(0.0006, 0.011, 600))),
            "GOOGL": 2800 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, 600))),
        },
        index=price_index,
    )


@pytest.fixture(scope="session")
def ohlcv(rng: np.random.Generator, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Single-ticker OHLCV DataFrame (600 business days)."""
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 600))),
        index=price_index,
        name="Close",
    )
    return pd.DataFrame(
        {
            "Open":   close * 0.999,
            "High":   close * 1.010,
            "Low":    close * 0.990,
            "Close":  close,
            "Volume": pd.Series(
                rng.integers(1_000_000, 10_000_000, 600).astype(float),
                index=price_index,
            ),
        }
    )


@pytest.fixture(scope="session")
def ff3_factors(price_index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic Fama-French 3-factor daily returns (first 550 dates)."""
    idx = price_index[:550]
    df = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.009, 550),
            "SMB":    rng.normal(0.0001, 0.004, 550),
            "HML":    rng.normal(0.0001, 0.004, 550),
            "RF":     np.full(550, 0.00015),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def portfolio(prices: pd.DataFrame) -> dict:
    """Equal-weighted 3-asset portfolio over 2020-01-01 → 2022-12-31."""
    from aleph_toolkit.preprocessing import build_portfolio_returns, equal_weights
    w = equal_weights(3)
    p = build_portfolio_returns(
        prices,
        w,
        datetime.date(2020, 1, 1),
        datetime.date(2022, 12, 31),
    )
    p["alpha"] = 0.99
    return p


@pytest.fixture(scope="session")
def port_rets(portfolio: dict) -> pd.Series:
    return portfolio["port_rets"]


@pytest.fixture(scope="session")
def log_rets(portfolio: dict) -> pd.DataFrame:
    return portfolio["log_returns"]


@pytest.fixture(scope="session")
def weights(portfolio: dict) -> np.ndarray:
    return portfolio["weights"]
