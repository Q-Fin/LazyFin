"""
cache.py — TTL-aware in-memory + partial-disk cache for market data.

Optimisations over the baseline
---------------------------------
* OHLCV superset lookup: a request for N years is served from any cached
  entry with >= N years of data, trimmed to the requested window.  This
  avoids re-downloading when the user shortens the look-back period.
* EF result memoisation: the Efficient Frontier computation (scipy
  optimisations + 3 000 Dirichlet portfolios) is keyed on a hash of the
  covariance matrix and parameters.  Repeated runs with the same data
  return instantly.
* Ticker-directory slot: the loaded DataFrame is cached in-memory so the
  file/URL is read only once per process.
* FF factor disk persistence: Fama-French factor DataFrames are written to
  a pickle file in the system temp directory so they survive app restarts.
  The pickle is invalidated when it is older than `ff_disk_ttl_days` days.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
import pickle
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)
_FF_PICKLE_NAME = "lazyfin_ff_cache_{model}.pkl"


class PortfolioCache:
    """
    TTL-aware in-memory cache for OHLCV, price series, EF results,
    Fama-French factors (+ disk persistence), and the ticker directory.
    """

    def __init__(
        self,
        ttl_hours: float = 8.0,
        ff_disk_ttl_days: float = 7.0,
    ) -> None:
        self.cache_ttl_hours  = ttl_hours
        self.ff_disk_ttl_days = ff_disk_ttl_days

        # OHLCV: (symbol, years) → DataFrame
        self._ohlcv:     dict = {}
        self._ohlcv_ts:  dict = {}

        # Price series: (symbol, start_iso, end_iso) → Series
        self._price:     dict = {}
        self._price_ts:  dict = {}

        # Fama-French factors: model → DataFrame (in-memory only after load)
        self._ff:        dict = {}

        # EF results: hash_str → dict
        self._ef:        dict = {}

        # Ticker directory: single slot
        self._ticker_dir: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    def set_ttl(self, ttl_hours: float) -> None:
        self.cache_ttl_hours = ttl_hours
        self.clear_expired()

    def _is_expired(self, ts_dict: dict, key) -> bool:
        if self.cache_ttl_hours == float("inf"):
            return False
        ts = ts_dict.get(key)
        if ts is None:
            return False
        age_h = (datetime.datetime.now() - ts).total_seconds() / 3600.0
        return age_h > self.cache_ttl_hours

    # ------------------------------------------------------------------
    # OHLCV — with superset lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _ohlcv_key(symbol: str, years: int) -> tuple:
        return (symbol.upper().strip(), int(years))

    def get_ohlcv(self, symbol: str, years: int) -> Optional[pd.DataFrame]:
        """
        Return a cached DataFrame for (symbol, years).

        If an exact match is not found, look for any cached entry for the
        same symbol with MORE years of data and return a trimmed copy.
        This avoids re-downloading when the user shortens the look-back.
        """
        key = self._ohlcv_key(symbol, years)

        # Exact hit
        if key in self._ohlcv and not self._is_expired(self._ohlcv_ts, key):
            return self._ohlcv[key].copy()

        # Superset lookup: find any cached entry with >= years for this symbol
        sym = symbol.upper().strip()
        best_df: Optional[pd.DataFrame] = None
        for (s, y), df in self._ohlcv.items():
            if s != sym or y <= years:
                continue
            k2 = (s, y)
            if self._is_expired(self._ohlcv_ts, k2):
                continue
            # Trim to the requested window
            cutoff = datetime.datetime.now() - datetime.timedelta(days=365 * years)
            trimmed = df[df.index >= cutoff]
            if len(trimmed) >= 14:
                if best_df is None or len(trimmed) > len(best_df):
                    best_df = trimmed
        if best_df is not None:
            _LOG.debug("OHLCV superset hit: %s/%dyr from larger cache.", sym, years)
            return best_df.copy()

        return None

    def put_ohlcv(self, symbol: str, years: int, df: pd.DataFrame) -> None:
        key = self._ohlcv_key(symbol, years)
        self._ohlcv[key]    = df.copy()
        self._ohlcv_ts[key] = datetime.datetime.now()

    def ohlcv_entry_count(self) -> int:
        return sum(1 for k in self._ohlcv
                   if not self._is_expired(self._ohlcv_ts, k))

    # ------------------------------------------------------------------
    # Price series
    # ------------------------------------------------------------------

    @staticmethod
    def _price_key(symbol: str, start_date, end_date) -> tuple:
        def _iso(d):
            return d.isoformat()[:10] if hasattr(d, "isoformat") else str(d)[:10]
        return (symbol.upper().strip(), _iso(start_date), _iso(end_date))

    def get_price(self, symbol: str, start_date, end_date) -> Optional[pd.Series]:
        key = self._price_key(symbol, start_date, end_date)
        if self._is_expired(self._price_ts, key):
            self._price.pop(key, None); self._price_ts.pop(key, None)
            return None
        s = self._price.get(key)
        return s.copy() if s is not None else None

    def put_price(self, symbol: str, start_date, end_date,
                  close_series: pd.Series) -> None:
        key = self._price_key(symbol, start_date, end_date)
        self._price[key]    = close_series.copy()
        self._price_ts[key] = datetime.datetime.now()

    def price_entry_count(self) -> int:
        return sum(1 for k in self._price
                   if not self._is_expired(self._price_ts, k))

    # ------------------------------------------------------------------
    # Efficient Frontier memoisation
    # ------------------------------------------------------------------

    @staticmethod
    def _ef_key(log_returns: pd.DataFrame, rf_annual: float,
                n_portfolios: int, seed: int) -> str:
        """
        Hash the covariance matrix + parameters.
        Fast: hashing a float array is O(n^2) for n assets.
        """
        cov = log_returns.cov().values
        buf = hashlib.md5(cov.tobytes()).hexdigest()
        return f"{buf}_{rf_annual:.6f}_{n_portfolios}_{seed}"

    def get_ef(self, log_returns: pd.DataFrame, rf_annual: float,
               n_portfolios: int, seed: int) -> Optional[dict]:
        key = self._ef_key(log_returns, rf_annual, n_portfolios, seed)
        return self._ef.get(key)

    def put_ef(self, log_returns: pd.DataFrame, rf_annual: float,
               n_portfolios: int, seed: int, result: dict) -> None:
        key = self._ef_key(log_returns, rf_annual, n_portfolios, seed)
        self._ef[key] = result

    # ------------------------------------------------------------------
    # Fama-French factors  (in-memory + disk persistence)
    # ------------------------------------------------------------------

    def _ff_pickle_path(self, model: str) -> str:
        fname = _FF_PICKLE_NAME.format(model=model.upper())
        return os.path.join(tempfile.gettempdir(), fname)

    def _ff_disk_expired(self, path: str) -> bool:
        if not os.path.exists(path):
            return True
        age_days = (
            datetime.datetime.now()
            - datetime.datetime.fromtimestamp(os.path.getmtime(path))
        ).total_seconds() / 86_400.0
        return age_days > self.ff_disk_ttl_days

    def get_ff_factors(self, model: str) -> Optional[pd.DataFrame]:
        key = model.upper()
        # 1. In-memory first
        if key in self._ff:
            return self._ff[key].copy()
        # 2. Disk fallback
        path = self._ff_pickle_path(key)
        if not self._ff_disk_expired(path):
            try:
                df = pd.read_pickle(path)
                self._ff[key] = df          # warm in-memory cache
                _LOG.info("FF %s loaded from disk cache (%s).", key, path)
                return df.copy()
            except Exception as exc:
                _LOG.warning("Disk FF cache unreadable: %s", exc)
        return None

    def put_ff_factors(self, model: str, df: pd.DataFrame) -> None:
        key  = model.upper()
        self._ff[key] = df.copy()
        path = self._ff_pickle_path(key)
        try:
            df.to_pickle(path)
            _LOG.debug("FF %s persisted to %s.", key, path)
        except Exception as exc:
            _LOG.warning("Could not persist FF cache to disk: %s", exc)

    # ------------------------------------------------------------------
    # Ticker directory
    # ------------------------------------------------------------------

    def get_ticker_directory(self) -> Optional[pd.DataFrame]:
        return self._ticker_dir.copy() if self._ticker_dir is not None else None

    def put_ticker_directory(self, df: pd.DataFrame) -> None:
        self._ticker_dir = df.copy()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_expired(self) -> int:
        removed = 0
        for key in list(self._ohlcv):
            if self._is_expired(self._ohlcv_ts, key):
                del self._ohlcv[key]; self._ohlcv_ts.pop(key, None); removed += 1
        for key in list(self._price):
            if self._is_expired(self._price_ts, key):
                del self._price[key]; self._price_ts.pop(key, None); removed += 1
        return removed

    def clear_all(self) -> None:
        self._ohlcv.clear();  self._ohlcv_ts.clear()
        self._price.clear();  self._price_ts.clear()
        self._ff.clear();     self._ef.clear()
        self._ticker_dir = None
        # Remove disk-persisted FF factor pickles
        import os as _os
        for model in ("FF3", "FF5"):
            path = self._ff_pickle_path(model)
            try:
                if _os.path.exists(path):
                    _os.remove(path)
            except OSError:
                pass

    def info(self) -> str:
        now = datetime.datetime.now()
        def _age(ts_dict, key):
            ts = ts_dict.get(key)
            return f"{(now-ts).total_seconds()/60:.0f}m ago" if ts else "?"
        rows = [f"TTL: {self.cache_ttl_hours} h  |  FF disk TTL: {self.ff_disk_ttl_days} d"]
        if self._ohlcv:
            rows.append("OHLCV:")
            for (sym, yr), df in sorted(self._ohlcv.items()):
                rows.append(f"  {sym}/{yr}yr  {len(df):,} rows  [{_age(self._ohlcv_ts,(sym,yr))}]")
        if self._price:
            rows.append(f"Price series: {len(self._price)} entries")
        if self._ff:
            rows.append(f"FF factors:   {list(self._ff.keys())}")
        if self._ef:
            rows.append(f"EF results:   {len(self._ef)} memoised")
        if self._ticker_dir is not None:
            rows.append(f"Ticker dir:   {len(self._ticker_dir):,} rows")
        return "\n".join(rows)
