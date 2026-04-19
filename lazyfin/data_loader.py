"""
data_loader.py — Network and file-system data fetching.
"""
from __future__ import annotations

import io
import logging
import time
import urllib.request
import zipfile
import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from .cache import PortfolioCache
from .preprocessing import clean_ohlcv

__all__ = [
    "load_ticker_directory", "search_ticker_directory",
    "fetch_ohlcv", "fetch_price_series",
    "fetch_multi_price_series", "fetch_ff_factors",
]


_LOG = logging.getLogger(__name__)

_FF_URLS: dict[str, str] = {
    "FF3": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    ),
    "FF5": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    ),
}


# ---------------------------------------------------------------------------
# Ticker directory
# ---------------------------------------------------------------------------

def load_ticker_directory(
    *,
    url: Optional[str] = None,
    local_path: Optional[str] = None,
    content_base64: Optional[str] = None,
    filename: Optional[str] = None,
    cache: Optional["PortfolioCache"] = None,
    logger: logging.Logger = _LOG,
) -> dict:
    """
    Load a ticker directory from one of:
      - content_base64 : raw base64 string from dcc.Upload (highest priority)
      - local_path     : local CSV/TSV file path
      - url            : remote URL
      - fallback       : Wikipedia S&P 500 → hard-coded stub

    Supported CSV formats
    ---------------------
    Both comma-separated and semicolon-separated files are detected
    automatically.  Column names must include "Symbol" and optionally
    "Instrument Fullname" (or "Instrument Full Name", "Name", "Description").
    Blank rows and leading/trailing whitespace are removed.

    Typical semicolon format (matches the original notebook):

        Symbol;Instrument Fullname

        ASIG;iShares $ Asia Investment Grade Corp Bond UCITS ETF USD (Acc)

        SDIS;Leverage Shares -1x Short Disney ETP Securities

    Result is cached in the PortfolioCache ticker-directory slot so
    subsequent calls return immediately without re-reading the file.
    """
    # Serve from cache if already loaded
    if cache is not None:
        cached = cache.get_ticker_directory()
        if cached is not None:
            return {"data": cached, "source": "cache"}

    def _parse_csv(text: str) -> Optional[pd.DataFrame]:
        """Try comma then semicolon separator; return the one that produces >= 2 columns."""
        import io as _io
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(
                    _io.StringIO(text),
                    sep=sep,
                    dtype=str,
                    engine="python",
                    encoding="utf-8",
                    keep_default_na=False,
                    skip_blank_lines=True,
                )
                if df.shape[1] >= 2 and len(df) > 0:
                    return df
            except Exception:
                continue
        return None

    data, source = None, None

    # ── Priority 1: base64 upload from dcc.Upload ─────────────────────────
    if content_base64:
        try:
            import base64 as _b64, io as _io
            # Strip data-URI prefix if present
            raw = content_base64
            if "," in raw:
                raw = raw.split(",", 1)[1]
            decoded = _b64.b64decode(raw).decode("utf-8", errors="ignore")
            df = _parse_csv(decoded)
            if df is not None:
                data   = df
                source = f"uploaded file: {filename or 'unknown'}"
        except Exception as exc:
            logger.warning("Upload parse failed: %s", exc)

    # ── Priority 2: local file ─────────────────────────────────────────────
    if data is None and local_path:
        try:
            with open(local_path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            df = _parse_csv(text)
            if df is not None:
                data   = df
                source = f"local file: {local_path}"
        except Exception as exc:
            logger.warning("Local file read failed (%s): %s", local_path, exc)

    # ── Priority 3: remote URL ─────────────────────────────────────────────
    if data is None and url:
        try:
            import urllib.request as _ur
            req = _ur.Request(url, headers={"User-Agent": "Python/3"})
            with _ur.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            df = _parse_csv(text)
            if df is not None:
                data   = df
                source = f"remote URL: {url}"
        except Exception as exc:
            logger.warning("URL fetch failed (%s): %s", url, exc)

    # ── Priority 4: Wikipedia S&P 500 ─────────────────────────────────────
    if data is None:
        logger.info("Fetching S&P 500 from Wikipedia…")
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            sp500 = tables[0][["Symbol", "Security"]].rename(
                columns={"Security": "Instrument Fullname"}
            )
            data   = sp500
            source = "Wikipedia S&P 500"
        except Exception as exc:
            logger.warning("Wikipedia fetch failed: %s", exc)

    # ── Priority 5: hard-coded stub ────────────────────────────────────────
    if data is None:
        data = pd.DataFrame({
            "Symbol":              ["SPY", "QQQ", "IWM"],
            "Instrument Fullname": [
                "SPDR S&P 500 ETF Trust",
                "Invesco QQQ Trust (Nasdaq-100)",
                "iShares Russell 2000 ETF",
            ],
        })
        source = "hard-coded fallback"

    # ── Normalise columns ──────────────────────────────────────────────────
    data = data.copy()
    data.columns = [c.strip() for c in data.columns]

    # Detect symbol column
    sym_col = next(
        (c for c in data.columns if c.strip().lower() == "symbol"),
        data.columns[0],
    )
    if sym_col != "Symbol":
        data = data.rename(columns={sym_col: "Symbol"})

    # Detect name column (various spellings)
    name_aliases = {
        "instrument fullname", "instrument full name",
        "name", "description", "full name", "company",
    }
    name_col = next(
        (c for c in data.columns if c.strip().lower() in name_aliases),
        None,
    )
    if name_col and name_col != "Instrument Fullname":
        data = data.rename(columns={name_col: "Instrument Fullname"})
    if "Instrument Fullname" not in data.columns:
        data["Instrument Fullname"] = ""

    # Clean: strip whitespace, drop blanks, deduplicate
    data["Symbol"] = data["Symbol"].astype(str).str.strip().str.upper()
    data["Instrument Fullname"] = (
        data["Instrument Fullname"].astype(str).str.strip()
    )
    tidy = (
        data[["Symbol", "Instrument Fullname"]]
        .pipe(lambda d: d[d["Symbol"].str.len() > 0])
        .drop_duplicates(subset="Symbol")
        .reset_index(drop=True)
    )

    logger.info("Ticker directory: %d symbols from %s.", len(tidy), source)
    if cache is not None:
        cache.put_ticker_directory(tidy)
    return {"data": tidy, "source": source}


def search_ticker_directory(
    directory: pd.DataFrame,
    query: str,
    *,
    max_results: int = 200,
) -> list[tuple[str, str]]:
    if directory.empty or not query.strip():
        subset = directory.sort_values("Symbol").head(max_results)
        return [
            (f"{r['Symbol']} — {r['Instrument Fullname']}", r["Symbol"])
            for _, r in subset.iterrows()
        ]
    q_up  = query.strip().upper()
    df    = directory
    exact  = df[df["Symbol"] == q_up]
    starts = df[df["Symbol"].str.startswith(q_up) & (df["Symbol"] != q_up)]
    cont   = df[
        df["Symbol"].str.contains(q_up, regex=False) &
        ~df["Symbol"].str.startswith(q_up)
    ]
    name_m = df[
        df["Instrument Fullname"].str.lower().str.contains(query.lower(), na=False)
    ]
    subset = (
        pd.concat([exact, starts, cont, name_m], ignore_index=True)
        .drop_duplicates(subset="Symbol")
        .head(max_results)
    )
    return [
        (f"{r['Symbol']} — {r['Instrument Fullname']}", r["Symbol"])
        for _, r in subset.iterrows()
    ]


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    years: int,
    cache: PortfolioCache,
    *,
    logger: logging.Logger = _LOG,
) -> Optional[dict]:
    symbol = symbol.upper().strip()
    years  = int(years)

    cached = cache.get_ohlcv(symbol, years)
    if cached is not None:
        logger.info("Cache hit: %s/%dyr (%d rows).", symbol, years, len(cached))
        return {"df": cached, "symbol": symbol, "years": years, "from_cache": True}

    start = datetime.datetime.now() - datetime.timedelta(days=365 * years)
    logger.info("Downloading %s (%d yr) from yfinance…", symbol, years)
    try:
        raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.warning("Download failed for %s: %s", symbol, exc)
        return None

    if raw.empty:
        logger.warning("No data returned for '%s'.", symbol)
        return None

    try:
        df = clean_ohlcv(raw)
    except ValueError as exc:
        logger.warning("OHLCV cleaning failed for %s: %s", symbol, exc)
        return None

    if len(df) < 14:
        logger.warning("Too few rows for '%s': %d (min 14).", symbol, len(df))
        return None

    cache.put_ohlcv(symbol, years, df)
    logger.info("Cached %s: %d clean rows.", symbol, len(df))
    return {"df": df.copy(), "symbol": symbol, "years": years, "from_cache": False}


# ---------------------------------------------------------------------------
# Price series
# ---------------------------------------------------------------------------

def fetch_price_series(
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
    cache: PortfolioCache,
    *,
    logger: logging.Logger = _LOG,
) -> Optional[dict]:
    symbol = symbol.upper().strip()

    cached = cache.get_price(symbol, start_date, end_date)
    if cached is not None:
        logger.info("Price cache hit: %s (%d obs).", symbol, len(cached))
        return {
            "series":     cached,
            "symbol":     symbol,
            "start_date": start_date,
            "end_date":   end_date,
            "from_cache": True,
        }

    logger.info("Downloading price series: %s (%s → %s)…", symbol, start_date, end_date)
    try:
        raw = yf.download(
            symbol, start=start_date, end=end_date,
            auto_adjust=True, progress=False,
        )
    except Exception as exc:
        logger.warning("Download failed for %s: %s", symbol, exc)
        return None

    if raw.empty:
        logger.warning("No data for '%s'.", symbol)
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    col = "Close" if "Close" in raw.columns else raw.columns[0]
    close = raw[col].dropna()
    if len(close) < 2:
        logger.warning("Insufficient data for '%s'.", symbol)
        return None

    cache.put_price(symbol, start_date, end_date, close)
    return {
        "series":     close.copy(),
        "symbol":     symbol,
        "start_date": start_date,
        "end_date":   end_date,
        "from_cache": False,
    }


def fetch_multi_price_series(
    symbols: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    cache: PortfolioCache,
    *,
    logger: logging.Logger = _LOG,
) -> Optional[pd.DataFrame]:
    symbols = [s.upper().strip() for s in symbols]

    # Serve from cache where possible; collect misses
    cached_frames  = {}
    missing_syms   = []
    for sym in symbols:
        c = cache.get_price(sym, start_date, end_date)
        if c is not None:
            cached_frames[sym] = c
        else:
            missing_syms.append(sym)

    # Batch-download all misses in one call
    if missing_syms:
        logger.info("Downloading %s (%s → %s)…", missing_syms, start_date, end_date)
        try:
            raw = yf.download(
                missing_syms, start=start_date, end=end_date,
                auto_adjust=True, progress=False,
            )
        except Exception as exc:
            logger.warning("Batch download failed: %s", exc)
            raw = pd.DataFrame()

        if not raw.empty:
            # Robustly extract the Close column for single- and multi-ticker downloads.
            if isinstance(raw.columns, pd.MultiIndex):
                # Multi-ticker: columns are (field, ticker). "Close" is top-level.
                level0 = raw.columns.get_level_values(0)
                if "Close" in level0:
                    close_raw = raw["Close"]          # → DataFrame, columns = tickers
                else:
                    # Fall back to the first available field
                    first_field = level0[0]
                    close_raw = raw[first_field]
                    logger.warning("'Close' not in MultiIndex level-0; using '%s'.", first_field)
            else:
                # Single-ticker: flat columns
                if "Close" in raw.columns:
                    close_raw = raw[["Close"]].rename(columns={"Close": missing_syms[0]})
                elif "Adj Close" in raw.columns:
                    close_raw = raw[["Adj Close"]].rename(columns={"Adj Close": missing_syms[0]})
                else:
                    close_raw = raw.iloc[:, :1]
                    close_raw.columns = [missing_syms[0]]

            if isinstance(close_raw, pd.Series):
                close_raw = close_raw.to_frame(name=missing_syms[0])

            for sym in missing_syms:
                if sym in close_raw.columns:
                    s = close_raw[sym].dropna()
                    if len(s) >= 2:
                        cache.put_price(sym, start_date, end_date, s)
                        cached_frames[sym] = s
                    else:
                        logger.warning("Insufficient data for '%s' — dropped.", sym)
                else:
                    logger.warning("No column for '%s' in download — dropped.", sym)

    if not cached_frames:
        return None

    df = pd.DataFrame(cached_frames).dropna(how="all")
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Fama-French factor data
# ---------------------------------------------------------------------------

def fetch_ff_factors(
    model: str,
    cache: PortfolioCache,
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 2.0,
    logger: logging.Logger = _LOG,
) -> pd.DataFrame:
    key = model.upper()
    if key not in ("FF3", "FF5"):
        raise ValueError(f"Unknown model '{model}'. Valid: 'FF3', 'FF5'.")

    cached = cache.get_ff_factors(key)
    if cached is not None:
        logger.info("FF cache hit: %s (%d obs).", key, len(cached))
        return cached

    url = _FF_URLS[key]
    logger.info("Downloading %s factors from Ken French's library…", key)

    buf, last_exc = None, None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                buf = io.BytesIO(resp.read())
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay_seconds * (2.0 ** attempt)
                logger.warning("Download attempt %d failed: %s. Retrying in %.0fs…",
                               attempt + 1, exc, delay)
                time.sleep(delay)

    if buf is None:
        raise ConnectionError(
            f"Failed to download {key} factor data after {max_retries} attempt(s). "
            f"Last error: {last_exc}"
        )

    with zipfile.ZipFile(buf) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("latin-1")

    # Parse the French CSV format
    header, rows = None, {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        first = parts[0]
        if (header is None and first == "" and
                all(p.replace("-", "").replace("_", "").isalpha()
                    for p in parts[1:] if p)):
            header = [p for p in parts[1:] if p]
            continue
        if header and len(first) == 8 and first.isdigit():
            try:
                date = pd.to_datetime(first, format="%Y%m%d")
                vals = {
                    header[j]: float(parts[j + 1]) / 100.0
                    for j in range(min(len(header), len(parts) - 1))
                    if parts[j + 1].replace(".", "").replace("-", "").isdigit()
                }
                rows[date] = vals
            except Exception:
                continue
        elif header and rows and first and not first.isdigit():
            break

    if not rows:
        raise ValueError(f"Parsed 0 rows from {key} factor file.")

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index().dropna()
    df.index.name = "Date"
    cache.put_ff_factors(key, df)
    logger.info("Cached %s: %d trading days.", key, len(df))
    return df
