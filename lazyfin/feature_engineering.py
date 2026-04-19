"""
feature_engineering.py — Technical indicator computation.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _wilder_sum(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed cumulative sum (EWM mean * period)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean() * period


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def compute_bollinger(
    close: pd.Series,
    *,
    window: int = 20,
    k: float = 2.0,
) -> dict:
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    return {"upper": ma + k * std, "mid": ma, "lower": ma - k * std}


def compute_dema(close: pd.Series, *, window: int = 20) -> pd.Series:
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return (2 * ema1 - ema2).rename("DEMA")


def compute_rsi(close: pd.Series, *, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().rename("ATR")


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).fillna(0).cumsum().rename("OBV")


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = 14,
) -> dict:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    tr14       = _wilder_sum(tr,       period)
    plus_dm14  = _wilder_sum(plus_dm,  period)
    minus_dm14 = _wilder_sum(minus_dm, period)

    plus_di  = 100 * plus_dm14  / tr14.replace(0, np.nan)
    minus_di = 100 * minus_dm14 / tr14.replace(0, np.nan)
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / di_sum
    adx      = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return {"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx}


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    *,
    period: int = 14,
) -> pd.Series:
    tp       = (high + low + close) / 3.0
    mf       = tp * volume
    tp_diff  = tp.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, mf, 0.0), index=close.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, mf, 0.0), index=close.index)
    pmf      = pos_flow.rolling(window=period, min_periods=1).sum()
    nmf      = neg_flow.rolling(window=period, min_periods=1).sum()
    mfr      = pmf / nmf.replace(0, np.nan)
    return (100 - 100 / (1 + mfr)).rename("MFI")


def compute_psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    af_start: float = 0.02,
    af_max: float = 0.20,
) -> pd.Series:
    af        = af_start
    trend_up  = bool(close.iat[1] >= close.iat[0])
    ep        = float(high.iat[0]) if trend_up else float(low.iat[0])
    prev_psar = float(low.iat[0])  if trend_up else float(high.iat[0])
    psar_list = [prev_psar]

    for i in range(1, len(close)):
        new_psar = prev_psar + af * (ep - prev_psar)
        low_i1   = float(low.iat[i - 1])
        low_i2   = float(low.iat[i - 2]) if i > 1 else low_i1
        high_i1  = float(high.iat[i - 1])
        high_i2  = float(high.iat[i - 2]) if i > 1 else high_i1

        if trend_up:
            new_psar = min(new_psar, low_i1, low_i2)
            if float(low.iat[i]) < new_psar:
                trend_up, new_psar, ep, af = False, ep, float(low.iat[i]), af_start
            elif float(high.iat[i]) > ep:
                ep = float(high.iat[i])
                af = min(af + af_start, af_max)
        else:
            new_psar = max(new_psar, high_i1, high_i2)
            if float(high.iat[i]) > new_psar:
                trend_up, new_psar, ep, af = True, ep, float(high.iat[i]), af_start
            elif float(low.iat[i]) < ep:
                ep = float(low.iat[i])
                af = min(af + af_start, af_max)

        psar_list.append(new_psar)
        prev_psar = new_psar

    return pd.Series(psar_list, index=close.index, name="PSAR")


def compute_macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow}).")
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return {
        "macd_line":   macd_line,
        "signal_line": signal_line,
        "histogram":   histogram,
    }


# ---------------------------------------------------------------------------
# Indicator bundle
# ---------------------------------------------------------------------------

__all__ = [
    "ALL_INDICATOR_NAMES",
    "compute_bollinger", "compute_dema", "compute_rsi",
    "compute_atr", "compute_obv", "compute_adx",
    "compute_mfi", "compute_psar", "compute_macd",
    "compute_indicator_bundle",
]

ALL_INDICATOR_NAMES: frozenset[str] = frozenset({
    "Bollinger Bands", "DEMA", "Parabolic SAR",
    "RSI", "ATR", "OBV", "MFI", "ADX", "Volume", "MACD",
})


def compute_indicator_bundle(
    ohlcv: pd.DataFrame,
    selected: frozenset[str],
    params: dict,
) -> dict:
    unknown = selected - ALL_INDICATOR_NAMES
    if unknown:
        raise ValueError(f"Unknown indicators: {unknown}")

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

    high  = ohlcv["High"].squeeze()
    low   = ohlcv["Low"].squeeze()
    close = ohlcv["Close"].squeeze()
    vol   = ohlcv["Volume"].squeeze()

    bundle: dict = {"volume": vol}  # always populated

    if "Bollinger Bands" in selected:
        bundle["bollinger"] = compute_bollinger(
            close,
            window=int(params.get("bb_window", 20)),
            k=float(params.get("bb_k", 2.0)),
        )
    if "DEMA" in selected:
        bundle["dema"] = compute_dema(
            close, window=int(params.get("dema_window", 20))
        )
    if "RSI" in selected:
        bundle["rsi"] = compute_rsi(
            close, period=int(params.get("rsi_period", 14))
        )
    if "ATR" in selected:
        bundle["atr"] = compute_atr(
            high, low, close, period=int(params.get("atr_period", 14))
        )
    if "OBV" in selected:
        bundle["obv"] = compute_obv(close, vol)
    if "ADX" in selected:
        bundle["adx"] = compute_adx(
            high, low, close, period=int(params.get("adx_period", 14))
        )
    if "MFI" in selected:
        bundle["mfi"] = compute_mfi(
            high, low, close, vol, period=int(params.get("mfi_period", 14))
        )
    if "Parabolic SAR" in selected:
        bundle["psar"] = compute_psar(
            high, low, close,
            af_start=float(params.get("psar_af_start", 0.02)),
            af_max=float(params.get("psar_af_max", 0.20)),
        )
    if "MACD" in selected:
        mf = int(params.get("macd_fast",   12))
        ms = int(params.get("macd_slow",   26))
        mg = int(params.get("macd_signal",  9))
        if mf >= ms:
            raise ValueError(f"MACD fast ({mf}) must be < slow ({ms}).")
        bundle["macd"] = compute_macd(close, fast=mf, slow=ms, signal_period=mg)

    return bundle
