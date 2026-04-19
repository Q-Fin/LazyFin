"""
backtesting.py — Vectorised backtesting engine and walk-forward validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import compute_rsi, compute_bollinger, compute_macd

__all__ = [
    "VALID_STRATEGIES",
    "compute_sma_signals", "compute_rsi_signals",
    "compute_macd_signals", "compute_bb_signals",
    "generate_signals",
    "run_backtest", "format_backtest_table",
    "run_walkforward", "format_walkforward_aggregate_table",
]

VALID_STRATEGIES: frozenset[str] = frozenset(
    {"SMA Crossover", "RSI Mean-Reversion", "MACD", "Bollinger Band"}
)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def compute_sma_signals(close: pd.Series, *, fast: int, slow: int) -> pd.Series:
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow}).")
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    return (sma_fast > sma_slow).astype(int).fillna(0)


def compute_rsi_signals(
    close: pd.Series,
    *,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    rsi    = compute_rsi(close, period=period)
    signal = pd.Series(np.nan, index=close.index)
    signal[rsi <= oversold]   = 1
    signal[rsi >= overbought] = 0
    return signal.ffill().fillna(0).astype(int)


def compute_macd_signals(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow}).")
    m = compute_macd(close, fast=fast, slow=slow, signal_period=signal_period)
    return (m["macd_line"] > m["signal_line"]).astype(int).fillna(0)


def compute_bb_signals(
    close: pd.Series,
    *,
    window: int = 20,
    k: float = 2.0,
) -> pd.Series:
    bb     = compute_bollinger(close, window=window, k=k)
    signal = pd.Series(np.nan, index=close.index)
    signal[close <= bb["lower"]] = 1
    signal[close >= bb["mid"]]   = 0
    return signal.ffill().fillna(0).astype(int)


def generate_signals(
    close: pd.Series,
    strategy: str,
    params: dict,
) -> pd.Series:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid: {sorted(VALID_STRATEGIES)}"
        )
    if strategy == "SMA Crossover":
        return compute_sma_signals(
            close, fast=int(params["fast"]), slow=int(params["slow"])
        )
    if strategy == "RSI Mean-Reversion":
        return compute_rsi_signals(
            close,
            period     = int(params.get("period",     14)),
            oversold   = float(params.get("oversold",  30.0)),
            overbought = float(params.get("overbought",70.0)),
        )
    if strategy == "MACD":
        return compute_macd_signals(
            close,
            fast          = int(params.get("fast",   12)),
            slow          = int(params.get("slow",   26)),
            signal_period = int(params.get("signal",  9)),
        )
    # Bollinger Band
    return compute_bb_signals(
        close,
        window = int(params.get("window", 20)),
        k      = float(params.get("k",    2.0)),
    )


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    *,
    commission: float = 0.001,
) -> dict:
    if len(df) < 50:
        raise ValueError(f"Need ≥ 50 bars; got {len(df)}.")
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    close = df["Close"].squeeze()

    raw_signals  = generate_signals(close, strategy, params)
    position     = raw_signals.shift(1).fillna(0)

    log_rets     = np.log(close / close.shift(1)).fillna(0)
    trade_events = position.diff().abs().fillna(0)
    comm_cost    = trade_events * commission
    strat_rets   = position * log_rets - comm_cost

    strat_equity = np.exp(strat_rets.cumsum())
    bnh_equity   = np.exp(log_rets.cumsum())

    peak     = strat_equity.cummax()
    drawdown = (strat_equity - peak) / peak

    entries = (position.diff() ==  1)
    exits   = (position.diff() == -1)

    entry_dates = close.index[entries]
    exit_dates  = close.index[exits]
    trade_rets  = []
    ei          = 0
    for ed in entry_dates:
        while ei < len(exit_dates) and exit_dates[ei] <= ed:
            ei += 1
        if ei >= len(exit_dates):
            break
        xd  = exit_dates[ei]
        ret = float(np.log(close[xd] / close[ed]) - 2 * commission)
        trade_rets.append(ret)
        ei += 1

    n_trades  = len(trade_rets)
    win_rate  = float(np.mean([r > 0 for r in trade_rets])) if n_trades else np.nan
    avg_trade = float(np.mean(trade_rets)) if n_trades else np.nan

    n_years   = max(len(strat_rets) / 252, 0.01)
    total_ret = float(np.exp(strat_rets.sum()) - 1)
    cagr      = float(max(1 + total_ret, 1e-9) ** (1 / n_years) - 1)

    mean_r = float(strat_rets.mean())
    std_r  = float(strat_rets.std())
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else np.nan

    neg_r    = strat_rets.clip(upper=0.0)
    semi_std = float(np.sqrt((neg_r ** 2).mean()) * np.sqrt(252))
    sortino  = float(mean_r * 252 / semi_std) if semi_std > 1e-12 else np.nan

    mdd = float(-drawdown.min())

    return {
        "close":        close,
        "position":     position,
        "signals":      raw_signals,
        "log_rets":     log_rets,
        "strat_rets":   strat_rets,
        "strat_equity": strat_equity,
        "bnh_equity":   bnh_equity,
        "drawdown":     drawdown,
        "entries":      entries,
        "exits":        exits,
        "trade_rets":   trade_rets,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "avg_trade":    avg_trade,
        "total_ret":    total_ret,
        "cagr":         cagr,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": mdd,
        "commission":   commission,
        "strategy":     strategy,
        "params":       params,
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_backtest_table(bt: dict, symbol: str) -> pd.DataFrame:
    strategy = bt["strategy"]
    log_rets  = bt["log_rets"]
    bnh_eq    = bt["bnh_equity"]
    n_years   = max(len(log_rets) / 252, 0.01)

    bnh_total  = float(np.exp(log_rets.sum()) - 1)
    bnh_cagr   = float(max(1 + bnh_total, 1e-9) ** (1 / n_years) - 1)
    bnh_sharpe = (float(log_rets.mean() / log_rets.std() * np.sqrt(252))
                  if log_rets.std() > 1e-12 else np.nan)
    bnh_peak   = bnh_eq.cummax()
    bnh_mdd    = float(-((bnh_eq - bnh_peak) / bnh_peak).min())

    def _pct(v): return f"{v:+.2%}" if not np.isnan(v) else "N/A"
    def _f3(v):  return f"{v:.3f}"  if not np.isnan(v) else "N/A"

    rows = [
        ("Total Return",     _pct(bt["total_ret"]),                   _pct(bnh_total)),
        ("CAGR",             _pct(bt["cagr"]),                        _pct(bnh_cagr)),
        ("Sharpe Ratio",     _f3(bt["sharpe"]),                       _f3(bnh_sharpe)),
        ("Sortino Ratio",    _f3(bt["sortino"]),                      "—"),
        ("Max Drawdown",     f"{-bt['max_drawdown']:.2%}",            f"{-bnh_mdd:.2%}"),
        ("# Trades",         str(bt["n_trades"]),                     "—"),
        ("Win Rate",         _pct(bt["win_rate"]) if bt["n_trades"] > 0 else "N/A", "—"),
        ("Avg Trade Return", _pct(bt["avg_trade"]) if bt["n_trades"] > 0 else "N/A", "—"),
        ("Commission (bps)", f"{bt['commission'] * 10_000:.0f}",      "—"),
    ]
    return (
        pd.DataFrame(rows, columns=["Metric", strategy, "Buy & Hold"])
        .set_index("Metric")
    )


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walkforward(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    *,
    train_days: int = 252,
    test_days: int = 63,
    commission: float = 0.001,
) -> dict:
    n = len(df)
    if train_days + test_days > n:
        raise ValueError(
            f"train_days ({train_days}) + test_days ({test_days}) = "
            f"{train_days + test_days} > available bars ({n})."
        )

    close      = df["Close"].squeeze()
    folds      = []
    fold_start = 0

    while True:
        train_end = fold_start + train_days
        test_end  = train_end  + test_days
        if test_end > n:
            break

        test_df   = df.iloc[fold_start:test_end]
        bt_full   = run_backtest(test_df, strategy, params, commission=commission)
        test_idx  = df.index[train_end:test_end]

        if len(test_idx) < 10:
            fold_start += test_days
            continue

        s_eq_test  = bt_full["strat_equity"].loc[test_idx]
        bnh_eq_test= bt_full["bnh_equity"].loc[test_idx]
        s_dd_test  = bt_full["drawdown"].loc[test_idx]
        s_rets_test= bt_full["strat_rets"].loc[test_idx]
        bnh_rets   = bt_full["log_rets"].loc[test_idx]

        n_te    = len(s_rets_test)
        n_yrs   = max(n_te / 252, 0.01)
        tot_ret = float(np.exp(s_rets_test.sum()) - 1)
        cagr    = float(max(1 + tot_ret, 1e-9) ** (1 / n_yrs) - 1)
        mn_r    = float(s_rets_test.mean())
        sd_r    = float(s_rets_test.std())
        sharpe  = float(mn_r / sd_r * np.sqrt(252)) if sd_r > 1e-12 else np.nan
        mdd     = float(-s_dd_test.min())

        pos_test  = bt_full["position"].loc[test_idx]
        entries_t = (pos_test.diff() ==  1)
        exits_t   = (pos_test.diff() == -1)
        e_dates   = close.loc[test_idx].index[entries_t.loc[test_idx]]
        x_dates   = close.loc[test_idx].index[exits_t.loc[test_idx]]
        t_rets    = []
        xi        = 0
        for ed in e_dates:
            while xi < len(x_dates) and x_dates[xi] <= ed:
                xi += 1
            if xi >= len(x_dates):
                break
            xd     = x_dates[xi]
            t_ret  = float(np.log(close[xd] / close[ed]) - 2 * commission)
            t_rets.append(t_ret)
            xi += 1
        win_rate  = float(np.mean([r > 0 for r in t_rets])) if t_rets else np.nan
        n_trades  = int((pos_test.diff().abs() == 1).sum() // 2)

        bnh_tot = float(np.exp(bnh_rets.sum()) - 1)
        bnh_sh  = (float(bnh_rets.mean() / bnh_rets.std() * np.sqrt(252))
                   if bnh_rets.std() > 1e-12 else np.nan)

        folds.append({
            "fold":         len(folds) + 1,
            "test_start":   test_idx[0].date(),
            "test_end":     test_idx[-1].date(),
            "total_ret":    tot_ret,
            "cagr":         cagr,
            "sharpe":       sharpe,
            "mdd":          mdd,
            "n_trades":     n_trades,
            "win_rate":     win_rate,
            "bnh_total_ret":bnh_tot,
            "bnh_sharpe":   bnh_sh,
            "strat_equity": s_eq_test,
            "bnh_equity":   bnh_eq_test,
        })
        fold_start += test_days

    if not folds:
        raise ValueError(
            "No complete folds with the given window sizes. "
            "Reduce train_days / test_days or extend the date range."
        )

    def _nanmean(key):
        vals = [f[key] for f in folds
                if not (isinstance(f[key], float) and np.isnan(f[key]))]
        return float(np.mean(vals)) if vals else np.nan

    agg = {
        "mean_total_ret":  _nanmean("total_ret"),
        "mean_cagr":       _nanmean("cagr"),
        "mean_sharpe":     _nanmean("sharpe"),
        "mean_mdd":        _nanmean("mdd"),
        "mean_win_rate":   _nanmean("win_rate"),
        "mean_bnh_ret":    _nanmean("bnh_total_ret"),
        "mean_bnh_sharpe": _nanmean("bnh_sharpe"),
    }

    def _pct(v): return f"{v:+.2%}" if not np.isnan(v) else "N/A"
    def _f2(v):  return f"{v:.3f}"  if not np.isnan(v) else "N/A"

    fold_rows = []
    for f in folds:
        fold_rows.append({
            "Fold":         f["fold"],
            "Test Start":   str(f["test_start"]),
            "Test End":     str(f["test_end"]),
            "Total Return": _pct(f["total_ret"]),
            "CAGR":         _pct(f["cagr"]),
            "Sharpe":       _f2(f["sharpe"]),
            "Max DD":       f"{-f['mdd']:.2%}",
            "# Trades":     str(f["n_trades"]),
            "Win Rate":     f"{f['win_rate']:.1%}" if not np.isnan(f["win_rate"]) else "N/A",
            "B&H Return":   _pct(f["bnh_total_ret"]),
        })
    fold_df = pd.DataFrame(fold_rows).set_index("Fold")

    return {
        "folds":      folds,
        "fold_df":    fold_df,
        "strategy":   strategy,
        "train_days": train_days,
        "test_days":  test_days,
        "n_folds":    len(folds),
        "agg":        agg,
    }


def format_walkforward_aggregate_table(wf: dict) -> pd.DataFrame:
    agg = wf["agg"]
    def _pct(v): return f"{v:+.2%}" if not np.isnan(v) else "N/A"
    def _f2(v):  return f"{v:.3f}"  if not np.isnan(v) else "N/A"
    rows = [
        ("Strategy",                wf["strategy"]),
        ("Walk-Forward Folds",      str(wf["n_folds"])),
        ("Train Window (days)",      str(wf["train_days"])),
        ("Test Window (days)",       str(wf["test_days"])),
        ("Mean Fold Total Return",   _pct(agg["mean_total_ret"])),
        ("Mean Fold CAGR",           _pct(agg["mean_cagr"])),
        ("Mean Fold Sharpe",         _f2(agg["mean_sharpe"])),
        ("Mean Fold Max Drawdown",   f"{-agg['mean_mdd']:.2%}"),
        ("Mean Fold Win Rate",
         f"{agg['mean_win_rate']:.1%}" if not np.isnan(agg["mean_win_rate"]) else "N/A"),
        ("Mean B&H Return (folds)",  _pct(agg["mean_bnh_ret"])),
        ("Mean B&H Sharpe (folds)",  _f2(agg["mean_bnh_sharpe"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
