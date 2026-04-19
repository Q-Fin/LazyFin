"""
pipeline.py
===========
Orchestration layer for the AlephLens-Toolkit.

Contract
--------
- Every pipeline function calls the appropriate core modules in sequence
  and returns a structured, typed result containing BOTH numeric data AND
  Plotly figures.
- No UI code of any kind: no widgets, no display(), no clear_output(),
  no IPython imports.
- No global references.  Every dependency is an explicit parameter.
- Network calls (yfinance, FF download) are isolated to the data_loader
  functions that are called from here; the pipeline itself contains no
  direct HTTP requests.
- Errors in optional analysis sections (e.g. GARCH, Benchmark) are
  captured per-section rather than propagated globally, so one failure
  does not suppress all other outputs.

Usage example
-------------
    from aleph_toolkit.cache import PortfolioCache
    from aleph_toolkit.pipeline import run_var_analysis

    cache  = PortfolioCache(ttl_hours=8.0)
    result = run_var_analysis(
        tickers      = ["AAPL", "MSFT", "GOOGL"],
        start_date   = datetime.date(2020, 1, 1),
        end_date     = datetime.date(2024, 12, 31),
        alpha        = 0.99,
        weights      = None,           # equal-weighted
        methods      = ["historical", "parametric", "montecarlo"],
        rf_annual    = 0.045,
        rolling_window  = 252,
        bench_symbol    = "SPY",
        ff_model        = "FF3",
        garch_p         = 1,
        garch_q         = 1,
        n_sims          = 10_000,
        cache           = cache,
    )

    # Data access
    print(result.portfolio["port_rets"].mean())
    print(result.tables["var_summary"])

    # Figure access
    result.figures["var_comparison"].show()
    result.figures["rolling_var"].show()
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Core computation modules
from .analytics import (
    TRADING_DAYS_PER_YEAR,
    compute_factor_regression,
    compute_garch_model,
    compute_performance_metrics,
    compute_rolling_var,
    compute_var_results,
    format_cornishfisher_diagnostics,
    format_factor_table,
    format_garch_table,
    format_performance_table,
    format_var_summary_table,
)
from .backtesting import (
    format_backtest_table,
    format_walkforward_aggregate_table,
    run_backtest as _run_backtest_core,
    run_walkforward as _run_walkforward_core,
)
from .cache import PortfolioCache
from .data_loader import (
    fetch_ff_factors,
    fetch_multi_price_series,
    fetch_ohlcv,
    fetch_price_series,
)
from .feature_engineering import compute_indicator_bundle
from .preprocessing import (
    build_portfolio_returns,
    compute_log_returns,
    compute_single_log_returns,
    equal_weights,
    normalise_weights,
    resolve_date_range,
)
from .stress_testing import (
    DEFAULT_STRESS_PRESETS,
    StressScenario,
    compute_stress_scenarios,
)
from .visualization import (
    DEFAULT_PLOTLY_TEMPLATE,
    plot_backtest_results,
    plot_benchmark_comparison,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_factor_attribution,
    plot_garch_volatility,
    plot_indicator_chart,
    plot_rolling_var,
    plot_stress_results,
    plot_var_comparison,
    plot_walkforward_results,
)

__all__ = [
    "VarAnalysisResult", "BacktestPipelineResult",
    "WalkForwardPipelineResult", "StressTestPipelineResult",
    "IndicatorPipelineResult",
    "run_var_analysis", "run_backtest",
    "run_walkforward", "run_stress_test",
    "run_indicator_analysis",
]


_LOG = logging.getLogger(__name__)

# Minimum trading days required for reliable analytics
_MIN_OBS: int = 30


# ---------------------------------------------------------------------------
# Shared helper: compute efficient frontier inline
# (avoids a circular import; calls scipy directly)
# ---------------------------------------------------------------------------

def _compute_ef_inline(
    log_returns: pd.DataFrame,
    rf_annual: float,
    n_portfolios: int,
    seed: int,
    cache=None,
) -> dict:
    """
    Thin wrapper; delegates to analytics.compute_efficient_frontier.
    Passes the cache for EF memoisation.  n_portfolios is adapted
    automatically inside compute_efficient_frontier based on n_assets.
    """
    from .analytics import compute_efficient_frontier  # noqa: F811
    return compute_efficient_frontier(
        log_returns,
        rf_annual    = rf_annual,
        n_portfolios = n_portfolios,
        seed         = seed,
        cache        = cache,
    )


def _compute_benchmark_inline(
    port_rets: pd.Series,
    bench_rets: pd.Series,
) -> dict:
    """
    Compute all derived benchmark quantities for the plot and for the table.
    Returns a dict matching the fields expected by plot_benchmark_comparison().
    """
    ann = TRADING_DAYS_PER_YEAR

    # Align to common dates
    common      = port_rets.index.intersection(bench_rets.index)
    p           = port_rets.loc[common]
    b           = bench_rets.loc[common]

    p_wealth    = np.exp(p.cumsum())
    b_wealth    = np.exp(b.cumsum())
    p_peak      = p_wealth.cummax()
    b_peak      = b_wealth.cummax()
    p_dd        = (p_wealth - p_peak) / p_peak
    b_dd        = (b_wealth - b_peak) / b_peak

    return {
        "port_wealth"   : p_wealth,
        "bench_wealth"  : b_wealth,
        "port_drawdown" : p_dd,
        "bench_drawdown": b_dd,
        "port_ann_ret"  : float(np.exp(p.mean() * ann) - 1.0),
        "port_ann_vol"  : float(p.std() * np.sqrt(ann)),
        "port_mdd"      : float(-p_dd.min()),
        "bench_ann_ret" : float(np.exp(b.mean() * ann) - 1.0),
        "bench_ann_vol" : float(b.std() * np.sqrt(ann)),
        "bench_mdd"     : float(-b_dd.min()),
    }


def _compute_corr_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.dropna().corr()


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class VarAnalysisResult:
    """
    Structured output of run_var_analysis().

    All optional fields are None when the corresponding analysis section
    was skipped (e.g. fewer than 2 tickers → no efficient frontier) or
    failed (captured in `errors`).

    Fields
    ------
    portfolio       : core price + return data for the full period.
    var_result      : per-method VaR/CVaR numbers.
    perf_metrics    : annualised risk/return metrics.
    rolling_var_df  : rolling VaR DataFrame (or None if window > n_obs).
    corr_matrix     : Pearson correlation matrix (or None if < 2 tickers).
    ef_result       : efficient frontier (or None if < 2 tickers).
    bench_data      : benchmark wealth/drawdown bundle (or None).
    ff_result       : Fama-French OLS regression (or None).
    garch_result    : GARCH model fit (or None if arch not installed).
    figures         : mapping of section name → go.Figure.
    tables          : mapping of section name → pd.DataFrame.
    errors          : mapping of section name → error message string.
    """
    # Core data
    portfolio:      dict                                   # PortfolioReturns
    var_result:     dict                                   # VarAnalysisResult
    perf_metrics:   dict                                   # PerformanceMetrics

    # Optional analytics
    rolling_var_df: Optional[pd.DataFrame]  = None
    corr_matrix:    Optional[pd.DataFrame]  = None
    ef_result:      Optional[dict]          = None
    bench_data:     Optional[dict]          = None
    ff_result:      Optional[dict]          = None
    garch_result:   Optional[dict]          = None

    # Rendered outputs
    figures: dict[str, go.Figure]           = field(default_factory=dict)
    tables:  dict[str, pd.DataFrame]        = field(default_factory=dict)
    errors:  dict[str, str]                 = field(default_factory=dict)


@dataclass
class BacktestPipelineResult:
    """
    Structured output of run_backtest().

    Fields
    ------
    bt        : raw backtest result dict (equity curves, metrics, trade log).
    bt_table  : formatted strategy-vs-buy-and-hold comparison DataFrame.
    figure    : 3-panel Plotly backtest chart.
    errors    : section name → error string (empty when all succeeds).
    """
    bt:      dict
    bt_table: pd.DataFrame
    figure:  go.Figure
    errors:  dict[str, str] = field(default_factory=dict)


@dataclass
class WalkForwardPipelineResult:
    """
    Structured output of run_walkforward().

    Fields
    ------
    wf         : raw walk-forward result dict (folds, agg, fold_df).
    agg_table  : formatted aggregate metrics DataFrame.
    fold_table : per-fold summary DataFrame.
    figure     : 2-panel Plotly walk-forward chart.
    errors     : section name → error string.
    """
    wf:        dict
    agg_table: pd.DataFrame
    fold_table: pd.DataFrame
    figure:    go.Figure
    errors:    dict[str, str] = field(default_factory=dict)


@dataclass
class StressTestPipelineResult:
    """
    Structured output of run_stress_test().

    Fields
    ------
    scenarios_df : impact table; columns: Scenario, Factor Shocks,
                   Est. 1-Day Impact, Impact %.
    ff_model     : FF model name used in the regression.
    figure       : horizontal bar chart of scenario impacts.
    errors       : section name → error string.
    """
    scenarios_df: pd.DataFrame
    ff_model:     str
    figure:       go.Figure
    errors:       dict[str, str] = field(default_factory=dict)


@dataclass
class IndicatorPipelineResult:
    """
    Structured output of run_indicator_analysis().

    Fields
    ------
    ohlcv      : validated OHLCV DataFrame used for the chart.
    indicators : computed indicator bundle (all selected indicators).
    figure     : unified multi-panel Plotly indicator chart.
    errors     : section name → error string.
    """
    ohlcv:      pd.DataFrame
    indicators: dict
    figure:     go.Figure
    errors:     dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline 1: Full VaR / CVaR analysis
# ---------------------------------------------------------------------------

def run_var_analysis(
    tickers: list[str],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    alpha: float,
    weights: Optional[list[float]],
    methods: list[str],
    rf_annual: float,
    rolling_window: int,
    bench_symbol: str,
    ff_model: str,
    garch_p: int,
    garch_q: int,
    cache: PortfolioCache,
    *,
    n_sims: int = 10_000,
    seed: int = 42,
    n_portfolios_ef: int = 3_000,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
    logger: logging.Logger = _LOG,
) -> VarAnalysisResult:
    """
    Execute the complete VaR/CVaR analysis pipeline and return all data
    and figures in a single structured result.

    Sections (run in dependency order)
    -----------------------------------
    1.  Resolve dates and validate inputs.
    2.  Download price series (cache-backed).
    3.  Build portfolio returns (log-returns, weights).
    4.  Compute VaR/CVaR for all requested methods.
    5.  Compute performance metrics.
    6.  Compute rolling VaR (skipped if n_obs < window).
    7.  Compute correlation matrix (skipped if < 2 tickers).
    8.  Compute efficient frontier (skipped if < 2 tickers).
    9.  Fetch benchmark + compute wealth/drawdown comparison.
    10. Download FF factors + run OLS regression.
    11. Fit GARCH(p, q) model.

    Figures produced (keys in result.figures)
    ------------------------------------------
    'var_comparison'     — return histogram with VaR/CVaR lines
    'rolling_var'        — 2-panel rolling VaR chart
    'correlation'        — Pearson correlation heatmap
    'efficient_frontier' — Markowitz frontier scatter
    'benchmark'          — wealth + drawdown comparison
    'factor_attribution' — FF factor loading bar chart
    'garch'              — conditional volatility chart

    Tables produced (keys in result.tables)
    ----------------------------------------
    'var_summary'         — method × {VaR, CVaR, VaR%, CVaR%}
    'cf_diagnostics'      — CF expansion stats (if CF method run)
    'performance'         — annualised return/risk metrics
    'factor_regression'   — OLS estimates with HC3 SE, t-stats, p-values
    'garch_summary'       — model parameters + persistence + forecast vol

    Parameters
    ----------
    tickers         : list[str]            Non-empty list of yfinance symbols.
    start_date      : datetime.date | None  Window start (None → today − 5yr).
    end_date        : datetime.date | None  Window end (None → today).
    alpha           : float                Confidence level, e.g. 0.99.
    weights         : list[float] | None   Raw un-normalised weights.
                                           None → equal weighting.
    methods         : list[str]            Subset of:
                                           'historical', 'parametric',
                                           'montecarlo', 'cornishfisher'.
    rf_annual       : float                Annual risk-free rate (decimal).
    rolling_window  : int                  Rolling VaR window in trading days.
    bench_symbol    : str                  Benchmark ticker, e.g. 'SPY'.
                                           Empty string → skip benchmark.
    ff_model        : str                  'FF3' or 'FF5'.
    garch_p         : int                  GARCH lag order.
    garch_q         : int                  ARCH lag order.
    cache           : PortfolioCache       Injected cache (read + write).
    n_sims          : int                  Monte Carlo simulation count.
    seed            : int                  RNG seed.
    n_portfolios_ef : int                  Random portfolios for EF scatter.
    plotly_template : str                  Plotly layout template.
    logger          : Logger               Progress message destination.

    Returns
    -------
    VarAnalysisResult   Always returns; section failures are in `.errors`.

    Raises
    ------
    ValueError   If tickers is empty, methods is empty, or the price
                 download yields fewer than _MIN_OBS trading days.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if not methods:
        raise ValueError("methods must be a non-empty list.")

    figures: dict[str, go.Figure] = {}
    tables:  dict[str, pd.DataFrame] = {}
    errors:  dict[str, str] = {}

    # ── 1. Resolve dates ─────────────────────────────────────────────────────
    start_date, end_date = resolve_date_range(start_date, end_date,
                                              fallback_years=5)

    # ── 2. Download prices ────────────────────────────────────────────────────
    logger.info("Downloading prices for %s (%s → %s)…",
                tickers, start_date, end_date)
    prices = fetch_multi_price_series(
        tickers, start_date, end_date, cache, logger=logger,
    )
    if prices is None or prices.empty:
        raise ValueError(
            f"No price data returned for {tickers} "
            f"({start_date} → {end_date})."
        )

    # ── 3. Build portfolio returns ────────────────────────────────────────────
    n_assets = prices.shape[1]
    if weights is not None:
        norm_w = normalise_weights(weights)
        if norm_w.size != n_assets:
            raise ValueError(
                f"weights length ({norm_w.size}) ≠ number of usable tickers ({n_assets})."
            )
    else:
        norm_w = equal_weights(n_assets)

    portfolio = build_portfolio_returns(prices, norm_w, start_date, end_date)
    portfolio["alpha"] = float(alpha)   # inject user-supplied confidence level
    port_rets = portfolio["port_rets"]

    # ── 4. VaR / CVaR ────────────────────────────────────────────────────────
    logger.info("Computing VaR/CVaR (%s)…", methods)
    var_result = compute_var_results(
        portfolio, methods, n_sims=n_sims, seed=seed,
    )

    try:
        tables["var_summary"] = format_var_summary_table(var_result, portfolio)
        if "cornishfisher" in var_result["method_results"]:
            tables["cf_diagnostics"] = format_cornishfisher_diagnostics(
                var_result["method_results"]["cornishfisher"]
            )
        figures["var_comparison"] = plot_var_comparison(
            port_rets            = port_rets,
            method_results       = var_result["method_results"],
            alpha                = alpha,
            tickers              = portfolio["tickers"],
            plotly_template      = plotly_template,
        )
    except Exception as exc:
        errors["var_comparison"] = str(exc)
        logger.warning("VaR comparison figure failed: %s", exc)

    # ── 5. Performance metrics ────────────────────────────────────────────────
    logger.info("Computing performance metrics…")
    perf_metrics = compute_performance_metrics(port_rets, rf_annual=rf_annual)
    tables["performance"] = format_performance_table(perf_metrics)

    # ── 6. Rolling VaR ───────────────────────────────────────────────────────
    rolling_var_df = None
    if len(port_rets) >= rolling_window:
        logger.info("Computing rolling VaR (window=%d)…", rolling_window)
        try:
            rolling_var_df = compute_rolling_var(
                port_rets, alpha, window=rolling_window,
            )["rolling_df"]
            figures["rolling_var"] = plot_rolling_var(
                rolling_df      = rolling_var_df,
                alpha           = alpha,
                window          = rolling_window,
                tickers         = portfolio["tickers"],
                plotly_template = plotly_template,
            )
        except Exception as exc:
            errors["rolling_var"] = str(exc)
            logger.warning("Rolling VaR failed: %s", exc)
    else:
        errors["rolling_var"] = (
            f"Insufficient observations ({len(port_rets)}) for "
            f"a {rolling_window}-day rolling window."
        )

    # ── 7. Correlation matrix ─────────────────────────────────────────────────
    corr_matrix = None
    if n_assets >= 2:
        logger.info("Computing correlation matrix…")
        try:
            corr_matrix = _compute_corr_matrix(portfolio["log_returns"])
            figures["correlation"] = plot_correlation_heatmap(
                corr_matrix,
                plotly_template=plotly_template,
            )
        except Exception as exc:
            errors["correlation"] = str(exc)
            logger.warning("Correlation heatmap failed: %s", exc)

    # ── 8. Efficient Frontier ─────────────────────────────────────────────────
    ef_result = None
    if n_assets >= 2:
        logger.info("Computing efficient frontier…")
        try:
            ef_result = _compute_ef_inline(
                portfolio["log_returns"],
                rf_annual    = rf_annual,
                n_portfolios = n_portfolios_ef,
                seed         = seed,
                cache        = cache,
            )
            figures["efficient_frontier"] = plot_efficient_frontier(
                ef_result,
                plotly_template=plotly_template,
            )
        except Exception as exc:
            errors["efficient_frontier"] = str(exc)
            logger.warning("Efficient frontier failed: %s", exc)

    # ── 9. Benchmark comparison ───────────────────────────────────────────────
    bench_data = None
    if bench_symbol.strip():
        logger.info("Fetching benchmark %s…", bench_symbol)
        try:
            bench_pd = fetch_price_series(
                bench_symbol.strip().upper(), start_date, end_date,
                cache, logger=logger,
            )
            if bench_pd is not None:
                bench_series = bench_pd["series"]
                bench_log    = compute_single_log_returns(bench_series)
                bench_data   = _compute_benchmark_inline(port_rets, bench_log)

                port_label = _ticker_label(portfolio["tickers"])
                figures["benchmark"] = plot_benchmark_comparison(
                    port_wealth    = bench_data["port_wealth"],
                    bench_wealth   = bench_data["bench_wealth"],
                    port_drawdown  = bench_data["port_drawdown"],
                    bench_drawdown = bench_data["bench_drawdown"],
                    port_label     = port_label,
                    bench_symbol   = bench_symbol.upper(),
                    port_ann_ret   = bench_data["port_ann_ret"],
                    port_ann_vol   = bench_data["port_ann_vol"],
                    port_mdd       = bench_data["port_mdd"],
                    bench_ann_ret  = bench_data["bench_ann_ret"],
                    bench_ann_vol  = bench_data["bench_ann_vol"],
                    bench_mdd      = bench_data["bench_mdd"],
                    plotly_template= plotly_template,
                )
            else:
                errors["benchmark"] = f"No data returned for benchmark '{bench_symbol}'."
        except Exception as exc:
            errors["benchmark"] = str(exc)
            logger.warning("Benchmark comparison failed: %s", exc)

    # ── 10. Fama-French factor regression ─────────────────────────────────────
    ff_result = None
    logger.info("Fetching FF factors (%s)…", ff_model)
    try:
        ff_factors = fetch_ff_factors(ff_model, cache, logger=logger)
        ff_result  = compute_factor_regression(port_rets, ff_factors, ff_model)
        tables["factor_regression"] = format_factor_table(ff_result)
        figures["factor_attribution"] = plot_factor_attribution(
            ff_result,
            plotly_template=plotly_template,
        )
    except Exception as exc:
        errors["fama_french"] = str(exc)
        logger.warning("Fama-French regression failed: %s", exc)

    # ── 11. GARCH model ───────────────────────────────────────────────────────
    garch_result = None
    logger.info("Fitting GARCH(%d,%d)…", garch_p, garch_q)
    try:
        garch_result = compute_garch_model(port_rets, p=garch_p, q=garch_q)
        tables["garch_summary"] = format_garch_table(garch_result)
        figures["garch"] = plot_garch_volatility(
            port_rets,
            garch_result,
            plotly_template=plotly_template,
        )
    except RuntimeError as exc:
        # arch package not installed — surface as a clear message
        errors["garch"] = str(exc)
        logger.warning("GARCH skipped: %s", exc)
    except Exception as exc:
        errors["garch"] = str(exc)
        logger.warning("GARCH failed: %s", exc)

    return VarAnalysisResult(
        portfolio      = portfolio,
        var_result     = var_result,
        perf_metrics   = perf_metrics,
        rolling_var_df = rolling_var_df,
        corr_matrix    = corr_matrix,
        ef_result      = ef_result,
        bench_data     = bench_data,
        ff_result      = ff_result,
        garch_result   = garch_result,
        figures        = figures,
        tables         = tables,
        errors         = errors,
    )


# ---------------------------------------------------------------------------
# Pipeline 2: Single-ticker backtest
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    strategy: str,
    strategy_params: dict,
    commission: float,
    cache: PortfolioCache,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
    logger: logging.Logger = _LOG,
) -> BacktestPipelineResult:
    """
    Execute a single vectorised backtest and return data + figure.

    Steps
    -----
    1. Resolve date range.
    2. Fetch OHLCV via cache (downloads if necessary).
    3. Filter to requested date range.
    4. Run run_backtest_core().
    5. Format the metrics comparison table.
    6. Build the 3-panel Plotly figure.

    Parameters
    ----------
    symbol          : str                   yfinance-recognised ticker.
    start_date      : datetime.date | None  Analysis start.
    end_date        : datetime.date | None  Analysis end (None → today).
    strategy        : str                   One of: 'SMA Crossover',
                                            'RSI Mean-Reversion', 'MACD',
                                            'Bollinger Band'.
    strategy_params : dict                  Strategy-specific parameters.
                                            See backtesting.StrategyParams.
    commission      : float                 Per-leg commission (decimal).
    cache           : PortfolioCache        Injected cache.
    plotly_template : str
    logger          : Logger

    Returns
    -------
    BacktestPipelineResult
        Always returns; failures are stored in `.errors` rather than raised.

    Raises
    ------
    ValueError   If the OHLCV download fails or fewer than 50 bars are
                 available in the requested date range.
    """
    errors: dict[str, str] = {}

    # ── 1. Resolve dates ──────────────────────────────────────────────────────
    start_date, end_date = resolve_date_range(start_date, end_date,
                                              fallback_years=5)

    # ── 2. Fetch OHLCV ────────────────────────────────────────────────────────
    n_years = max(int(math.ceil((end_date - start_date).days / 365)), 1)
    logger.info("Fetching OHLCV for %s (%d yr)…", symbol, n_years)
    ohlcv_data = fetch_ohlcv(symbol, n_years, cache, logger=logger)
    if ohlcv_data is None:
        raise ValueError(f"Could not retrieve OHLCV data for '{symbol}'.")

    # ── 3. Filter to requested date range ────────────────────────────────────
    df_full = ohlcv_data["df"]
    df = df_full[
        (df_full.index.date >= start_date) &
        (df_full.index.date <= end_date)
    ]
    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} trading days available for '{symbol}' "
            f"({start_date} → {end_date}).  Minimum required: 50."
        )

    # ── 4. Run backtest ───────────────────────────────────────────────────────
    logger.info("Running %s backtest on %s (%d bars)…",
                strategy, symbol, len(df))
    bt = _run_backtest_core(df, strategy, strategy_params, commission=commission)

    # ── 5. Format table ───────────────────────────────────────────────────────
    bt_table = format_backtest_table(bt, symbol)

    # ── 6. Build figure ───────────────────────────────────────────────────────
    try:
        figure = plot_backtest_results(
            bt, symbol, plotly_template=plotly_template,
        )
    except Exception as exc:
        errors["figure"] = str(exc)
        logger.warning("Backtest figure failed: %s", exc)
        figure = go.Figure()  # empty fallback

    return BacktestPipelineResult(
        bt       = bt,
        bt_table = bt_table,
        figure   = figure,
        errors   = errors,
    )


# ---------------------------------------------------------------------------
# Pipeline 3: Walk-forward validation
# ---------------------------------------------------------------------------

def run_walkforward(
    symbol: str,
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
    strategy: str,
    strategy_params: dict,
    commission: float,
    train_days: int,
    test_days: int,
    cache: PortfolioCache,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
    logger: logging.Logger = _LOG,
) -> WalkForwardPipelineResult:
    """
    Execute walk-forward out-of-sample validation and return data + figure.

    Steps
    -----
    1. Resolve date range.
    2. Fetch OHLCV (cache-backed).
    3. Filter to requested date range.
    4. Run run_walkforward_core() across all folds.
    5. Format aggregate and per-fold tables.
    6. Build the 2-panel Plotly figure.

    Parameters
    ----------
    symbol          : str
    start_date      : datetime.date | None
    end_date        : datetime.date | None
    strategy        : str                   Same strategies as run_backtest().
    strategy_params : dict
    commission      : float                 Per-leg commission (decimal).
    train_days      : int                   Indicator warm-up window length.
    test_days       : int                   OOS evaluation window length.
    cache           : PortfolioCache
    plotly_template : str
    logger          : Logger

    Returns
    -------
    WalkForwardPipelineResult

    Raises
    ------
    ValueError   If OHLCV download fails or fewer than train_days + test_days
                 bars are available.
    """
    errors: dict[str, str] = {}

    # ── 1. Resolve dates ──────────────────────────────────────────────────────
    start_date, end_date = resolve_date_range(start_date, end_date,
                                              fallback_years=5)

    # ── 2–3. Fetch and filter OHLCV ───────────────────────────────────────────
    n_years = max(int(math.ceil((end_date - start_date).days / 365)), 1)
    logger.info("Fetching OHLCV for %s (%d yr)…", symbol, n_years)
    ohlcv_data = fetch_ohlcv(symbol, n_years, cache, logger=logger)
    if ohlcv_data is None:
        raise ValueError(f"Could not retrieve OHLCV data for '{symbol}'.")

    df_full = ohlcv_data["df"]
    df = df_full[
        (df_full.index.date >= start_date) &
        (df_full.index.date <= end_date)
    ]
    if len(df) < train_days + test_days:
        raise ValueError(
            f"Only {len(df)} bars available for '{symbol}' "
            f"({start_date} → {end_date}), but train_days ({train_days}) + "
            f"test_days ({test_days}) = {train_days + test_days} are required."
        )

    # ── 4. Walk-forward ───────────────────────────────────────────────────────
    logger.info(
        "Running walk-forward: %s on %s — train %d / test %d…",
        strategy, symbol, train_days, test_days,
    )
    wf = _run_walkforward_core(
        df,
        strategy,
        strategy_params,
        train_days = train_days,
        test_days  = test_days,
        commission = commission,
    )

    # ── 5. Tables ─────────────────────────────────────────────────────────────
    agg_table  = format_walkforward_aggregate_table(wf)
    fold_table = wf["fold_df"]

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    try:
        figure = plot_walkforward_results(
            wf, symbol, plotly_template=plotly_template,
        )
    except Exception as exc:
        errors["figure"] = str(exc)
        logger.warning("Walk-forward figure failed: %s", exc)
        figure = go.Figure()

    return WalkForwardPipelineResult(
        wf         = wf,
        agg_table  = agg_table,
        fold_table = fold_table,
        figure     = figure,
        errors     = errors,
    )


# ---------------------------------------------------------------------------
# Pipeline 4: Stress testing
# ---------------------------------------------------------------------------

def run_stress_test(
    ff_result: dict,
    *,
    extra_scenarios: Optional[list[StressScenario]] = None,
    presets: tuple[StressScenario, ...] = DEFAULT_STRESS_PRESETS,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
    logger: logging.Logger = _LOG,
) -> StressTestPipelineResult:
    """
    Apply factor-shock scenarios to a portfolio's Fama-French regression
    betas and return the impact table + chart.

    The Fama-French regression (`ff_result`) must already have been computed.
    This pipeline does not re-download data or re-run the regression.

    Steps
    -----
    1. Validate that ff_result contains the required keys.
    2. Merge preset and extra_scenarios (extra prepended).
    3. Compute 1-day impact for every scenario.
    4. Build the horizontal bar chart.

    Parameters
    ----------
    ff_result       : dict                  Output of analytics.compute_factor_regression().
                                            Must have 'factor_cols', 'betas', 'model'.
    extra_scenarios : list[StressScenario]  User-defined scenarios (prepended to presets).
                                            Each must have 'name' and 'shocks' keys.
    presets         : tuple[StressScenario,...] Built-in scenario library.
                                            Defaults to DEFAULT_STRESS_PRESETS.
                                            Pass an empty tuple to suppress built-ins.
    plotly_template : str
    logger          : Logger

    Returns
    -------
    StressTestPipelineResult

    Raises
    ------
    ValueError   If ff_result is missing required keys.
    """
    errors: dict[str, str] = {}

    # ── 1. Validate ff_result ─────────────────────────────────────────────────
    for key in ("factor_cols", "betas", "model"):
        if key not in ff_result:
            raise ValueError(
                f"ff_result is missing required key '{key}'.  "
                "Run analytics.compute_factor_regression() first."
            )

    model = ff_result["model"]
    logger.info("Running stress test with %s regression betas…", model)

    # ── 2–3. Compute scenarios ────────────────────────────────────────────────
    stress = compute_stress_scenarios(
        ff_result,
        presets         = presets,
        extra_scenarios = extra_scenarios or [],
    )
    scenarios_df = stress["scenarios_df"]

    # ── 4. Figure ─────────────────────────────────────────────────────────────
    try:
        figure = plot_stress_results(
            scenarios_df,
            model,
            plotly_template=plotly_template,
        )
    except Exception as exc:
        errors["figure"] = str(exc)
        logger.warning("Stress test figure failed: %s", exc)
        figure = go.Figure()

    return StressTestPipelineResult(
        scenarios_df = scenarios_df,
        ff_model     = model,
        figure       = figure,
        errors       = errors,
    )


# ---------------------------------------------------------------------------
# Pipeline 5: Technical indicator analysis
# ---------------------------------------------------------------------------

def run_indicator_analysis(
    symbol: str,
    years: int,
    selected_indicators: frozenset[str],
    indicator_params: dict,
    cache: PortfolioCache,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
    logger: logging.Logger = _LOG,
) -> IndicatorPipelineResult:
    """
    Fetch OHLCV data, compute all selected technical indicators, and build
    the unified multi-panel Plotly chart.

    Steps
    -----
    1. Fetch OHLCV via cache.
    2. Compute all selected indicators via compute_indicator_bundle().
    3. Build the multi-panel indicator chart via plot_indicator_chart().

    Parameters
    ----------
    symbol              : str             yfinance-recognised ticker.
    years               : int             Historical look-back in calendar years.
    selected_indicators : frozenset[str]  Names of indicators to compute and display.
                                          Must be a subset of
                                          feature_engineering.ALL_INDICATOR_NAMES.
    indicator_params    : dict            IndicatorParams overrides.
                                          Missing keys fall back to each
                                          function's documented default.
    cache               : PortfolioCache  Injected cache.
    plotly_template     : str             Plotly layout template.  This is the
                                          only place user theme selection enters
                                          the pipeline; all other pipelines
                                          use the parameter default.
    logger              : Logger

    Returns
    -------
    IndicatorPipelineResult

    Raises
    ------
    ValueError   If OHLCV download fails or fewer than 14 clean rows are
                 available for the ticker.
    ValueError   If selected_indicators contains an unrecognised name.
    """
    errors: dict[str, str] = {}

    # ── 1. Fetch OHLCV ────────────────────────────────────────────────────────
    logger.info("Fetching OHLCV for %s (%d yr)…", symbol, years)
    ohlcv_data = fetch_ohlcv(symbol, years, cache, logger=logger)
    if ohlcv_data is None:
        raise ValueError(
            f"Could not retrieve OHLCV data for '{symbol}' "
            f"({years} yr look-back)."
        )
    df = ohlcv_data["df"]

    # ── 2. Compute indicator bundle ───────────────────────────────────────────
    logger.info("Computing indicator bundle for %s: %s…",
                symbol, sorted(selected_indicators))
    indicators = compute_indicator_bundle(df, selected_indicators, indicator_params)

    # ── 3. Build chart ────────────────────────────────────────────────────────
    try:
        figure = plot_indicator_chart(
            symbol          = symbol,
            ohlcv           = df,
            indicators      = indicators,
            selected        = selected_indicators,
            plotly_template = plotly_template,
        )
    except Exception as exc:
        errors["figure"] = str(exc)
        logger.warning("Indicator chart failed: %s", exc)
        figure = go.Figure()

    return IndicatorPipelineResult(
        ohlcv      = df,
        indicators = indicators,
        figure     = figure,
        errors     = errors,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ticker_label(tickers: list[str], max_shown: int = 4) -> str:
    """Compact ticker string for use in figure labels."""
    if len(tickers) <= max_shown:
        return ", ".join(tickers)
    return f"{', '.join(tickers[:max_shown])} +{len(tickers) - max_shown} more"
