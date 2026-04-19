"""
visualization.py
================
All Plotly figure builders for LazyFin.

Contract
--------
- Every public function is a pure transformation: typed data in → go.Figure out.
- No network calls, no cache access, no print/display/show calls.
- No global variable references.  Template and display config are explicit
  parameters with defaults.
- Computation that was previously inlined inside chart functions (e.g. drawdown
  computation inside plot_benchmark_comparison) has been factored into the
  analytics layer; this module only renders.

Dependency tree (imports only)
-------------------------------
  numpy, pandas, plotly   ← no aleph_toolkit imports; this file is a leaf.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "DEFAULT_PLOTLY_TEMPLATE", "DEFAULT_PLOTLY_CONFIG",
    "build_chart_config",
    "plot_var_comparison", "plot_rolling_var",
    "plot_correlation_heatmap", "plot_efficient_frontier",
    "plot_benchmark_comparison", "plot_factor_attribution",
    "plot_garch_volatility", "plot_indicator_chart",
    "plot_backtest_results", "plot_walkforward_results",
    "plot_stress_results",
]


# ---------------------------------------------------------------------------
# Module-level constants (read-only; callers may pass overrides as kwargs)
# ---------------------------------------------------------------------------

DEFAULT_PLOTLY_TEMPLATE: str = "plotly_white"

DEFAULT_PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "lazyfin_chart",
        "scale": 2,
    },
    "responsive": True,
}

# VaR method display metadata
_METHOD_LABELS: dict[str, str] = {
    "historical":    "Historical",
    "parametric":    "Parametric (Gaussian)",
    "montecarlo":    "Monte Carlo",
    "cornishfisher": "Cornish-Fisher",
}

_METHOD_COLORS: dict[str, tuple[str, str]] = {
    "historical":    ("#1565C0", "#1E88E5"),
    "parametric":    ("#6A1B9A", "#9C27B0"),
    "montecarlo":    ("#E65100", "#FF9800"),
    "cornishfisher": ("#2E7D32", "#66BB6A"),
}

# Canonical sub-panel order for the indicator chart
_INDICATOR_SUB_PANEL_ORDER: tuple[str, ...] = (
    "RSI", "ATR", "OBV", "MFI", "ADX", "Volume", "MACD",
)

# Colour palette for walk-forward folds
_WF_FOLD_PALETTE: tuple[str, ...] = (
    "#1565C0", "#2E7D32", "#6A1B9A", "#D84315",
    "#00695C", "#AD1457", "#0277BD", "#558B2F",
)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

def build_chart_config(filename: str) -> dict[str, Any]:
    """
    Return a per-chart Plotly display-config dict with a context-sensitive
    PNG download filename.

    Merges DEFAULT_PLOTLY_CONFIG with a filename override so each exported
    chart carries a meaningful name rather than the generic fallback.

    Parameters
    ----------
    filename : str
        Suggested PNG download filename, without extension.
        Plotly appends '.png' automatically.

    Returns
    -------
    dict   Merged Plotly config dict.
    """
    return {
        **DEFAULT_PLOTLY_CONFIG,
        "toImageButtonOptions": {
            **DEFAULT_PLOTLY_CONFIG["toImageButtonOptions"],
            "filename": filename,
        },
    }


# ---------------------------------------------------------------------------
# Helper: ticker label for chart titles
# ---------------------------------------------------------------------------

def _ticker_title(tickers: list[str], max_shown: int = 5) -> str:
    if len(tickers) <= max_shown:
        return ", ".join(tickers)
    return f"{', '.join(tickers[:max_shown - 1])} +{len(tickers) - max_shown + 1} more"


# ---------------------------------------------------------------------------
# 1. VaR / CVaR comparison histogram
# ---------------------------------------------------------------------------

def plot_var_comparison(
    port_rets: pd.Series,
    method_results: dict[str, dict],
    alpha: float,
    tickers: list[str],
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Plotly histogram of the portfolio return distribution overlaid with
    VaR/CVaR threshold vertical lines for every computed VaR method.

    When Monte Carlo results are present, the simulated return distribution
    is overlaid at 32% opacity for direct tail comparison.

    Parameters
    ----------
    port_rets      : pd.Series              Empirical daily portfolio log-returns.
    method_results : dict[str, dict]        Mapping of method key →
                                            {'VaR': float, 'CVaR': float, ...}
                                            e.g. from VarAnalysisResult['method_results'].
    alpha          : float                  Confidence level (e.g. 0.99).
    tickers        : list[str]              Ticker symbols for the chart title.
    plotly_template: str                    Plotly layout template.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Empirical return distribution
    fig.add_trace(go.Histogram(
        x          = port_rets,
        nbinsx     = 60,
        name       = "Historical Returns",
        opacity    = 0.75,
        marker_color = "steelblue",
        histnorm   = "probability density",
    ))

    # Monte Carlo simulated distribution overlay
    if "montecarlo" in method_results:
        mc = method_results["montecarlo"]
        if "sim_returns" in mc and mc["sim_returns"] is not None:
            fig.add_trace(go.Histogram(
                x          = mc["sim_returns"],
                nbinsx     = 80,
                name       = "MC Simulated Returns",
                opacity    = 0.32,
                marker_color = "#FF9800",
                histnorm   = "probability density",
            ))

    # VaR / CVaR threshold lines
    for method, res in method_results.items():
        var_color, cvar_color = _METHOD_COLORS.get(method, ("#000", "#444"))
        label = _METHOD_LABELS.get(method, method)
        var_val  = float(res["VaR"])
        cvar_val = float(res["CVaR"])

        fig.add_vline(
            x    = -var_val,
            line = dict(color=var_color, width=2.0, dash="dash"),
            annotation = dict(
                text      = f"{label} VaR<br>{-var_val:.4f}",
                font_size = 9,
                textangle = -90,
                showarrow = False,
                yref      = "paper",
                y         = 0.97,
            ),
        )
        fig.add_vline(
            x    = -cvar_val,
            line = dict(color=cvar_color, width=1.5, dash="dot"),
            annotation = dict(
                text      = f"{label} CVaR<br>{-cvar_val:.4f}",
                font_size = 9,
                textangle = -90,
                showarrow = False,
                yref      = "paper",
                y         = 0.60,
            ),
        )

    n_methods = len(method_results)
    fig.update_layout(
        title        = f"{_ticker_title(tickers)} — Return Distribution & VaR/CVaR ({alpha:.0%})",
        xaxis_title  = "Daily Log-Return",
        yaxis_title  = "Probability Density",
        barmode      = "overlay",
        # Height grows slightly when many methods add many vlines/annotations
        height       = max(420, 380 + 20 * n_methods),
        template     = plotly_template,
        hovermode    = "x unified",
        legend       = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=10),
        ),
        xaxis = dict(automargin=True),
        yaxis = dict(automargin=True),
        margin = dict(l=60, r=60, t=90, b=60, autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling VaR
# ---------------------------------------------------------------------------

def plot_rolling_var(
    rolling_df: pd.DataFrame,
    alpha: float,
    window: int,
    tickers: list[str],
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Two-panel interactive Plotly figure showing rolling VaR and daily returns.

    Row 1 (65%): Rolling historical and parametric VaR as line traces.
                 Area beneath historical VaR shaded at low opacity.
    Row 2 (35%): Daily portfolio log-returns as colour-coded bar chart
                 (teal = positive day, coral = negative day).
    All rows share the x-axis for synchronised pan/zoom.

    Parameters
    ----------
    rolling_df  : pd.DataFrame   Columns: 'Historical VaR', 'Parametric VaR',
                                 'Portfolio Return'.  DatetimeIndex.
                                 Output of analytics.compute_rolling_var().
    alpha       : float          Confidence level for the title.
    window      : int            Rolling window in trading days for the title.
    tickers     : list[str]      Ticker labels for the chart title.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 560 px.
    """
    ticker_str = _ticker_title(tickers)
    x_idx      = rolling_df.index

    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.65, 0.35],
        vertical_spacing = 0.04,
        subplot_titles   = (
            f"Rolling {window}-Day VaR ({alpha:.0%} confidence)",
            "Daily Portfolio Log-Returns",
        ),
    )

    # -- Row 1: shaded area beneath historical VaR (no legend entry) ----------
    fig.add_trace(go.Scatter(
        x         = x_idx,
        y         = rolling_df["Historical VaR"],
        fill      = "tozeroy",
        fillcolor = "rgba(33,150,243,0.07)",
        line      = dict(width=0),
        showlegend= False,
        hoverinfo = "skip",
        name      = "_hist_fill",
    ), row=1, col=1)

    # Historical VaR line
    fig.add_trace(go.Scatter(
        x    = x_idx,
        y    = rolling_df["Historical VaR"],
        name = "Historical VaR",
        line = dict(color="#1565C0", width=1.5),
    ), row=1, col=1)

    # Parametric VaR line
    fig.add_trace(go.Scatter(
        x    = x_idx,
        y    = rolling_df["Parametric VaR"],
        name = "Parametric VaR",
        line = dict(color="#6A1B9A", width=1.5, dash="dash"),
    ), row=1, col=1)

    fig.update_yaxes(
        title_text = f"VaR (log-return, {alpha:.0%})",
        row=1, col=1,
    )

    # -- Row 2: Daily returns bar chart ----------------------------------------
    rets   = rolling_df["Portfolio Return"]
    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in rets]

    fig.add_trace(go.Bar(
        x            = rets.index,
        y            = rets,
        marker_color = colors,
        name         = "Daily Return",
        showlegend   = True,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Log-Return", row=2, col=1)
    fig.update_xaxes(
        title_text           = "Date",
        rangeslider_visible  = False,
        row=2, col=1,
    )
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_layout(
        title    = dict(
            text     = f"{ticker_str} — Rolling {window}-Day VaR",
            font     = dict(size=13),
            automargin = True,
            pad      = dict(b=8),
        ),
        height   = 560,
        template = plotly_template,
        hovermode= "x unified",
        legend   = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=10),
            tracegroupgap = 0,
        ),
        xaxis  = dict(automargin=True),
        xaxis2 = dict(automargin=True),
        yaxis  = dict(automargin=True),
        yaxis2 = dict(automargin=True),
        margin = dict(l=70, r=50, t=90, b=60, autoexpand=True),
        bargap = 0.0,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Annotated Plotly Pearson correlation heatmap.

    Coloured from red (−1) through white (0) to blue (+1) using RdBu_r.
    Cell annotations show "+0.83" / "−0.12" formatted values; font colour
    switches to white for cells where |rho| > 0.65 for readability.
    Chart height scales with the number of assets.

    Parameters
    ----------
    corr_matrix     : pd.DataFrame   Square Pearson correlation matrix.
                                     Index and columns = ticker symbols.
                                     Output of analytics.compute_correlation_matrix().
    plotly_template : str

    Returns
    -------
    go.Figure   Height = max(350, 60*n + 140) px.
    """
    tickers     = list(corr_matrix.columns)
    n           = len(tickers)
    z           = corr_matrix.values

    # Build per-cell annotation list
    annotations = []
    for i in range(n):
        for j in range(n):
            val   = z[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            annotations.append(dict(
                x         = tickers[j],
                y         = tickers[i],
                text      = f"{val:+.2f}",
                xref      = "x",
                yref      = "y",
                showarrow = False,
                font      = dict(
                    size  = max(7, min(11, 110 // max(n, 1))),
                    color = color,
                ),
            ))

    fig = go.Figure(go.Heatmap(
        z          = z,
        x          = tickers,
        y          = tickers,
        colorscale = "RdBu_r",
        zmin       = -1.0,
        zmax       =  1.0,
        colorbar   = dict(
            title     = "Pearson ρ",
            tickvals  = [-1, -0.5, 0, 0.5, 1],
            thickness = 14,
        ),
    ))

    # Tick font size shrinks for large matrices to keep labels readable
    tick_font_size = max(7, min(11, 110 // max(n, 1)))
    # Per-cell pixel budget: shrinks for large n so the chart stays usable
    cell_px        = max(40, min(70, 560 // max(n, 1)))
    chart_height   = max(380, cell_px * n + 160)
    # Left margin must accommodate the longest ticker symbol
    max_sym_len    = max((len(t) for t in tickers), default=4)
    left_margin    = max(90, 8 * max_sym_len + 20)

    fig.update_layout(
        title       = dict(
            text       = "Pairwise Pearson Correlation of Daily Log-Returns",
            automargin = True,
            pad        = dict(b=6),
        ),
        xaxis       = dict(
            tickangle  = -45,
            tickfont   = dict(size=tick_font_size),
            automargin = True,
            side       = "bottom",
        ),
        yaxis       = dict(
            autorange  = "reversed",
            tickfont   = dict(size=tick_font_size),
            automargin = True,
        ),
        annotations = annotations,
        height      = chart_height,
        template    = plotly_template,
        margin      = dict(l=left_margin, r=90, t=80, b=max(80, 7 * max_sym_len),
                           autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Efficient Frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    ef: dict,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Interactive Markowitz efficient-frontier scatter plot.

    Traces (in order)
    -----------------
    1. Monte Carlo random-portfolio cloud — coloured by Sharpe ratio (RdYlGn).
    2. Individual assets — labelled diamond markers.
    3. Minimum-variance portfolio — blue star.
    4. Maximum-Sharpe portfolio — orange star.

    Parameters
    ----------
    ef : dict
        Output of analytics.compute_efficient_frontier().
        Required keys: mc_vols, mc_rets, mc_sharpes, min_var, max_sharpe,
        asset_vols, asset_rets, tickers, rf_annual.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 520 px.
    """
    fig = go.Figure()

    # -- Monte Carlo scatter ---------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = ef["mc_vols"] * 100,
        y    = ef["mc_rets"] * 100,
        mode = "markers",
        marker = dict(
            size       = 4,
            color      = ef["mc_sharpes"],
            colorscale = "RdYlGn",
            showscale  = True,
            colorbar   = dict(title="Sharpe Ratio", thickness=12, len=0.6, y=0.5),
            opacity    = 0.55,
        ),
        name = "Random Portfolios",
        hovertemplate = (
            "Vol: %{x:.2f}%<br>"
            "Ret: %{y:.2f}%<br>"
            "Sharpe: %{marker.color:.3f}<extra></extra>"
        ),
    ))

    # -- Individual assets -----------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = ef["asset_vols"] * 100,
        y    = ef["asset_rets"] * 100,
        mode = "markers+text",
        marker = dict(
            size   = 10,
            color  = "royalblue",
            symbol = "diamond",
            line   = dict(width=1, color="white"),
        ),
        text          = ef["tickers"],
        textposition  = "top center",
        textfont      = dict(size=10, color="royalblue"),
        name          = "Individual Assets",
        hovertemplate = (
            "<b>%{text}</b><br>"
            "Vol: %{x:.2f}%<br>"
            "Ret: %{y:.2f}%<extra></extra>"
        ),
    ))

    # -- Minimum-variance portfolio --------------------------------------------
    mv    = ef["min_var"]
    w_mv  = ", ".join(
        f"{t}:{v:.1%}"
        for t, v in zip(ef["tickers"], mv["weights"])
        if v > 0.005
    )
    fig.add_trace(go.Scatter(
        x    = [mv["ann_vol"] * 100],
        y    = [mv["ann_ret"] * 100],
        mode = "markers+text",
        marker = dict(
            size   = 16,
            color  = "#1565C0",
            symbol = "star",
            line   = dict(width=1, color="white"),
        ),
        text          = ["Min Var"],
        textposition  = "top right",
        textfont      = dict(size=11, color="#1565C0"),
        name          = f"Min Variance  (Sharpe {mv['sharpe']:.3f})",
        hovertemplate = (
            "<b>Min-Variance</b><br>"
            f"Weights: {w_mv}<br>"
            "Vol: %{x:.2f}%<br>"
            "Ret: %{y:.2f}%<extra></extra>"
        ),
    ))

    # -- Maximum-Sharpe portfolio ----------------------------------------------
    ms   = ef["max_sharpe"]
    w_ms = ", ".join(
        f"{t}:{v:.1%}"
        for t, v in zip(ef["tickers"], ms["weights"])
        if v > 0.005
    )
    fig.add_trace(go.Scatter(
        x    = [ms["ann_vol"] * 100],
        y    = [ms["ann_ret"] * 100],
        mode = "markers+text",
        marker = dict(
            size   = 16,
            color  = "#E65100",
            symbol = "star",
            line   = dict(width=1, color="white"),
        ),
        text          = ["Max Sharpe"],
        textposition  = "top right",
        textfont      = dict(size=11, color="#E65100"),
        name          = f"Max Sharpe  ({ms['sharpe']:.3f})",
        hovertemplate = (
            "<b>Max-Sharpe</b><br>"
            f"Weights: {w_ms}<br>"
            "Vol: %{x:.2f}%<br>"
            "Ret: %{y:.2f}%<extra></extra>"
        ),
    ))

    rf_label   = f" (Rf = {ef['rf_annual']:.2%})" if ef["rf_annual"] > 0 else ""
    n_assets   = len(ef["tickers"])
    # Height grows with asset count to prevent label overlap
    ef_height  = max(520, 480 + 12 * n_assets)
    # Label font shrinks for crowded charts
    label_size = max(8, min(10, 100 // max(n_assets, 1)))
    # Update asset label font retroactively on the trace
    for trace in fig.data:
        if trace.mode and "text" in trace.mode and trace.name == "Individual Assets":
            trace.textfont = dict(size=label_size, color="royalblue")

    fig.update_layout(
        title       = dict(
            text       = (
                f"Efficient Frontier{rf_label}  \u2014  "
                f"{n_assets} assets, "
                f"{len(ef['mc_vols']):,} random portfolios"
            ),
            automargin = True,
            pad        = dict(b=6),
        ),
        xaxis_title = "Annualised Volatility (%)",
        yaxis_title = "Annualised Return (%)",
        height      = ef_height,
        template    = plotly_template,
        hovermode   = "closest",
        xaxis       = dict(automargin=True, autorange=True),
        yaxis       = dict(automargin=True, autorange=True),
        legend      = dict(
            orientation  = "h",
            yanchor      = "top",
            y            = -0.12,
            xanchor      = "center",
            x            = 0.5,
            font         = dict(size=10),
            tracegroupgap= 4,
        ),
        margin = dict(l=70, r=100, t=80, b=120, autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Benchmark comparison
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(
    port_wealth: pd.Series,
    bench_wealth: pd.Series,
    port_drawdown: pd.Series,
    bench_drawdown: pd.Series,
    port_label: str,
    bench_symbol: str,
    port_ann_ret: float,
    port_ann_vol: float,
    port_mdd: float,
    bench_ann_ret: float,
    bench_ann_vol: float,
    bench_mdd: float,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Two-panel Plotly figure comparing portfolio to a benchmark.

    Row 1 (60%): Cumulative wealth index for both series (starting at $1).
    Row 2 (40%): Underwater drawdown chart (both filled below zero).
    All axes share the x-axis; pan/zoom is synchronised.

    Parameters
    ----------
    port_wealth     : pd.Series   Portfolio cumulative wealth (base = 1.0).
    bench_wealth    : pd.Series   Benchmark cumulative wealth (base = 1.0).
    port_drawdown   : pd.Series   Portfolio drawdown series (≤ 0).
    bench_drawdown  : pd.Series   Benchmark drawdown series (≤ 0).
    port_label      : str         Label for the portfolio curve.
    bench_symbol    : str         Benchmark ticker symbol.
    port_ann_ret    : float       Portfolio annualised return (decimal).
    port_ann_vol    : float       Portfolio annualised volatility (decimal).
    port_mdd        : float       Portfolio maximum drawdown (positive decimal).
    bench_ann_ret   : float       Benchmark annualised return.
    bench_ann_vol   : float       Benchmark annualised volatility.
    bench_mdd       : float       Benchmark maximum drawdown.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 560 px.
    """
    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.60, 0.40],
        vertical_spacing = 0.04,
        subplot_titles   = [
            "Cumulative Wealth Index ($1 invested)",
            "Underwater Drawdown",
        ],
    )

    # -- Row 1: wealth curves --------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = port_wealth.index,
        y    = port_wealth,
        name = (
            f"{port_label}  "
            f"(ret {port_ann_ret:+.1%} | vol {port_ann_vol:.1%} | MDD {port_mdd:.1%})"
        ),
        line = dict(color="#1565C0", width=2.0),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x    = bench_wealth.index,
        y    = bench_wealth,
        name = (
            f"{bench_symbol}  "
            f"(ret {bench_ann_ret:+.1%} | vol {bench_ann_vol:.1%} | MDD {bench_mdd:.1%})"
        ),
        line = dict(color="#E65100", width=1.8, dash="dash"),
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dot", line_color="lightgrey",
                  line_width=1, row=1, col=1)
    fig.update_yaxes(title_text="Wealth ($)", row=1, col=1)

    # -- Row 2: drawdown bands -------------------------------------------------
    fig.add_trace(go.Scatter(
        x         = port_drawdown.index,
        y         = port_drawdown * 100,
        name      = port_label,
        fill      = "tozeroy",
        fillcolor = "rgba(21,101,192,0.15)",
        line      = dict(color="#1565C0", width=1.2),
        showlegend= False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x         = bench_drawdown.index,
        y         = bench_drawdown * 100,
        name      = bench_symbol,
        fill      = "tozeroy",
        fillcolor = "rgba(230,81,0,0.12)",
        line      = dict(color="#E65100", width=1.2, dash="dash"),
        showlegend= False,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", rangeslider_visible=False, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_layout(
        title    = dict(
            text       = f"{port_label} vs {bench_symbol} \u2014 Performance Comparison",
            font       = dict(size=13),
            automargin = True,
            pad        = dict(b=6),
        ),
        height   = 580,
        template = plotly_template,
        hovermode= "x unified",
        legend   = dict(
            orientation  = "h",
            yanchor      = "top",
            y            = -0.10,
            xanchor      = "center",
            x            = 0.5,
            font         = dict(size=10),
            tracegroupgap= 4,
        ),
        xaxis  = dict(automargin=True),
        xaxis2 = dict(automargin=True),
        yaxis  = dict(automargin=True),
        yaxis2 = dict(automargin=True),
        margin = dict(l=70, r=50, t=90, b=110, autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Fama-French factor attribution
# ---------------------------------------------------------------------------

def plot_factor_attribution(
    reg: dict,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Interactive horizontal bar chart of factor loadings (betas) from an
    Fama-French regression.

    Alpha is excluded from the bars (displayed in the title instead).
    Positive betas are rendered in blue; negative in orange.
    Chart height scales with the number of factors.

    Parameters
    ----------
    reg : dict
        Output of analytics.compute_factor_regression().
        Required keys: factor_cols, betas, r_squared, adj_r_squared,
        n_obs, alpha_annual, model.
    plotly_template : str

    Returns
    -------
    go.Figure
    """
    factors = reg["factor_cols"]
    # betas[0] is alpha; betas[1:] are the factor loadings in factor_cols order
    betas   = reg["betas"][1:]
    colors  = ["#1565C0" if b >= 0 else "#E65100" for b in betas]

    fig = go.Figure(go.Bar(
        x            = betas,
        y            = factors,
        orientation  = "h",
        marker_color = colors,
        text         = [f"{b:+.3f}" for b in betas],
        textposition = "outside",
    ))

    alpha_ann = reg["alpha_annual"]
    r2        = reg["r_squared"]
    adj_r2    = reg["adj_r_squared"]
    n_obs     = reg["n_obs"]

    n_factors   = len(factors)
    # Longest factor name drives the left margin
    max_fac_len = max((len(f) for f in factors), default=6)
    left_margin = max(80, 8 * max_fac_len + 20)

    fig.update_layout(
        title = dict(
            text       = (
                f"{reg['model']} Factor Exposures  |  "
                f"\u03b1_ann = {alpha_ann:+.2%}  |  "
                f"R\u00b2 = {r2:.3f}  |  adj R\u00b2 = {adj_r2:.3f}  |  "
                f"n = {n_obs:,}"
            ),
            font       = dict(size=12),
            automargin = True,
            pad        = dict(b=6),
        ),
        xaxis_title = "Factor Loading (\u03b2)",
        yaxis_title = "Factor",
        # At least 280 px; grows 50 px per factor
        height      = max(280, 200 + 55 * n_factors),
        template    = plotly_template,
        xaxis       = dict(
            zeroline      = True,
            zerolinewidth = 1.5,
            zerolinecolor = "lightgrey",
            automargin    = True,
            autorange     = True,
        ),
        yaxis       = dict(
            automargin = True,
            autorange  = True,
        ),
        margin = dict(l=left_margin, r=110, t=80, b=60, autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. GARCH volatility model
# ---------------------------------------------------------------------------

def plot_garch_volatility(
    port_rets: pd.Series,
    garch: dict,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Two-panel Plotly GARCH volatility figure.

    Row 1 (55%): Annualised GARCH conditional volatility vs 21-day rolling std.
                 A 22-day ahead forecast extension is appended as a dotted line.
    Row 2 (45%): Daily log-returns with ±2σ GARCH confidence bands (shaded).
    Both rows share the x-axis.

    Parameters
    ----------
    port_rets       : pd.Series   Daily portfolio log-returns (DatetimeIndex).
                                  Used for the return series and for deriving
                                  the rolling-std reference line.
    garch           : dict        Output of analytics.compute_garch_model().
                                  Required keys: cond_vol, fcast_vol, p, q,
                                  persistence, long_run_vol_annual.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 560 px.
    """
    cond_vol    = garch["cond_vol"]           # pd.Series, decimal daily vol
    fcast_vol   = garch["fcast_vol"]          # np.ndarray, 22 decimal daily vals
    p, q        = garch["p"], garch["q"]

    # Annualise
    ann_cond    = cond_vol * np.sqrt(252)
    rolling_vol = port_rets.rolling(21).std() * np.sqrt(252)

    # Build 22-day forecast date index (business days)
    last_date   = cond_vol.index[-1]
    fcast_dates = pd.bdate_range(start=last_date, periods=23)[1:]
    fcast_ann   = fcast_vol * np.sqrt(252)

    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.55, 0.45],
        vertical_spacing = 0.04,
        subplot_titles   = [
            f"GARCH({p},{q}) Conditional Volatility vs Rolling 21-day Std (Annualised)",
            "Daily Log-Returns with \u00b12\u03c3 GARCH Bands",
        ],
    )

    # -- Row 1: conditional vol ------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = ann_cond.index,
        y    = ann_cond * 100,
        name = f"GARCH({p},{q}) \u03c3",
        line = dict(color="#1565C0", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x       = rolling_vol.index,
        y       = rolling_vol * 100,
        name    = "Rolling 21-day \u03c3",
        line    = dict(color="#E65100", width=1.2, dash="dash"),
        opacity = 0.75,
    ), row=1, col=1)

    # 22-day forecast extension
    if len(fcast_dates) == len(fcast_ann):
        fig.add_trace(go.Scatter(
            x    = fcast_dates,
            y    = fcast_ann * 100,
            name = "22-day Forecast",
            line = dict(color="#1565C0", width=1.5, dash="dot"),
            mode = "lines",
        ), row=1, col=1)

    fig.update_yaxes(title_text="Volatility (% ann.)", row=1, col=1)

    # -- Row 2: returns with ±2σ bands -----------------------------------------
    upper_band = 2 * cond_vol
    lower_band = -2 * cond_vol

    fig.add_trace(go.Scatter(
        x         = upper_band.index,
        y         = upper_band,
        fill      = None,
        line      = dict(width=0),
        showlegend= False,
        hoverinfo = "skip",
        name      = "_ub",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x         = lower_band.index,
        y         = lower_band,
        fill      = "tonexty",
        fillcolor = "rgba(21,101,192,0.10)",
        line      = dict(width=0),
        name      = "\u00b12\u03c3 GARCH band",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x    = port_rets.index,
        y    = port_rets,
        name = "Daily Return",
        line = dict(color="#424242", width=0.8),
    ), row=2, col=1)

    fig.update_yaxes(title_text="Log-Return", row=2, col=1)

    pers    = garch["persistence"]
    lr_ann  = garch["long_run_vol_annual"]
    lr_str  = (
        f"LR \u03c3 = {lr_ann * 100:.1f}% ann."
        if not np.isnan(lr_ann)
        else "non-stationary"
    )

    fig.update_layout(
        title    = dict(
            text       = (
                f"GARCH({p},{q}) Volatility Model  |  "
                f"Persistence = {pers:.4f}  |  {lr_str}"
            ),
            font       = dict(size=12),
            automargin = True,
            pad        = dict(b=6),
        ),
        height   = 580,
        template = plotly_template,
        hovermode= "x unified",
        legend   = dict(
            orientation  = "h",
            yanchor      = "top",
            y            = -0.08,
            xanchor      = "center",
            x            = 0.5,
            font         = dict(size=10),
            tracegroupgap= 4,
        ),
        xaxis  = dict(automargin=True),
        xaxis2 = dict(automargin=True),
        yaxis  = dict(automargin=True),
        yaxis2 = dict(automargin=True),
        margin = dict(l=70, r=50, t=90, b=100, autoexpand=True),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Technical indicator chart
# ---------------------------------------------------------------------------

def plot_indicator_chart(
    symbol: str,
    ohlcv: pd.DataFrame,
    indicators: dict,
    selected: frozenset[str],
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Build the unified multi-panel interactive Plotly indicator chart.

    Layout
    ------
    Row 1 (always, 42% height): Candlestick price chart with configurable overlays.
    Sub-panel rows (1 per selected sub-panel, in fixed order):
        RSI → ATR → OBV → MFI → ADX → Volume → MACD

    All rows share the x-axis for synchronised pan/zoom.

    Parameters
    ----------
    symbol          : str            Ticker symbol for the title.
    ohlcv           : pd.DataFrame   Validated OHLCV DataFrame (DatetimeIndex).
                                     Required columns: Open, High, Low, Close, Volume.
    indicators      : dict           Output of feature_engineering.compute_indicator_bundle().
                                     Keys map indicator names to their computed values.
    selected        : frozenset[str] Indicator names to render.
    plotly_template : str            Overrides the chart theme.

    Returns
    -------
    go.Figure   Height = 380 + 180 * n_sub_panels px.
    """
    high  = ohlcv["High"].squeeze()
    low   = ohlcv["Low"].squeeze()
    close = ohlcv["Close"].squeeze()
    open_ = ohlcv["Open"].squeeze()
    vol   = ohlcv["Volume"].squeeze()

    # Determine sub-panel rows in canonical order
    sub_panels = [p for p in _INDICATOR_SUB_PANEL_ORDER if p in selected]
    n_rows     = 1 + len(sub_panels)

    if sub_panels:
        price_frac  = 0.42
        sub_frac    = round((1.0 - price_frac) / len(sub_panels), 4)
        row_heights = [price_frac] + [sub_frac] * len(sub_panels)
    else:
        row_heights = [1.0]

    fig = make_subplots(
        rows             = n_rows,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = row_heights,
        vertical_spacing = 0.03,
        subplot_titles   = [f"{symbol} \u2014 Price"] + sub_panels,
    )

    # -- Row 1: Candlestick ----------------------------------------------------
    fig.add_trace(go.Candlestick(
        x                     = ohlcv.index,
        open                  = ohlcv["Open"],
        high                  = ohlcv["High"],
        low                   = ohlcv["Low"],
        close                 = ohlcv["Close"],
        name                  = "Price",
        increasing_line_color = "#26a69a",
        decreasing_line_color = "#ef5350",
        showlegend            = True,
    ), row=1, col=1)

    # -- Row 1 overlay: Bollinger Bands ----------------------------------------
    if "Bollinger Bands" in selected and "bollinger" in indicators:
        bb = indicators["bollinger"]
        # Upper — legendonly by default so chart opens clean
        fig.add_trace(go.Scatter(
            x       = ohlcv.index,
            y       = bb["upper"],
            name    = "BB Upper",
            line    = dict(color="rgba(100,149,237,0.75)", width=1.0, dash="dot"),
            visible = "legendonly",
        ), row=1, col=1)
        # Mid — always visible (primary trend anchor)
        fig.add_trace(go.Scatter(
            x    = ohlcv.index,
            y    = bb["mid"],
            name = "BB Mid (SMA)",
            line = dict(color="rgba(148,0,211,0.80)", width=1.2, dash="dash"),
        ), row=1, col=1)
        # Invisible anchor for fill
        fig.add_trace(go.Scatter(
            x          = ohlcv.index,
            y          = bb["upper"],
            fill       = None,
            line       = dict(width=0),
            showlegend = False,
            hoverinfo  = "skip",
            name       = "_bb_anchor",
        ), row=1, col=1)
        # Lower — legendonly; fills to anchor
        fig.add_trace(go.Scatter(
            x         = ohlcv.index,
            y         = bb["lower"],
            name      = "BB Lower",
            line      = dict(color="rgba(100,149,237,0.75)", width=1.0, dash="dot"),
            fill      = "tonexty",
            fillcolor = "rgba(100,149,237,0.06)",
            visible   = "legendonly",
        ), row=1, col=1)

    # -- Row 1 overlay: DEMA ---------------------------------------------------
    if "DEMA" in selected and "dema" in indicators:
        fig.add_trace(go.Scatter(
            x    = ohlcv.index,
            y    = indicators["dema"],
            name = "DEMA",
            line = dict(color="darkorange", width=1.5),
        ), row=1, col=1)

    # -- Row 1 overlay: Parabolic SAR ------------------------------------------
    if "Parabolic SAR" in selected and "psar" in indicators:
        fig.add_trace(go.Scatter(
            x      = ohlcv.index,
            y      = indicators["psar"],
            name   = "Parabolic SAR",
            mode   = "markers",
            marker = dict(symbol="circle", size=4, color="orange", opacity=0.85),
        ), row=1, col=1)

    # -- Sub-panel rows --------------------------------------------------------
    for row_idx, panel in enumerate(sub_panels, start=2):

        if panel == "RSI" and "rsi" in indicators:
            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = indicators["rsi"],
                name = "RSI",
                line = dict(color="mediumpurple", width=1.2),
            ), row=row_idx, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="crimson",
                          line_width=0.8, row=row_idx, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="forestgreen",
                          line_width=0.8, row=row_idx, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=row_idx, col=1)

        elif panel == "ATR" and "atr" in indicators:
            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = indicators["atr"],
                name = "ATR",
                line = dict(color="teal", width=1.2),
            ), row=row_idx, col=1)
            fig.update_yaxes(title_text="ATR", row=row_idx, col=1)

        elif panel == "OBV" and "obv" in indicators:
            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = indicators["obv"],
                name = "OBV",
                line = dict(color="steelblue", width=1.0),
            ), row=row_idx, col=1)
            fig.update_yaxes(title_text="OBV", row=row_idx, col=1)

        elif panel == "MFI" and "mfi" in indicators:
            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = indicators["mfi"],
                name = "MFI",
                line = dict(color="darkcyan", width=1.2),
            ), row=row_idx, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="crimson",
                          line_width=0.8, row=row_idx, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="forestgreen",
                          line_width=0.8, row=row_idx, col=1)
            fig.update_yaxes(title_text="MFI", range=[0, 100], row=row_idx, col=1)

        elif panel == "ADX" and "adx" in indicators:
            adx_d = indicators["adx"]
            # +DI and -DI start legendonly; ADX line always visible
            fig.add_trace(go.Scatter(
                x       = ohlcv.index,
                y       = adx_d["plus_di"],
                name    = "+DI",
                line    = dict(color="mediumseagreen", width=1.1),
                visible = "legendonly",
            ), row=row_idx, col=1)
            fig.add_trace(go.Scatter(
                x       = ohlcv.index,
                y       = adx_d["minus_di"],
                name    = "-DI",
                line    = dict(color="tomato", width=1.1),
                visible = "legendonly",
            ), row=row_idx, col=1)
            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = adx_d["adx"],
                name = "ADX",
                line = dict(color="#222222", width=2.0),
            ), row=row_idx, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="slategray",
                          line_width=0.8, row=row_idx, col=1)
            fig.update_yaxes(title_text="ADX / DI", row=row_idx, col=1)

        elif panel == "Volume":
            raw_vol = indicators.get("volume", vol)
            colors  = np.where(close >= open_, "#26a69a", "#ef5350")
            fig.add_trace(go.Bar(
                x                 = ohlcv.index,
                y                 = raw_vol,
                marker_color      = colors,
                marker_line_width = 0,
                name              = "Volume",
                showlegend        = True,
            ), row=row_idx, col=1)
            fig.update_yaxes(title_text="Volume", row=row_idx, col=1)

        elif panel == "MACD" and "macd" in indicators:
            macd_d       = indicators["macd"]
            macd_line    = macd_d["macd_line"]
            signal_line  = macd_d["signal_line"]
            histogram    = macd_d["histogram"]
            hist_colors  = np.where(histogram >= 0, "#26a69a", "#ef5350")

            fig.add_trace(go.Bar(
                x                 = ohlcv.index,
                y                 = histogram,
                name              = "MACD Histogram",
                marker_color      = hist_colors,
                marker_line_width = 0,
                opacity           = 0.65,
            ), row=row_idx, col=1)

            fig.add_trace(go.Scatter(
                x    = ohlcv.index,
                y    = macd_line,
                name = "MACD",
                line = dict(color="#1565C0", width=1.3),
            ), row=row_idx, col=1)

            fig.add_trace(go.Scatter(
                x       = ohlcv.index,
                y       = signal_line,
                name    = "Signal",
                line    = dict(color="#E65100", width=1.3, dash="dash"),
                visible = "legendonly",
            ), row=row_idx, col=1)

            fig.add_hline(y=0, line_dash="dot", line_color="lightgrey",
                          line_width=0.8, row=row_idx, col=1)
            fig.update_yaxes(title_text="MACD", row=row_idx, col=1)

    # -- Global layout ---------------------------------------------------------
    total_height = 380 + 180 * len(sub_panels)
    for r in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, rangeslider_visible=False,
                         row=r, col=1)
    fig.update_xaxes(title_text="Date", rangeslider_visible=False,
                     row=n_rows, col=1)

    fig.update_layout(
        title    = dict(
            text = f"{symbol} \u2014 Technical Indicators",
            font = dict(size=14),
        ),
        height   = total_height,
        template = plotly_template,
        hovermode= "x unified",
        legend   = dict(
            orientation     = "h",
            yanchor         = "bottom",
            y               = 1.01,
            xanchor         = "right",
            x               = 1,
            font            = dict(size=10),
            itemclick       = "toggle",
            itemdoubleclick = "toggleothers",
        ),
        margin = dict(l=60, r=40, t=80, b=50),
        bargap = 0.0,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Backtest results
# ---------------------------------------------------------------------------

def plot_backtest_results(
    bt: dict,
    symbol: str,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Three-panel interactive backtest figure.

    Row 1 (40%): Close price with entry (▲) and exit (▼) trade markers.
    Row 2 (35%): Equity curves — strategy vs buy-and-hold ($1 base).
    Row 3 (25%): Underwater drawdown — strategy vs buy-and-hold.
    All rows share the x-axis.

    Parameters
    ----------
    bt   : dict   Output of backtesting.run_backtest().
                  Required keys: close, strat_equity, bnh_equity, drawdown,
                  entries, exits, strategy.
    symbol          : str   Ticker symbol for the chart title.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 660 px.
    """
    strategy  = bt["strategy"]
    close     = bt["close"]
    s_eq      = bt["strat_equity"]
    bnh_eq    = bt["bnh_equity"]
    s_dd      = bt["drawdown"]
    entries   = bt["entries"]
    exits     = bt["exits"]

    # Buy-and-hold drawdown (computed here for display; source is bt["log_rets"])
    bnh_peak = bnh_eq.cummax()
    bnh_dd   = (bnh_eq - bnh_peak) / bnh_peak

    fig = make_subplots(
        rows             = 3,
        cols             = 1,
        shared_xaxes     = True,
        row_heights      = [0.40, 0.35, 0.25],
        vertical_spacing = 0.03,
        subplot_titles   = [
            f"{symbol} — Price & Trade Signals",
            f"Equity Curve: {strategy} vs Buy & Hold",
            "Underwater Drawdown",
        ],
    )

    # -- Row 1: close price ----------------------------------------------------
    fig.add_trace(go.Scatter(
        x          = close.index,
        y          = close,
        name       = "Close Price",
        line       = dict(color="#424242", width=1.0),
        showlegend = True,
    ), row=1, col=1)

    if entries.sum() > 0:
        fig.add_trace(go.Scatter(
            x      = close.index[entries],
            y      = close[entries],
            mode   = "markers",
            name   = "Entry (Long)",
            marker = dict(
                symbol = "triangle-up",
                size   = 9,
                color  = "#26a69a",
                line   = dict(width=1, color="white"),
            ),
        ), row=1, col=1)

    if exits.sum() > 0:
        fig.add_trace(go.Scatter(
            x      = close.index[exits],
            y      = close[exits],
            mode   = "markers",
            name   = "Exit (Flat)",
            marker = dict(
                symbol = "triangle-down",
                size   = 9,
                color  = "#ef5350",
                line   = dict(width=1, color="white"),
            ),
        ), row=1, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)

    # -- Row 2: equity curves --------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = s_eq.index,
        y    = s_eq,
        name = strategy,
        line = dict(color="#1565C0", width=2.0),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x    = bnh_eq.index,
        y    = bnh_eq,
        name = "Buy & Hold",
        line = dict(color="#E65100", width=1.5, dash="dash"),
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash="dot", line_color="lightgrey",
                  line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="Wealth ($)", row=2, col=1)

    # -- Row 3: drawdown -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x         = s_dd.index,
        y         = s_dd * 100,
        name      = f"{strategy} DD",
        fill      = "tozeroy",
        fillcolor = "rgba(21,101,192,0.15)",
        line      = dict(color="#1565C0", width=1.0),
        showlegend= False,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x         = bnh_dd.index,
        y         = bnh_dd * 100,
        name      = "B&H DD",
        fill      = "tozeroy",
        fillcolor = "rgba(230,81,0,0.10)",
        line      = dict(color="#E65100", width=1.0, dash="dash"),
        showlegend= False,
    ), row=3, col=1)

    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    for r in range(1, 3):
        fig.update_xaxes(rangeslider_visible=False, row=r, col=1)
    fig.update_xaxes(title_text="Date", rangeslider_visible=False, row=3, col=1)

    fig.update_layout(
        title    = dict(
            text = f"{symbol} — {strategy} Backtest",
            font = dict(size=13),
        ),
        height   = 660,
        template = plotly_template,
        hovermode= "x unified",
        legend   = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=10),
        ),
        margin = dict(l=60, r=40, t=90, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Walk-forward results
# ---------------------------------------------------------------------------

def plot_walkforward_results(
    wf: dict,
    symbol: str,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Two-panel walk-forward validation results figure.

    Row 1 (60%): Per-fold strategy equity curves on a shared $1 base,
                 coloured by fold, with B&H equity as a thinner dotted
                 reference for each fold.
    Row 2 (40%): Per-fold total return bar chart (strategy vs B&H),
                 annotated with per-fold Sharpe ratios above each bar.

    Parameters
    ----------
    wf     : dict   Output of backtesting.run_walkforward().
                    Required keys: folds, strategy, n_folds, train_days,
                    test_days, agg.
    symbol          : str   Ticker symbol for the title.
    plotly_template : str

    Returns
    -------
    go.Figure   Height = 640 px.
    """
    folds    = wf["folds"]
    strategy = wf["strategy"]
    n_folds  = wf["n_folds"]

    fig = make_subplots(
        rows             = 2,
        cols             = 1,
        shared_xaxes     = False,
        row_heights      = [0.60, 0.40],
        vertical_spacing = 0.12,
        subplot_titles   = [
            (
                f"{symbol} — Per-Fold Equity Curves "
                f"(train {wf['train_days']}d / test {wf['test_days']}d)"
            ),
            "Per-Fold Total Return: Strategy vs Buy & Hold",
        ],
    )

    # -- Row 1: per-fold equity curves -----------------------------------------
    for k, fold in enumerate(folds):
        col   = _WF_FOLD_PALETTE[k % len(_WF_FOLD_PALETTE)]
        label = (
            f"Fold {fold['fold']} "
            f"({fold['test_start']} \u2013 {fold['test_end']})"
        )
        idx = fold["strat_equity"].index

        # Renormalise each fold to $1 at fold start
        s_eq = fold["strat_equity"] / fold["strat_equity"].iloc[0]
        b_eq = fold["bnh_equity"]   / fold["bnh_equity"].iloc[0]

        fig.add_trace(go.Scatter(
            x    = idx,
            y    = s_eq,
            name = label,
            line = dict(color=col, width=1.6),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x          = idx,
            y          = b_eq,
            name       = f"B&H Fold {fold['fold']}",
            line       = dict(color=col, width=0.9, dash="dot"),
            showlegend = False,
        ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="lightgrey",
                  line_width=1, row=1, col=1)
    fig.update_yaxes(title_text="Normalised Wealth ($1 base)", row=1, col=1)

    # -- Row 2: per-fold total return bars -------------------------------------
    fold_labels   = [f"F{f['fold']}" for f in folds]
    strat_returns = [f["total_ret"]     * 100 for f in folds]
    bnh_returns   = [f["bnh_total_ret"] * 100 for f in folds]
    sharpes       = [f["sharpe"]               for f in folds]

    strat_colors  = ["#ef5350" if r < 0 else "#26a69a" for r in strat_returns]

    fig.add_trace(go.Bar(
        x            = fold_labels,
        y            = strat_returns,
        name         = strategy,
        marker_color = strat_colors,
        text         = [f"{s:.2f}" if not np.isnan(s) else "" for s in sharpes],
        textposition = "outside",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x            = fold_labels,
        y            = bnh_returns,
        name         = "Buy & Hold",
        marker_color = "rgba(100,100,200,0.45)",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Fold", row=2, col=1)

    agg = wf["agg"]
    fig.update_layout(
        title = dict(
            text = (
                f"{symbol} — {strategy}  Walk-Forward  |  "
                f"{n_folds} folds  |  "
                f"Mean Sharpe {agg['mean_sharpe']:.2f}  |  "
                f"Mean Return {agg['mean_total_ret']:+.1%}"
            ),
            font = dict(size=12),
        ),
        height   = 640,
        template = plotly_template,
        hovermode= "x unified",
        barmode  = "group",
        legend   = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
            font        = dict(size=10),
        ),
        margin = dict(l=60, r=40, t=90, b=55),
    )
    return fig


# ---------------------------------------------------------------------------
# 11. Stress test results
# ---------------------------------------------------------------------------

def plot_stress_results(
    scenarios_df: pd.DataFrame,
    model: str,
    *,
    plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
) -> go.Figure:
    """
    Horizontal bar chart of stress-test scenario estimated portfolio impacts.

    Bars are coloured by sign: coral (negative impact) / teal (positive).
    Each bar is annotated with the formatted impact percentage.
    A vertical reference line is drawn at x = 0.
    The chart is ordered descending by absolute impact (matching the
    DataFrame row order produced by compute_stress_scenarios()).

    Parameters
    ----------
    scenarios_df    : pd.DataFrame   Output of stress_testing.compute_stress_scenarios()
                                     ['scenarios_df'].  Required columns:
                                     'Scenario', 'Est. 1-Day Impact', 'Factor Shocks',
                                     'Impact %'.
    model           : str            FF model name (for the title).
    plotly_template : str

    Returns
    -------
    go.Figure   Height = max(350, 60 + 50 * n_scenarios) px.
    """
    impacts = scenarios_df["Est. 1-Day Impact"].values.astype(float)
    colors  = ["#ef5350" if v < 0 else "#26a69a" for v in impacts]

    fig = go.Figure(go.Bar(
        x             = impacts * 100,
        y             = scenarios_df["Scenario"],
        orientation   = "h",
        marker_color  = colors,
        text          = scenarios_df["Impact %"],
        textposition  = "outside",
        hovertext     = scenarios_df["Factor Shocks"],
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Impact: %{x:.3f}%<br>"
            "Shocks: %{hovertext}<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="lightgrey", line_dash="dot")

    fig.update_layout(
        title = dict(
            text = f"{model} Stress Test — Estimated 1-Day Portfolio Impact",
            font = dict(size=13),
        ),
        xaxis_title = "Estimated Portfolio Impact (%)",
        yaxis       = dict(autorange="reversed"),
        height      = max(350, 60 + 50 * len(scenarios_df)),
        template    = plotly_template,
        margin      = dict(l=220, r=100, t=70, b=55),
        hovermode   = "closest",
    )
    return fig
