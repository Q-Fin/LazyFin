"""
callbacks.py
============
All Dash callback functions for LazyFin.

Structure
---------
  1.  Shared helpers
  2.  Date-preset callbacks  (VaR + Backtesting)
  3.  Strategy param visibility
  4.  VaR / CVaR tab  -- single pipeline call, 18 outputs
  5.  VaR cache + export callbacks
  6.  Indicators tab
  7.  Backtesting tab
  8.  Walk-Forward tab
  9.  Stress Testing tab
  10. register_callbacks() -- public entry point

Design rules
------------
- ONE pipeline call per analysis button.  No duplicate runs.
- Each @callback targets ONLY its own tab's Output IDs.
- No global state.  PortfolioCache injected via closure.
- background=True is omitted -- it requires extra system
  dependencies not listed in requirements.txt.
"""

from __future__ import annotations

import datetime
import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import dash_table                    # Dash < 2.0
except ModuleNotFoundError:
    from dash import dash_table          # Dash 2+

from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from lazyfin import pipeline as pl
from lazyfin.stress_testing import parse_custom_scenario


# ============================================================
# 1. Shared helpers
# ============================================================

def _df_to_datatable(df: pd.DataFrame, table_id: str,
                     page_size: int = 20) -> dash_table.DataTable:
    df_r = df.reset_index()
    return dash_table.DataTable(
        id=table_id,
        data=df_r.to_dict("records"),
        columns=[{"name": str(c), "id": str(c)} for c in df_r.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#e8f0fe", "color": "#1565c0",
            "fontWeight": "bold", "fontSize": "0.75rem",
            "textTransform": "uppercase", "letterSpacing": "0.3px",
            "border": "1px solid #c5cae9",
        },
        style_cell={
            "fontSize": "0.78rem", "padding": "5px 10px",
            "textAlign": "left", "border": "1px solid #e8ecef",
            "whiteSpace": "normal",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9ff"}
        ],
        style_cell_conditional=[
            {"if": {"column_id": df_r.columns[0]},
             "fontWeight": "600", "color": "#37474f"},
        ],
    )


def _ok_alert(msg: str) -> tuple:
    return msg, "success", True


def _warn_alert(msg: str) -> tuple:
    return msg, "warning", True


def _err_alert(msg: str) -> tuple:
    return msg, "danger", True


def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#78909c"))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis={"visible": False}, yaxis={"visible": False},
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip().upper()
            for t in raw.replace(",", " ").split() if t.strip()]


def _parse_date(s: str | None) -> datetime.date | None:
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _parse_weights(raw: str | None) -> list[float] | None:
    if not raw or not raw.strip():
        return None
    try:
        vals = [float(x.strip()) for x in raw.replace(";", ",").split(",")
                if x.strip()]
        if not vals or any(v < 0 for v in vals) or sum(vals) == 0:
            return None
        return vals
    except ValueError:
        return None


def _ff_store_encode(ff_result: dict) -> dict:
    return {
        "factor_cols":   ff_result["factor_cols"],
        "betas":         ff_result["betas"].tolist(),
        "se":            ff_result["se"].tolist(),
        "t_stats":       ff_result["t_stats"].tolist(),
        "p_values":      ff_result["p_values"].tolist(),
        "model":         ff_result["model"],
        "param_names":   ff_result["param_names"],
        "r_squared":     ff_result["r_squared"],
        "adj_r_squared": ff_result["adj_r_squared"],
        "n_obs":         ff_result["n_obs"],
        "alpha_annual":  ff_result["alpha_annual"],
    }


def _ff_store_decode(data: dict) -> dict:
    return {
        **data,
        "betas":   np.array(data["betas"]),
        "se":      np.array(data["se"]),
        "t_stats": np.array(data["t_stats"]),
        "p_values":np.array(data["p_values"]),
    }


def _errors_summary(errors: dict) -> str:
    if not errors:
        return ""
    return "  Skipped: " + "; ".join(f"{k}: {v[:60]}" for k, v in errors.items())


# ============================================================
# 2. Date preset callbacks
# ============================================================

def _register_date_presets(prefix: str) -> None:
    @callback(
        Output(f"{prefix}-daterange", "start_date"),
        Output(f"{prefix}-daterange", "end_date"),
        Input({"type": f"{prefix}-preset", "days": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def apply_preset(n_clicks_list):
        if not any(n for n in n_clicks_list if n):
            return no_update, no_update
        triggered = ctx.triggered_id
        if triggered is None:
            return no_update, no_update
        today = datetime.date.today()
        start = today - datetime.timedelta(days=int(triggered["days"]))
        return start.isoformat(), today.isoformat()


# ============================================================
# 3. Strategy param visibility
# ============================================================

def _register_strategy_visibility() -> None:
    @callback(
        Output("bt-sma-params",  "style"),
        Output("bt-rsi-params",  "style"),
        Output("bt-macd-params", "style"),
        Output("bt-bb-params",   "style"),
        Input("bt-strategy-dropdown", "value"),
    )
    def toggle_strategy_params(strategy):
        show = {"display": "block"}
        hide = {"display": "none"}
        for s in ["SMA Crossover", "RSI Mean-Reversion", "MACD", "Bollinger Band"]:
            pass  # just to define the list
        strategies = ["SMA Crossover", "RSI Mean-Reversion", "MACD", "Bollinger Band"]
        return tuple(show if s == strategy else hide for s in strategies)


# ============================================================
# 4. VaR / CVaR tab -- 18 outputs, 14 states, single pipeline call
# ============================================================

def _register_var_callbacks(cache: Any) -> None:

    @callback(
        # [00-06] 7 charts
        Output("var-dist-graph",            "figure"),
        Output("var-rolling-graph",         "figure"),
        Output("var-corr-graph",            "figure"),
        Output("var-ef-graph",              "figure"),
        Output("var-bench-graph",           "figure"),
        Output("var-ff-graph",              "figure"),
        Output("var-garch-graph",           "figure"),
        # [07-11] 5 table containers
        Output("var-summary-table",         "children"),
        Output("var-perf-table",            "children"),
        Output("var-cf-container",          "children"),
        Output("var-ff-table-container",    "children"),
        Output("var-garch-table-container", "children"),
        # [12-14] stores + run-info
        Output("store-ff-result",           "data"),
        Output("store-var-result",          "data"),
        Output("var-run-info",              "children"),
        # [15-17] status alert
        Output("var-status",                "children"),
        Output("var-status",                "color"),
        Output("var-status",                "is_open"),
        # trigger
        Input("var-run-btn", "n_clicks"),
        # 14 states
        State("var-tickers-input",    "value"),
        State("var-daterange",        "start_date"),
        State("var-daterange",        "end_date"),
        State("var-alpha-slider",     "value"),
        State("var-methods-checklist","value"),
        State("var-weights-input",    "value"),
        State("var-rf-input",         "value"),
        State("var-rolling-slider",   "value"),
        State("var-bench-input",      "value"),
        State("var-ff-radio",         "value"),
        State("var-garch-p",          "value"),
        State("var-garch-q",          "value"),
        State("var-nsims-slider",     "value"),
        State("var-theme-dropdown",   "value"),
        prevent_initial_call=True,
    )
    def run_var(
        _n,
        raw_tickers, start_str, end_str,
        alpha, methods, raw_weights, rf_annual,
        rolling_window, bench_symbol, ff_model,
        garch_p, garch_q, n_sims, theme,
    ):
        # Default "no data" tuple: 7 figs + 5 tables + ff_store + var_store + run_info
        _empty7 = tuple(_empty_fig() for _ in range(7))
        _notbl5 = tuple(html.Div() for _ in range(5))
        _none15 = (*_empty7, *_notbl5, None, None, "")

        tickers = _parse_tickers(raw_tickers)
        if not tickers:
            return (*_none15, *_err_alert("Enter at least one ticker symbol."))
        if not methods:
            return (*_none15, *_err_alert("Select at least one VaR method."))

        try:
            result = pl.run_var_analysis(
                tickers         = tickers,
                start_date      = _parse_date(start_str),
                end_date        = _parse_date(end_str),
                alpha           = float(alpha or 0.99),
                weights         = _parse_weights(raw_weights),
                methods         = list(methods),
                rf_annual       = float(rf_annual or 0.0),
                rolling_window  = int(rolling_window or 252),
                bench_symbol    = (bench_symbol or "").strip(),
                ff_model        = ff_model or "FF3",
                garch_p         = int(garch_p or 1),
                garch_q         = int(garch_q or 1),
                cache           = cache,
                n_sims          = int(n_sims or 10_000),
                plotly_template = theme or "plotly_white",
            )
        except ValueError as exc:
            return (*_none15, *_err_alert(f"Analysis failed: {exc}"))
        except Exception as exc:
            return (*_none15, *_err_alert(f"Unexpected error: {exc}"))

        figs   = result.figures
        tables = result.tables
        errors = result.errors

        # Charts
        dist_fig  = figs.get("var_comparison",     _empty_fig("VaR distribution unavailable"))
        roll_fig  = figs.get("rolling_var",        _empty_fig("Insufficient data for rolling VaR"))
        corr_fig  = figs.get("correlation",        _empty_fig("Correlation requires >=2 tickers"))
        ef_fig    = figs.get("efficient_frontier", _empty_fig("Efficient frontier requires >=2 tickers"))
        bench_fig = figs.get("benchmark",          _empty_fig("Benchmark unavailable"))
        ff_fig    = figs.get("factor_attribution", _empty_fig("Fama-French unavailable"))
        garch_fig = figs.get("garch",              _empty_fig("GARCH unavailable -- pip install arch"))

        # Tables
        def _tbl(key: str, tbl_id: str) -> Any:
            if key in tables and not tables[key].empty:
                return _df_to_datatable(tables[key], tbl_id)
            return html.Div(style={"display": "none"})

        summary_tbl = _tbl("var_summary",      "var-summary-dt")
        perf_tbl    = _tbl("performance",       "var-perf-dt")
        cf_tbl      = (_tbl("cf_diagnostics",  "var-cf-dt")
                       if "cf_diagnostics" in tables else html.Div())
        ff_tbl      = _tbl("factor_regression", "var-ff-dt")
        garch_tbl   = _tbl("garch_summary",    "var-garch-dt")

        # FF store (stress testing needs it)
        ff_store = None
        if result.ff_result is not None:
            try:
                ff_store = _ff_store_encode(result.ff_result)
            except Exception:
                pass

        # var store (CSV export)
        var_store = None
        try:
            port = result.portfolio
            var_store = {
                "prices_json":    port["prices"].to_json(date_format="iso"),
                "lr_json":        port["log_returns"].to_json(date_format="iso"),
                "port_rets_json": port["port_rets"].to_json(date_format="iso"),
                "end_date":       str(port.get("end_date", "")),
                "rf_annual":      float(rf_annual or 0.0),
            }
        except Exception:
            pass

        # Run-info strip
        port     = result.portfolio
        n_obs    = len(port.get("port_rets", []))
        run_info = (
            f"Tickers: {', '.join(port.get('tickers', tickers))}  |  "
            f"{port.get('start_date','?')} to {port.get('end_date','?')}  |  "
            f"{n_obs:,} trading days"
        )

        ok_msg = f"Analysis complete -- {len(port.get('tickers', tickers))} asset(s)."
        if errors:
            alert = _warn_alert(ok_msg + _errors_summary(errors))
        else:
            alert = _ok_alert(ok_msg)

        return (
            dist_fig, roll_fig, corr_fig, ef_fig, bench_fig, ff_fig, garch_fig,
            summary_tbl, perf_tbl, cf_tbl, ff_tbl, garch_tbl,
            ff_store, var_store, run_info,
            *alert,
        )


# ============================================================
# 5. VaR cache management + CSV export
# ============================================================

def _register_var_extra_callbacks(cache: Any) -> None:

    @callback(
        Output("var-cache-info", "children"),
        Input("var-run-btn",         "n_clicks"),
        Input("var-cache-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_cache_info(_run, _clear):
        if ctx.triggered_id == "var-cache-clear-btn":
            cache.clear_all()
            return "Cache cleared."
        n_ohlcv = cache.ohlcv_entry_count()
        n_price = cache.price_entry_count()
        return (f"OHLCV: {n_ohlcv}  |  "
                f"Price series: {n_price}  |  "
                f"TTL: {cache.cache_ttl_hours:.0f} h")

    @callback(
        Output("var-download", "data", allow_duplicate=True),
        Input("var-export-prices-btn", "n_clicks"),
        State("store-var-result", "data"),
        prevent_initial_call=True,
    )
    def export_prices(_n, store):
        if not store or "prices_json" not in store:
            return no_update
        try:
            prices = pd.read_json(io.StringIO(store["prices_json"]))
            end    = str(store.get("end_date", "")).replace("-", "")
            return dcc.send_data_frame(prices.to_csv, f"prices_{end}.csv")
        except Exception:
            return no_update

    @callback(
        Output("var-download", "data", allow_duplicate=True),
        Input("var-export-returns-btn", "n_clicks"),
        State("store-var-result", "data"),
        prevent_initial_call=True,
    )
    def export_returns(_n, store):
        if not store or "lr_json" not in store:
            return no_update
        try:
            lr  = pd.read_json(io.StringIO(store["lr_json"]))
            pr  = pd.read_json(io.StringIO(store["port_rets_json"]), typ="series")
            pr.name = "Portfolio"
            out = pd.concat([lr, pr], axis=1)
            end = str(store.get("end_date", "")).replace("-", "")
            return dcc.send_data_frame(out.to_csv, f"returns_{end}.csv")
        except Exception:
            return no_update

    @callback(
        Output("var-download", "data", allow_duplicate=True),
        Input("var-export-metrics-btn", "n_clicks"),
        State("store-var-result", "data"),
        prevent_initial_call=True,
    )
    def export_metrics(_n, store):
        if not store or "port_rets_json" not in store:
            return no_update
        try:
            pr = pd.read_json(io.StringIO(store["port_rets_json"]), typ="series")
            pr.index = pd.to_datetime(pr.index)
            from lazyfin.analytics import (
                compute_performance_metrics, format_performance_table,
            )
            m   = compute_performance_metrics(
                pr, rf_annual=float(store.get("rf_annual", 0.0))
            )
            df  = format_performance_table(m).reset_index()
            end = str(store.get("end_date", "")).replace("-", "")
            return dcc.send_data_frame(df.to_csv, f"metrics_{end}.csv", index=False)
        except Exception:
            return no_update

    @callback(
        Output("var-tickers-input",  "value"),
        Output("store-ticker-dir",   "data"),
        Output("var-ticker-search",  "options"),
        Input("var-ticker-upload",   "contents"),
        State("var-ticker-upload",   "filename"),
        prevent_initial_call=True,
    )
    def load_tickers_from_file(contents, filename):
        """
        Upload handler for the ticker file (CSV or TXT).

        Tries to parse as a full ticker directory (Symbol;Instrument Fullname).
        If successful, populates the search Dropdown with all entries AND
        puts the first 50 symbols into the text box.
        If only a list of symbols is found, populates the text box only.
        """
        if contents is None:
            return no_update, no_update, no_update
        try:
            from lazyfin.data_loader import load_ticker_directory
            result = load_ticker_directory(
                content_base64 = contents,
                filename       = filename,
            )
            df = result["data"]   # DataFrame: Symbol, Instrument Fullname

            # Build Dropdown options
            options = [
                {"label": f"{r['Symbol']} — {r['Instrument Fullname']}", "value": r["Symbol"]}
                for _, r in df.head(5000).iterrows()
            ]

            # Populate text box with first 50 symbols
            tickers_str = " ".join(df["Symbol"].tolist()[:50])

            # Store directory as JSON for search callback
            dir_json = df.to_json(orient="split")

            return tickers_str, dir_json, options
        except Exception:
            return no_update, no_update, no_update

    @callback(
        Output("var-ticker-search", "options", allow_duplicate=True),
        Input("var-ticker-search",  "search_value"),
        State("store-ticker-dir",   "data"),
        prevent_initial_call=True,
    )
    def search_tickers(query, dir_json):
        """
        Live search: filter the loaded directory by symbol prefix or name substring.
        Returns up to 100 matching options for the Dropdown.
        """
        if not query or len(query.strip()) < 2:
            return no_update
        if not dir_json:
            return no_update
        try:
            from lazyfin.data_loader import search_ticker_directory
            df = pd.read_json(io.StringIO(dir_json), orient="split")
            matches = search_ticker_directory(df, query.strip(), max_results=100)
            return [{"label": label, "value": value} for label, value in matches]
        except Exception:
            return no_update

    @callback(
        Output("var-tickers-input", "value", allow_duplicate=True),
        Input("var-ticker-search",  "value"),
        prevent_initial_call=True,
    )
    def sync_dropdown_to_text(selected):
        """Append Dropdown selections to the tickers text box."""
        if not selected:
            return no_update
        if isinstance(selected, str):
            selected = [selected]
        return " ".join(t.strip().upper() for t in selected if t.strip())


# ============================================================
# 6. Indicators tab
# ============================================================

def _register_indicator_callbacks(cache: Any) -> None:

    @callback(
        Output("ind-graph",  "figure"),
        Output("ind-status", "children"),
        Output("ind-status", "color"),
        Output("ind-status", "is_open"),
        Input("ind-run-btn", "n_clicks"),
        State("ind-ticker-input",   "value"),
        State("ind-years-slider",   "value"),
        State("ind-checklist",      "value"),
        State("ind-bb-window",      "value"),
        State("ind-bb-k",           "value"),
        State("ind-dema-window",    "value"),
        State("ind-rsi-period",     "value"),
        State("ind-atr-period",     "value"),
        State("ind-mfi-period",     "value"),
        State("ind-adx-period",     "value"),
        State("ind-macd-fast",      "value"),
        State("ind-macd-slow",      "value"),
        State("ind-macd-signal",    "value"),
        State("ind-theme-dropdown", "value"),
        prevent_initial_call=True,
    )
    def run_indicators(
        _n,
        ticker, years, selected,
        bb_window, bb_k, dema_window,
        rsi_period, atr_period, mfi_period, adx_period,
        macd_fast, macd_slow, macd_signal,
        theme,
    ):
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return _empty_fig(), *_err_alert("Enter a ticker symbol.")
        if not selected:
            return _empty_fig(), *_err_alert("Select at least one indicator.")

        mf = int(macd_fast or 12)
        ms = int(macd_slow or 26)
        if mf >= ms:
            return (_empty_fig(f"MACD fast ({mf}) must be < slow ({ms})"),
                    *_err_alert(f"MACD fast ({mf}) must be < slow ({ms})."))

        params = {
            "bb_window":   int(bb_window   or 20),
            "bb_k":        float(bb_k      or 2.0),
            "dema_window": int(dema_window  or 20),
            "rsi_period":  int(rsi_period   or 14),
            "atr_period":  int(atr_period   or 14),
            "mfi_period":  int(mfi_period   or 14),
            "adx_period":  int(adx_period   or 14),
            "macd_fast":   mf,
            "macd_slow":   ms,
            "macd_signal": int(macd_signal  or 9),
        }

        try:
            result = pl.run_indicator_analysis(
                symbol              = symbol,
                years               = int(years or 3),
                selected_indicators = frozenset(selected),
                indicator_params    = params,
                cache               = cache,
                plotly_template     = theme or "plotly_white",
            )
        except ValueError as exc:
            return _empty_fig(), *_err_alert(str(exc))
        except Exception as exc:
            return _empty_fig(), *_err_alert(f"Unexpected error: {exc}")

        msg = f"{symbol} -- {len(selected)} indicator(s), {int(years or 3)} yr."
        if result.errors:
            return result.figure, *_warn_alert(msg + _errors_summary(result.errors))
        return result.figure, *_ok_alert(msg)


# ============================================================
# 7. Backtesting tab
# ============================================================

def _collect_bt_params(strategy, fast, slow, rsi_period, oversold, overbought,
                       macd_fast, macd_slow, macd_signal, bb_window, bb_k) -> dict:
    if strategy == "SMA Crossover":
        return {"fast": int(fast or 10), "slow": int(slow or 50)}
    if strategy == "RSI Mean-Reversion":
        return {"period": int(rsi_period or 14),
                "oversold": float(oversold or 30), "overbought": float(overbought or 70)}
    if strategy == "MACD":
        return {"fast": int(macd_fast or 12), "slow": int(macd_slow or 26),
                "signal": int(macd_signal or 9)}
    return {"window": int(bb_window or 20), "k": float(bb_k or 2.0)}


def _register_backtest_callbacks(cache: Any) -> None:

    @callback(
        Output("bt-graph",           "figure"),
        Output("bt-table-container", "children"),
        Output("bt-status",          "children"),
        Output("bt-status",          "color"),
        Output("bt-status",          "is_open"),
        Input("bt-run-btn", "n_clicks"),
        State("bt-ticker-input",      "value"),
        State("bt-daterange",         "start_date"),
        State("bt-daterange",         "end_date"),
        State("bt-strategy-dropdown", "value"),
        State("bt-commission-input",  "value"),
        State("bt-fast-slider",       "value"),
        State("bt-slow-slider",       "value"),
        State("bt-rsi-period",        "value"),
        State("bt-oversold",          "value"),
        State("bt-overbought",        "value"),
        State("bt-macd-fast",         "value"),
        State("bt-macd-slow",         "value"),
        State("bt-macd-signal",       "value"),
        State("bt-bb-window",         "value"),
        State("bt-bb-k",              "value"),
        prevent_initial_call=True,
    )
    def run_backtest(
        _n,
        ticker, start_str, end_str, strategy, commission,
        fast, slow, rsi_period, oversold, overbought,
        macd_fast, macd_slow, macd_signal, bb_window, bb_k,
    ):
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return _empty_fig(), html.Div(), *_err_alert("Enter a ticker symbol.")

        strategy = strategy or "SMA Crossover"
        if strategy == "SMA Crossover" and int(fast or 10) >= int(slow or 50):
            return (_empty_fig(), html.Div(),
                    *_err_alert(f"SMA fast ({fast}) must be < slow ({slow})."))

        params = _collect_bt_params(strategy, fast, slow, rsi_period, oversold,
                                    overbought, macd_fast, macd_slow, macd_signal,
                                    bb_window, bb_k)
        try:
            result = pl.run_backtest(
                symbol          = symbol,
                start_date      = _parse_date(start_str),
                end_date        = _parse_date(end_str),
                strategy        = strategy,
                strategy_params = params,
                commission      = float(commission or 0.001),
                cache           = cache,
            )
        except ValueError as exc:
            return _empty_fig(), html.Div(), *_err_alert(str(exc))
        except Exception as exc:
            return _empty_fig(), html.Div(), *_err_alert(f"Unexpected error: {exc}")

        tbl = _df_to_datatable(result.bt_table, "bt-metrics-dt")
        comm_bps = float(commission or 0.001) * 10_000
        msg = (f"{strategy} on {symbol} -- "
               f"{result.bt['n_trades']} trade(s), "
               f"commission {comm_bps:.0f} bps/leg.")
        return result.figure, tbl, *_ok_alert(msg)


# ============================================================
# 8. Walk-Forward tab
# ============================================================

def _register_walkforward_callbacks(cache: Any) -> None:

    @callback(
        Output("wf-graph",               "figure"),
        Output("wf-agg-table-container", "children"),
        Output("wf-fold-table-container","children"),
        Output("wf-status",              "children"),
        Output("wf-status",              "color"),
        Output("wf-status",              "is_open"),
        Input("bt-wf-run-btn", "n_clicks"),
        State("bt-ticker-input",      "value"),
        State("bt-daterange",         "start_date"),
        State("bt-daterange",         "end_date"),
        State("bt-strategy-dropdown", "value"),
        State("bt-commission-input",  "value"),
        State("bt-fast-slider",       "value"),
        State("bt-slow-slider",       "value"),
        State("bt-rsi-period",        "value"),
        State("bt-oversold",          "value"),
        State("bt-overbought",        "value"),
        State("bt-macd-fast",         "value"),
        State("bt-macd-slow",         "value"),
        State("bt-macd-signal",       "value"),
        State("bt-bb-window",         "value"),
        State("bt-bb-k",              "value"),
        State("bt-wf-train",          "value"),
        State("bt-wf-test",           "value"),
        prevent_initial_call=True,
    )
    def run_walkforward(
        _n,
        ticker, start_str, end_str, strategy, commission,
        fast, slow, rsi_period, oversold, overbought,
        macd_fast, macd_slow, macd_signal, bb_window, bb_k,
        train_days, test_days,
    ):
        _empty_out = (_empty_fig(), html.Div(), html.Div())
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return (*_empty_out, *_err_alert("Enter a ticker symbol."))

        strategy = strategy or "SMA Crossover"
        if strategy == "SMA Crossover" and int(fast or 10) >= int(slow or 50):
            return (*_empty_out,
                    *_err_alert(f"SMA fast ({fast}) must be < slow ({slow})."))

        params = _collect_bt_params(strategy, fast, slow, rsi_period, oversold,
                                    overbought, macd_fast, macd_slow, macd_signal,
                                    bb_window, bb_k)
        try:
            result = pl.run_walkforward(
                symbol          = symbol,
                start_date      = _parse_date(start_str),
                end_date        = _parse_date(end_str),
                strategy        = strategy,
                strategy_params = params,
                commission      = float(commission or 0.001),
                train_days      = int(train_days or 252),
                test_days       = int(test_days  or 63),
                cache           = cache,
            )
        except ValueError as exc:
            return (*_empty_out, *_err_alert(str(exc)))
        except Exception as exc:
            return (*_empty_out, *_err_alert(f"Unexpected error: {exc}"))

        agg_tbl  = _df_to_datatable(result.agg_table,  "wf-agg-dt")
        fold_tbl = _df_to_datatable(result.fold_table, "wf-fold-dt")
        agg      = result.wf.get("agg", {})
        sharpe   = agg.get("mean_sharpe", float("nan"))
        sharpe_s = f"{sharpe:.2f}" if not (isinstance(sharpe, float) and np.isnan(sharpe)) else "N/A"
        msg = (f"Walk-Forward: {strategy} on {symbol} -- "
               f"{result.wf.get('n_folds','?')} folds, mean Sharpe: {sharpe_s}.")
        return result.figure, agg_tbl, fold_tbl, *_ok_alert(msg)


# ============================================================
# 9. Stress Testing tab
# ============================================================

def _register_stress_callbacks() -> None:

    @callback(
        Output("stress-prereq-notice", "style"),
        Input("store-ff-result", "data"),
    )
    def toggle_prereq_notice(store_data):
        return {"display": "none"} if store_data else {"display": "block"}

    @callback(
        Output("stress-graph",           "figure"),
        Output("stress-table-container", "children"),
        Output("stress-status",          "children"),
        Output("stress-status",          "color"),
        Output("stress-status",          "is_open"),
        Input("stress-run-btn",    "n_clicks"),
        State("store-ff-result",   "data"),
        State("stress-custom-name",   "value"),
        State("stress-custom-shocks", "value"),
        prevent_initial_call=True,
    )
    def run_stress(_n, ff_store_data, custom_name, custom_shocks):
        _empty_out = (_empty_fig(), html.Div())
        if not ff_store_data:
            return (*_empty_out,
                    *_err_alert("Run VaR / CVaR analysis first to populate the FF regression."))

        ff_result = _ff_store_decode(ff_store_data)

        extra = []
        name_raw   = (custom_name   or "").strip()
        shocks_raw = (custom_shocks or "").strip()
        if name_raw or shocks_raw:
            try:
                extra = [parse_custom_scenario(name_raw, shocks_raw)]
            except ValueError as exc:
                return (*_empty_out, *_err_alert(f"Custom scenario error: {exc}"))

        try:
            result = pl.run_stress_test(
                ff_result       = ff_result,
                extra_scenarios = extra or None,
            )
        except ValueError as exc:
            return (*_empty_out, *_err_alert(str(exc)))
        except Exception as exc:
            return (*_empty_out, *_err_alert(f"Unexpected error: {exc}"))

        tbl = _df_to_datatable(
            result.scenarios_df[["Scenario", "Factor Shocks", "Impact %"]],
            "stress-scenarios-dt",
        )
        msg = (f"{result.ff_model} stress test -- "
               f"{result.scenarios_df.shape[0]} scenarios.")
        return result.figure, tbl, *_ok_alert(msg)


# ============================================================
# 10. Public entry point
# ============================================================

def register_callbacks(app, cache: Any) -> None:
    """Register all callbacks. Call once after app.layout is assigned."""
    _register_date_presets("var")
    _register_date_presets("bt")
    _register_strategy_visibility()
    _register_var_callbacks(cache)
    _register_var_extra_callbacks(cache)
    _register_indicator_callbacks(cache)
    _register_backtest_callbacks(cache)
    _register_walkforward_callbacks(cache)
    _register_stress_callbacks()
