"""
layout.py
=========
Pure layout module — all dash component trees, zero callback logic.

Every component that a callback touches is identified by a stable ID string.
All IDs follow the pattern: {tab_prefix}-{widget}-{type}
  var-*      VaR / CVaR tab
  ind-*      Indicators tab
  bt-*       Backtesting tab
  wf-*       Walk-Forward section (within Backtesting tab)
  stress-*   Stress Testing tab
  store-*    dcc.Store components (cross-tab shared state)
  preset-*   Date-preset buttons (pattern-matched by callbacks)
"""

from __future__ import annotations

import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html

# ── Shared constants ─────────────────────────────────────────────────────────

_TODAY      = datetime.date.today()
_START_5Y   = _TODAY - datetime.timedelta(days=5 * 365)
_START_1Y   = _TODAY - datetime.timedelta(days=365)

_BLUE  = "#1565c0"
_TEAL  = "#26a69a"

_METHODS = [
    {"label": "Historical",          "value": "historical"},
    {"label": "Parametric",          "value": "parametric"},
    {"label": "Monte Carlo",         "value": "montecarlo"},
    {"label": "Cornish-Fisher",      "value": "cornishfisher"},
]

_INDICATORS = [
    "Bollinger Bands", "DEMA", "Parabolic SAR",
    "RSI", "ATR", "OBV", "MFI", "ADX", "Volume", "MACD",
]

_STRATEGIES = [
    {"label": "SMA Crossover",       "value": "SMA Crossover"},
    {"label": "RSI Mean-Reversion",  "value": "RSI Mean-Reversion"},
    {"label": "MACD",                "value": "MACD"},
    {"label": "Bollinger Band",      "value": "Bollinger Band"},
]

_FF_MODELS = [
    {"label": "FF3 (Market, SMB, HML)",    "value": "FF3"},
    {"label": "FF5 (+ RMW, CMA)",          "value": "FF5"},
]

_THEMES = [
    {"label": "Light",      "value": "plotly_white"},
    {"label": "Dark",       "value": "plotly_dark"},
    {"label": "Seaborn",    "value": "seaborn"},
    {"label": "ggplot2",    "value": "ggplot2"},
]


# ── Re-usable primitives ─────────────────────────────────────────────────────

def _ctrl_label(text: str) -> html.Div:
    return html.Div(text, className="ctrl-label")


def _hint(text: str) -> html.Div:
    return html.Div(text, className="hint-text")


def _divider() -> html.Hr:
    return html.Hr(className="ctrl-divider")


def _section_card(icon: str, title: str, *children) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(icon, className="icon"),
            title,
        ], className="section-card-header"),
        html.Div(list(children), className="section-card-body"),
    ], className="section-card")


def _loading_graph(graph_id: str, height: int = 420) -> dcc.Loading:
    return dcc.Loading(
        type="circle",
        color=_BLUE,
        children=dcc.Graph(
            id=graph_id,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"scale": 2, "format": "png"}},
            style={"height": f"{height}px"},
        ),
    )


def _datatable(table_id: str) -> dbc.Table:
    """
    Placeholder div for a DataTable — the actual DataTable is injected
    by callbacks using the id directly on dash_table.DataTable.
    We wrap a div with the id so callbacks can target it.
    """
    return html.Div(id=table_id, className="table-scroll")


def _run_button(btn_id: str, label: str = "▶  Run Analysis") -> dbc.Button:
    return dbc.Button(
        label,
        id=btn_id,
        n_clicks=0,
        color="primary",
        className="run-btn mt-2",
    )


def _status_alert(alert_id: str) -> dbc.Alert:
    return dbc.Alert(
        id=alert_id,
        is_open=False,
        dismissable=True,
        className="status-alert",
        fade=True,
    )


def _date_presets(prefix: str) -> html.Div:
    presets = [
        ("1M",  31), ("3M",  91), ("6M", 182),
        ("1Y", 365), ("2Y", 730), ("5Y", 1826), ("MAX", 10950),
    ]
    return html.Div(
        [
            html.Span("Quick: ", style={"fontSize": ".72rem", "color": "#78909c",
                                        "marginRight": "4px", "lineHeight": "2.2"}),
        ] + [
            dbc.Button(
                label,
                id={"type": f"{prefix}-preset", "days": days},
                size="sm",
                outline=True,
                color="primary",
                className="preset-btn me-1 mb-1",
                n_clicks=0,
            )
            for label, days in presets
        ],
        className="d-flex flex-wrap align-items-center mb-1",
    )


# ── VaR / CVaR tab ──────────────────────────────────────────────────────────

def _var_control_panel() -> html.Div:
    return html.Div([

        # -- Tickers -----------------------------------------------------------
        _ctrl_label("Portfolio Tickers"),
        # -- Searchable dropdown (populated from loaded directory) -----------
        dcc.Dropdown(
            id="var-ticker-search",
            placeholder="Search by symbol or name… (e.g. AAPL or Apple)",
            multi=True,
            clearable=True,
            searchable=True,
            options=[],          # populated by callback on startup / upload
            style={"fontSize": ".82rem"},
            className="mb-1",
        ),
        _hint("Type ≥ 2 characters to search. Load a directory file below to unlock full search."),
        dcc.Input(
            id="var-tickers-input",
            type="text",
            placeholder="AAPL, MSFT, GOOGL, …",
            debounce=False,
            value="AAPL MSFT",
            className="form-control form-control-sm mb-1",
        ),
        _hint("Space- or comma-separated yfinance symbols."),

        # -- CSV / TXT upload --------------------------------------------------
        dcc.Upload(
            id="var-ticker-upload",
            children=html.Div([
                html.Span("📂 ", style={"fontSize": "1rem"}),
                "Drop a CSV/TXT or ",
                html.A("click to upload", style={"color": "#1565c0",
                                                  "textDecoration": "underline",
                                                  "cursor": "pointer"}),
            ]),
            style={
                "width": "100%", "padding": "6px 10px",
                "border": "1px dashed #90caf9", "borderRadius": "6px",
                "textAlign": "center", "fontSize": ".76rem",
                "color": "#546e7a", "cursor": "pointer",
                "backgroundColor": "#f5f7fb", "marginBottom": "6px",
            },
            multiple=False,
        ),
        _hint(
            "Accepted formats: CSV/TXT.  "
            "Semicolon-separated (Symbol;Instrument Fullname) or comma-separated.  "
            "Blank rows are ignored.  First column must be the ticker symbol."
        ),

        _divider(),

        # -- Date range --------------------------------------------------------
        _ctrl_label("Analysis Period"),
        _date_presets("var"),
        dcc.DatePickerRange(
            id="var-daterange",
            start_date=_START_5Y.isoformat(),
            end_date=_TODAY.isoformat(),
            display_format="YYYY-MM-DD",
            style={"fontSize": ".82rem"},
            className="mb-1",
        ),

        _divider(),

        # -- Confidence level --------------------------------------------------
        _ctrl_label("Confidence Level (α)"),
        dcc.Slider(
            id="var-alpha-slider",
            min=0.90, max=0.999, step=0.001,
            value=0.99,
            marks={0.90: "90%", 0.95: "95%", 0.99: "99%", 0.999: "99.9%"},
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-2",
        ),

        _divider(),

        # -- VaR methods -------------------------------------------------------
        _ctrl_label("VaR Methods"),
        dcc.Checklist(
            id="var-methods-checklist",
            options=_METHODS,
            value=["historical"],
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block", "fontSize": ".82rem",
                        "marginBottom": "2px"},
        ),
        _hint("Hold Ctrl/⌘ to select multiple. Monte Carlo uses the sim count below."),
        html.Div([
            html.Label("MC Simulations", style={"fontSize": ".78rem",
                                                "fontWeight": "600"}),
            dcc.Slider(
                id="var-nsims-slider",
                min=1000, max=100000, step=1000,
                value=10000,
                marks={1000: "1k", 10000: "10k", 50000: "50k", 100000: "100k"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], className="mt-1"),

        _divider(),

        # -- Weights -----------------------------------------------------------
        _ctrl_label("Custom Weights (optional)"),
        dcc.Input(
            id="var-weights-input",
            type="text",
            placeholder="0.5, 0.3, 0.2  (blank = equal)",
            className="form-control form-control-sm mb-1",
        ),
        _hint("One weight per ticker, comma-separated. Normalised automatically."),

        _divider(),

        # -- Risk-free rate ----------------------------------------------------
        _ctrl_label("Risk-Free Rate (annual)"),
        dbc.Input(
            id="var-rf-input",
            type="number",
            min=0.0, max=0.20, step=0.005,
            value=0.0,
            size="sm",
            placeholder="e.g. 0.04 for 4%",
            className="mb-1",
        ),
        _hint("Used for Sharpe, Sortino, and Efficient Frontier."),

        _divider(),

        # -- Rolling VaR window ------------------------------------------------
        _ctrl_label("Rolling VaR Window"),
        dcc.Slider(
            id="var-rolling-slider",
            min=63, max=504, step=21,
            value=252,
            marks={63: "63d", 126: "126d", 252: "1yr", 504: "2yr"},
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-2",
        ),

        _divider(),

        # -- Benchmark ---------------------------------------------------------
        _ctrl_label("Benchmark Symbol"),
        dcc.Input(
            id="var-bench-input",
            type="text",
            placeholder="SPY",
            value="SPY",
            className="form-control form-control-sm mb-1",
        ),
        _hint("Any yfinance ticker. Leave blank to skip."),

        _divider(),

        # -- Fama-French model -------------------------------------------------
        _ctrl_label("Fama-French Model"),
        dcc.RadioItems(
            id="var-ff-radio",
            options=_FF_MODELS,
            value="FF3",
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block", "fontSize": ".82rem",
                        "marginBottom": "2px"},
        ),
        _hint("Downloads from Ken French's library; cached after first run."),

        _divider(),

        # -- GARCH parameters --------------------------------------------------
        _ctrl_label("GARCH Orders"),
        dbc.Row([
            dbc.Col([
                html.Label("p (GARCH)", style={"fontSize": ".76rem"}),
                dbc.Input(id="var-garch-p", type="number",
                          min=1, max=3, step=1, value=1, size="sm"),
            ], width=6),
            dbc.Col([
                html.Label("q (ARCH)", style={"fontSize": ".76rem"}),
                dbc.Input(id="var-garch-q", type="number",
                          min=1, max=3, step=1, value=1, size="sm"),
            ], width=6),
        ], className="mb-1"),
        _hint("Requires pip install arch. GARCH(1,1) is the standard starting point."),

        _divider(),

        # -- Chart theme -------------------------------------------------------
        _ctrl_label("Chart Theme"),
        dcc.Dropdown(
            id="var-theme-dropdown",
            options=_THEMES,
            value="plotly_white",
            clearable=False,
            style={"fontSize": ".82rem"},
            className="mb-2",
        ),

        _divider(),

        # -- Cache info --------------------------------------------------------
        _ctrl_label("Data Cache"),
        html.Div(id="var-cache-info", className="hint-text mb-1",
                 style={"fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
        dbc.Button("\u2717  Clear Cache", id="var-cache-clear-btn",
                   size="sm", outline=True, color="secondary",
                   className="mb-2 w-100"),

        _divider(),

        _run_button("var-run-btn", "▶  Run Full Analysis"),

    ], className="control-panel")


def _var_output_panel() -> html.Div:
    return html.Div([
        _status_alert("var-status"),

        _section_card("📊", "VaR / CVaR — Return Distribution",
            _loading_graph("var-dist-graph", 430),
        ),

        _section_card("📋", "Summary Tables",
            dbc.Row([
                dbc.Col([
                    html.P("VaR / CVaR by Method",
                           style={"fontSize": ".8rem", "fontWeight": "700",
                                  "color": _BLUE, "marginBottom": "4px"}),
                    _datatable("var-summary-table"),
                ], width=12, lg=6),
                dbc.Col([
                    html.P("Performance Metrics",
                           style={"fontSize": ".8rem", "fontWeight": "700",
                                  "color": _BLUE, "marginBottom": "4px"}),
                    _datatable("var-perf-table"),
                ], width=12, lg=6),
            ]),
            html.Div(id="var-cf-container", className="mt-2"),
        ),

        _section_card("📈", "Rolling VaR",
            _loading_graph("var-rolling-graph", 560),
        ),

        _section_card("🔥", "Correlation Heatmap",
            _loading_graph("var-corr-graph", 480),
        ),

        _section_card("⭐", "Efficient Frontier",
            _loading_graph("var-ef-graph", 520),
        ),

        _section_card("⚖️", "Benchmark Comparison",
            _loading_graph("var-bench-graph", 560),
        ),

        _section_card("🧮", "Fama-French Attribution",
            _loading_graph("var-ff-graph", 360),
            html.Div(id="var-ff-table-container", className="mt-2"),
        ),

        _section_card("🌊", "GARCH Volatility Model",
            _loading_graph("var-garch-graph", 560),
            html.Div(id="var-garch-table-container", className="mt-2"),
        ),

        _section_card("💾", "Export Results",
            html.Div(id="var-run-info", className="hint-text mb-2"),
            dbc.Row([
                dbc.Col(
                    dbc.Button("📥 Download Prices CSV", id="var-export-prices-btn",
                               size="sm", outline=True, color="primary",
                               className="w-100"),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button("📥 Download Returns CSV", id="var-export-returns-btn",
                               size="sm", outline=True, color="primary",
                               className="w-100"),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button("📥 Download Metrics CSV", id="var-export-metrics-btn",
                               size="sm", outline=True, color="primary",
                               className="w-100"),
                    width=4,
                ),
            ], className="mb-2 g-1"),
            dcc.Download(id="var-download"),
        ),

    ], className="output-panel")


def var_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(_var_control_panel(), width=12, lg=3,
                    style={"padding": "0"}),
            dbc.Col(_var_output_panel(), width=12, lg=9,
                    style={"padding": "0"}),
        ], className="g-0 tab-content-area"),
    ])


# ── Indicators tab ───────────────────────────────────────────────────────────

def _ind_param_slider(slider_id: str, label: str,
                      min_val: int, max_val: int, step: int,
                      default: int) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": ".76rem", "marginBottom": "1px"}),
        dcc.Slider(
            id=slider_id,
            min=min_val, max=max_val, step=step, value=default,
            tooltip={"placement": "right", "always_visible": True},
            marks=None,
        ),
    ], className="mb-2")


def _ind_control_panel() -> html.Div:
    return html.Div([

        _ctrl_label("Ticker Symbol"),
        dcc.Input(id="ind-ticker-input", type="text",
                  placeholder="AAPL", value="AAPL",
                  className="form-control form-control-sm mb-1"),

        _divider(),

        _ctrl_label("Look-Back Period"),
        dcc.Slider(
            id="ind-years-slider",
            min=1, max=10, step=1, value=3,
            marks={1: "1yr", 3: "3yr", 5: "5yr", 10: "10yr"},
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-2",
        ),

        _divider(),

        _ctrl_label("Indicators"),
        dcc.Checklist(
            id="ind-checklist",
            options=[{"label": i, "value": i} for i in _INDICATORS],
            value=["Bollinger Bands", "RSI", "Volume", "MACD"],
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block", "fontSize": ".82rem",
                        "marginBottom": "2px"},
        ),

        _divider(),

        _ctrl_label("Indicator Parameters"),
        dbc.Accordion([
            dbc.AccordionItem([
                _ind_param_slider("ind-bb-window", "BB Window",    10, 50,  1, 20),
                _ind_param_slider("ind-bb-k",      "BB Width (σ)",  1,  4,  0.1, 2),
            ], title="Bollinger Bands"),
            dbc.AccordionItem([
                _ind_param_slider("ind-dema-window", "DEMA Period", 5, 100, 1, 20),
            ], title="DEMA"),
            dbc.AccordionItem([
                _ind_param_slider("ind-rsi-period", "RSI Period", 5, 50, 1, 14),
            ], title="RSI"),
            dbc.AccordionItem([
                _ind_param_slider("ind-atr-period", "ATR Period", 5, 50, 1, 14),
            ], title="ATR"),
            dbc.AccordionItem([
                _ind_param_slider("ind-mfi-period", "MFI Period", 5, 50, 1, 14),
            ], title="MFI"),
            dbc.AccordionItem([
                _ind_param_slider("ind-adx-period", "ADX Period", 5, 50, 1, 14),
            ], title="ADX"),
            dbc.AccordionItem([
                _ind_param_slider("ind-macd-fast",   "Fast EMA",    5, 30,  1, 12),
                _ind_param_slider("ind-macd-slow",   "Slow EMA",   10, 60,  1, 26),
                _ind_param_slider("ind-macd-signal", "Signal EMA",  3, 20,  1,  9),
            ], title="MACD"),
        ], start_collapsed=True, flush=True, className="mb-2"),

        _divider(),

        _ctrl_label("Chart Theme"),
        dcc.Dropdown(
            id="ind-theme-dropdown",
            options=_THEMES,
            value="plotly_white",
            clearable=False,
            style={"fontSize": ".82rem"},
            className="mb-2",
        ),

        _run_button("ind-run-btn", "▶  Plot Indicators"),

    ], className="control-panel")


def indicators_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(_ind_control_panel(), width=12, lg=3, style={"padding": "0"}),
            dbc.Col([
                html.Div([
                    _status_alert("ind-status"),
                    dcc.Loading(
                        type="circle", color=_BLUE,
                        children=dcc.Graph(
                            id="ind-graph",
                            config={"displayModeBar": True, "displaylogo": False,
                                    "toImageButtonOptions": {"scale": 2}},
                            style={"minHeight": "500px"},
                        ),
                    ),
                ], className="output-panel"),
            ], width=12, lg=9, style={"padding": "0"}),
        ], className="g-0 tab-content-area"),
    ])


# ── Backtesting tab ──────────────────────────────────────────────────────────

def _strategy_param_block(block_id: str,
                           children: list,
                           visible: bool = False) -> html.Div:
    return html.Div(
        children,
        id=block_id,
        className="strategy-params",
        style={"display": "block" if visible else "none"},
    )


def _bt_control_panel() -> html.Div:
    return html.Div([

        _ctrl_label("Ticker Symbol"),
        dcc.Input(id="bt-ticker-input", type="text",
                  placeholder="AAPL", value="AAPL",
                  className="form-control form-control-sm mb-1"),

        _divider(),

        _ctrl_label("Date Range"),
        _date_presets("bt"),
        dcc.DatePickerRange(
            id="bt-daterange",
            start_date=_START_5Y.isoformat(),
            end_date=_TODAY.isoformat(),
            display_format="YYYY-MM-DD",
            className="mb-1",
        ),

        _divider(),

        _ctrl_label("Strategy"),
        dcc.Dropdown(
            id="bt-strategy-dropdown",
            options=_STRATEGIES,
            value="SMA Crossover",
            clearable=False,
            style={"fontSize": ".82rem"},
            className="mb-2",
        ),

        # -- Strategy-specific parameter blocks (shown/hidden by callback) ----
        _strategy_param_block("bt-sma-params", visible=True, children=[
            html.Label("SMA Parameters",
                       style={"fontSize": ".78rem", "fontWeight": "700",
                              "marginBottom": "4px"}),
            _ind_param_slider("bt-fast-slider", "Fast Window",  3,  60, 1, 10),
            _ind_param_slider("bt-slow-slider", "Slow Window", 10, 200, 5, 50),
        ]),
        _strategy_param_block("bt-rsi-params", visible=False, children=[
            html.Label("RSI Parameters",
                       style={"fontSize": ".78rem", "fontWeight": "700",
                              "marginBottom": "4px"}),
            _ind_param_slider("bt-rsi-period",   "RSI Period",   5,  50, 1, 14),
            _ind_param_slider("bt-oversold",      "Oversold <",  10,  45, 5, 30),
            _ind_param_slider("bt-overbought",    "Overbought >", 55, 90, 5, 70),
        ]),
        _strategy_param_block("bt-macd-params", visible=False, children=[
            html.Label("MACD Parameters",
                       style={"fontSize": ".78rem", "fontWeight": "700",
                              "marginBottom": "4px"}),
            _ind_param_slider("bt-macd-fast",   "Fast EMA",   5, 30, 1, 12),
            _ind_param_slider("bt-macd-slow",   "Slow EMA",  10, 60, 1, 26),
            _ind_param_slider("bt-macd-signal", "Signal EMA", 3, 20, 1,  9),
        ]),
        _strategy_param_block("bt-bb-params", visible=False, children=[
            html.Label("Bollinger Band Parameters",
                       style={"fontSize": ".78rem", "fontWeight": "700",
                              "marginBottom": "4px"}),
            _ind_param_slider("bt-bb-window", "BB Window",   5, 50, 1,  20),
            _ind_param_slider("bt-bb-k",      "BB Width (σ)",1,  4, 0.1, 2),
        ]),

        _divider(),

        _ctrl_label("Commission (per leg, decimal)"),
        dbc.Input(id="bt-commission-input", type="number",
                  min=0.0, max=0.01, step=0.0005, value=0.001, size="sm",
                  className="mb-1"),
        _hint("0.001 = 10 bps.  Round-trip = 2× this value."),

        _divider(),

        _run_button("bt-run-btn", "▶  Run Backtest"),

        _divider(),

        _ctrl_label("Walk-Forward Parameters"),
        _ind_param_slider("bt-wf-train", "Train Window (days)",  60, 756, 21, 252),
        _ind_param_slider("bt-wf-test",  "Test Window (days)",   21, 252, 21,  63),
        _hint("63d ≈ 3mo  |  126d ≈ 6mo  |  252d ≈ 1yr"),

        _run_button("bt-wf-run-btn", "▶  Run Walk-Forward"),

    ], className="control-panel")


def _bt_output_panel() -> html.Div:
    return html.Div([
        _status_alert("bt-status"),

        dcc.Tabs(id="bt-output-tabs", value="bt-chart-tab", children=[
            dcc.Tab(label="Backtest", value="bt-chart-tab", children=[
                html.Div([
                    html.Div(id="bt-table-container", className="mb-3"),
                    dcc.Loading(type="circle", color=_BLUE,
                                children=dcc.Graph(
                                    id="bt-graph",
                                    config={"displayModeBar": True,
                                            "displaylogo": False,
                                            "toImageButtonOptions": {"scale": 2}},
                                    style={"height": "660px"},
                                )),
                ], style={"padding": "8px 4px"}),
            ]),
            dcc.Tab(label="Walk-Forward", value="wf-chart-tab", children=[
                html.Div([
                    _status_alert("wf-status"),
                    html.Div(id="wf-agg-table-container", className="mb-2"),
                    html.Div(id="wf-fold-table-container", className="mb-3"),
                    dcc.Loading(type="circle", color=_BLUE,
                                children=dcc.Graph(
                                    id="wf-graph",
                                    config={"displayModeBar": True,
                                            "displaylogo": False,
                                            "toImageButtonOptions": {"scale": 2}},
                                    style={"height": "640px"},
                                )),
                ], style={"padding": "8px 4px"}),
            ]),
        ], className="mt-0"),

    ], className="output-panel")


def backtesting_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(_bt_control_panel(), width=12, lg=3, style={"padding": "0"}),
            dbc.Col(_bt_output_panel(), width=12, lg=9, style={"padding": "0"}),
        ], className="g-0 tab-content-area"),
    ])


# ── Stress Testing tab ───────────────────────────────────────────────────────

def _stress_control_panel() -> html.Div:
    return html.Div([

        html.Div([
            html.Strong("How it works:"),
            html.P(
                "Stress testing uses the Fama-French regression betas "
                "from the VaR tab.  Run a full VaR analysis (with any "
                "FF model) first, then return here.",
                style={"margin": "4px 0 0 0"},
            ),
        ], className="stress-info"),

        html.Div(
            "⚠  No VaR/FF result available yet.  Run the VaR tab first.",
            id="stress-prereq-notice",
            className="prereq-notice",
        ),

        _ctrl_label("Custom Scenario (optional)"),
        _hint("Add one scenario on top of the 7 built-in presets."),

        html.Label("Name", style={"fontSize": ".78rem", "marginTop": "6px"}),
        dcc.Input(
            id="stress-custom-name",
            type="text",
            placeholder="My Custom Scenario",
            className="form-control form-control-sm mb-2",
        ),

        html.Label("Factor Shocks", style={"fontSize": ".78rem"}),
        dcc.Textarea(
            id="stress-custom-shocks",
            placeholder=(
                "One shock per line:\n"
                "Mkt-RF: -0.10\n"
                "SMB: -0.05\n"
                "# lines starting with # are comments"
            ),
            style={"width": "100%", "height": "100px", "fontSize": ".78rem",
                   "fontFamily": "monospace", "borderColor": "#cfd8dc",
                   "borderRadius": "4px", "padding": "6px"},
        ),
        _hint("Valid factors: Mkt-RF, SMB, HML, RMW, CMA.  Values are decimals (−0.10 = −10%)."),

        _divider(),

        _run_button("stress-run-btn", "▶  Run Stress Test"),

    ], className="control-panel")


def _stress_output_panel() -> html.Div:
    return html.Div([
        _status_alert("stress-status"),

        _section_card("💥", "Scenario Impact Table",
            html.Div(id="stress-table-container"),
        ),

        _section_card("📊", "Estimated 1-Day Portfolio Impact",
            dcc.Loading(type="circle", color=_BLUE,
                        children=dcc.Graph(
                            id="stress-graph",
                            config={"displayModeBar": True, "displaylogo": False,
                                    "toImageButtonOptions": {"scale": 2}},
                        )),
        ),

    ], className="output-panel")


def stress_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(_stress_control_panel(), width=12, lg=3, style={"padding": "0"}),
            dbc.Col(_stress_output_panel(), width=12, lg=9, style={"padding": "0"}),
        ], className="g-0 tab-content-area"),
    ])


# ── Cross-tab stores ─────────────────────────────────────────────────────────

def _stores() -> list:
    return [
        # Serialised ff_result for the Stress Testing tab
        dcc.Store(id="store-ff-result",  storage_type="memory"),
        # Last VaR portfolio result for CSV export (prices + log_returns)
        dcc.Store(id="store-var-result", storage_type="memory"),
        # Ticker directory DataFrame serialised as JSON (loaded once at startup)
        dcc.Store(id="store-ticker-dir", storage_type="memory"),
    ]


# ── Root layout ──────────────────────────────────────────────────────────────

def create_layout() -> html.Div:
    """
    Build and return the complete application layout.
    Call this once, assign to app.layout.
    """
    return html.Div([

        # -- Application header ------------------------------------------------
        html.Div([
            html.Div([
                html.Img(
                    src="/assets/lazyfin_logo.png",
                    style={"height": "40px", "marginRight": "12px", "verticalAlign": "middle"}
                ),
                html.Div([
                    html.H4("LazyFin", style={"display": "inline-block", "verticalAlign": "middle", "margin": "0"}),
                    html.P("Portfolio Risk & Analytics Dashboard  |  v0.6.3",
                        className="subtitle", style={"margin": "0", "fontSize": ".85rem"}),
                ], style={"display": "inline-block", "verticalAlign": "middle"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
        ], className="app-header"),


        # -- Cross-tab shared stores -------------------------------------------
        *_stores(),

        # -- Main tab strip ----------------------------------------------------
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-var",
            className="main-tabs px-3",
            children=[
                dbc.Tab(label="📉  VaR / CVaR",    tab_id="tab-var",
                        children=var_tab()),
                dbc.Tab(label="📈  Indicators",    tab_id="tab-ind",
                        children=indicators_tab()),
                dbc.Tab(label="🔄  Backtesting",   tab_id="tab-bt",
                        children=backtesting_tab()),
                dbc.Tab(label="💥  Stress Testing", tab_id="tab-stress",
                        children=stress_tab()),
            ],
        ),

    ])
