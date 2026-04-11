<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://via.placeholder.com/460x200/0d47a1/ffffff?text=LazyFin">
    <img src="https://via.placeholder.com/460x200/0d47a1/ffffff?text=LazyFin" alt="LazyFin" width="460">
  </picture>
</p>

<h1 align="center">LazyFin</h1>

<p align="center">
  <strong>Quantitative Portfolio Risk &amp; Analytics Dashboard</strong><br>
  <sub>A production-grade Dash application for portfolio risk management, factor attribution, technical analysis, and backtesting — built for analysts who want results without the boilerplate</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-navy?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Dash-4.x-informational?logo=plotly&logoColor=white" alt="Dash">
  <img src="https://img.shields.io/badge/License-AGPL-navy" alt="AGPL-3.0 License">
  <img src="https://img.shields.io/badge/Version-0.6.3-blue" alt="Version">
</p>

---

## Overview

**LazyFin** is a self-contained Dash web application for quantitative portfolio analysis. Load a ticker list, set a date range, and the system fetches market data, runs risk models, fits factor regressions, computes the efficient frontier, and renders every chart — automatically.

It is designed for analysts who need institutional-quality risk metrics without writing bespoke analysis code for every new portfolio. The name reflects the intent: the analyst selects inputs; the system handles the rest.

The analytics layer follows original academic specifications rather than library defaults — Wilder EWM for ATR/RSI/ADX, HC3 heteroskedasticity-robust standard errors for Fama-French OLS, look-ahead-bias-free backtesting via one-bar signal shifts.

---

## Table of Contents

1. [Features](#features)
2. [Demo](#demo)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Example Workflow](#example-workflow)
8. [Tech Stack](#tech-stack)
9. [Future Improvements](#future-improvements)
10. [License](#license)

---

## Features

### Tab 1 — VaR / CVaR

- Four concurrent VaR methodologies: Historical Simulation, Parametric (Gaussian), Monte Carlo (multivariate normal, up to 100 000 draws), Cornish-Fisher (skewness and excess kurtosis adjustment)
- CVaR (Expected Shortfall) computed as the mean of the tail distribution below the VaR threshold
- Rolling VaR over a configurable window (63–504 trading days), with historical and parametric estimates on a shared axis
- Markowitz Efficient Frontier: 3 000 Dirichlet-sampled portfolios coloured by Sharpe ratio; minimum-variance and maximum-Sharpe special portfolios via SLSQP
- Annotated Pearson correlation heatmap with adaptive font colouring
- Benchmark cumulative wealth index and underwater drawdown comparison (any yfinance-recognised symbol)
- Fama-French factor attribution (FF3 / FF5) with HC3 heteroskedasticity-robust OLS; Jensen's alpha, factor loadings, t-statistics, p-values, R²
- GARCH(p, q) conditional volatility modelling via `arch`; 22-day ahead forecast, persistence, long-run volatility, AIC/BIC
- Full performance metrics suite: annualised return (geometric), annualised volatility, Sharpe, Sortino, Maximum Drawdown, Calmar Ratio, cumulative return, percentage positive days
- CSV export: adjusted prices, log-returns, and performance metrics downloadable directly from the UI

### Tab 2 — Technical Indicators

- Unified multi-panel Plotly figure with candlestick price chart and dedicated oscillator sub-panels
- Ten indicators: Bollinger Bands, DEMA, Parabolic SAR, RSI, ATR, OBV, MFI, ADX/±DI, Volume, MACD
- All implementations follow original specifications: Wilder EWM smoothing, no look-ahead bias in SAR, vectorised OBV
- Per-indicator parameter controls; four chart themes (Light, Dark, Seaborn, ggplot2)

### Tab 3 — Backtesting

- Vectorised daily-bar backtest across four strategies: SMA Crossover, RSI Mean-Reversion, MACD, Bollinger Band
- One-bar signal shift eliminates look-ahead bias; commission deducted as `|Δposition| × rate` on every trade bar
- Output: strategy-vs-buy-and-hold metrics table, 3-panel Plotly figure (price with entry/exit markers, equity curves, drawdown)
- Walk-forward out-of-sample validation across configurable train/test windows with per-fold and aggregate Sharpe reporting

### Tab 4 — Stress Testing

- Factor-shock scenario analysis derived from the Fama-French regression betas computed in Tab 1
- Seven built-in presets: equity crash, flight to quality, small-cap selloff, value rally, growth selloff, profitability shock, investment shock
- Custom scenario entry: name and per-factor shock magnitudes parsed from a text field

### Across All Tabs

- Searchable ticker Dropdown populated from an uploaded CSV directory (Symbol; Instrument Fullname format)
- CSV upload supports semicolon-separated files with blank rows, matching the standard ticker master list format
- In-process TTL-aware cache: OHLCV superset lookup, FF factor disk persistence across restarts, EF result memoisation (108x speedup on repeated runs)
- Date-range preset buttons; responsive two-column layout (control panel / output panel)

---

## Demo

### VaR / CVaR — Return Distribution and Method Comparison

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=VaR+%2F+CVaR+Distribution+Chart" alt="VaR and CVaR" width="700"><br>
  <sub>Multi-method VaR/CVaR overlay on the portfolio return histogram</sub>
</p>

### Rolling VaR

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Rolling+VaR+Chart" alt="Rolling VaR" width="700"><br>
  <sub>Rolling historical and parametric VaR with daily return bar chart beneath</sub>
</p>

### Markowitz Efficient Frontier

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Efficient+Frontier" alt="Efficient Frontier" width="700"><br>
  <sub>Dirichlet scatter coloured by Sharpe ratio; SLSQP-optimised minimum-variance and maximum-Sharpe portfolios</sub>
</p>

### Performance Metrics

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Performance+Metrics+Table" alt="Performance Metrics" width="700"><br>
  <sub>Annualised metrics with configurable risk-free rate</sub>
</p>

### Correlation Heatmap

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Correlation+Heatmap" alt="Correlation Heatmap" width="700"><br>
  <sub>Annotated Pearson correlation matrix with adaptive font colouring</sub>
</p>

### Benchmark Comparison

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Benchmark+Comparison" alt="Benchmark Comparison" width="700"><br>
  <sub>Cumulative wealth index and underwater drawdown versus a user-supplied benchmark</sub>
</p>

### Technical Indicators

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Indicator+Chart+%E2%80%94+Price+Panel" alt="Indicators Price Panel" width="700">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Indicator+Chart+%E2%80%94+Oscillator+Panels" alt="Indicators Oscillator Panels" width="700"><br>
  <sub>Candlestick with overlays (upper panel) and dedicated oscillator sub-panels (lower panel)</sub>
</p>

### Backtesting

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Backtest+Equity+and+Drawdown" alt="Backtest" width="700"><br>
  <sub>Strategy equity curve vs buy-and-hold with entry/exit markers and underwater drawdown</sub>
</p>

### Stress Testing

<p align="center">
  <img src="https://via.placeholder.com/700x400/1a1a2e/ffffff?text=Stress+Test+Scenario+Chart" alt="Stress Testing" width="700"><br>
  <sub>Estimated 1-day portfolio impact across factor-shock scenarios</sub>
</p>

---

## Architecture

LazyFin enforces strict layer separation across three tiers. No tier reaches into another tier's concerns.
```markdown
┌──────────────────────────────────────────────────────────────────┐
│  Dash UI  (layout.py + callbacks.py)                             │
│  Pure component tree / pure callback functions                   │
│  Reads widget state → calls pipeline → maps outputs to Dash IDs  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ one call per button click
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Pipeline / Orchestration  (aleph_toolkit/pipeline.py)           │
│  Five functions: run_var_analysis, run_backtest,                 │
│  run_walkforward, run_stress_test, run_indicator_analysis        │
│  Calls core modules in dependency order; captures per-section    │
│  errors; returns structured dataclasses (figures + tables)       │
└────┬───────────────────────────────────────────────────────────┬─┘
     │                                                           │
     ▼                                                           ▼
┌────────────────────────┐                         ┌──────────────────────┐
│ Core Analytics         │                         │  Data Layer          │
│ analytics.py           │                         │  data_loader.py      │
│ backtesting.py         │                         │  preprocessing.py    │
│ feature_engineering.py │                         │  cache.py            │
│ stress_testing.py      │                         │                      │
│ visualization.py       │                         │  yfinance downloads  │
│                        │                         │  Ken French library  │
│ Pure functions only    │                         │  Ticker CSV/URL      │
│ No UI, no network      │                         │  TTL + disk cache    │
└────────────────────────┘                         └──────────────────────┘
```
### Key design decisions

**Single pipeline call per button click.** Each tab's Run button triggers exactly one pipeline function. That function returns a dataclass containing all figures and tables for the tab. The callback maps the dataclass fields to Output IDs. There is no state written to globals inside analytics functions.

**PortfolioCache is injected, not global.** The cache instance is created once in `app.py` and passed into every pipeline call via callback closure. This makes every analytics function independently testable without mocking module-level state.

**`dcc.Store` for the only cross-tab dependency.** The Stress Testing tab requires the Fama-French regression betas produced by the VaR tab. These are serialised to JSON in `store-ff-result` by the VaR callback and deserialised by the stress callback. No other cross-tab mutable state exists.

**Caching at three levels.** The `PortfolioCache` class handles: (1) in-memory OHLCV with superset lookup — a 3-year request is served from a cached 5-year dataset; (2) Fama-French factor DataFrames persisted to disk pickle so they survive restarts; (3) Efficient Frontier results memoised by a hash of the covariance matrix, eliminating repeated SLSQP optimisations on unchanged data.

---

## Project Structure
```markdown
lazyfin/
├── app.py                          Entry point; creates Dash app and PortfolioCache
├── layout.py                       All Dash component trees (zero callback logic)
├── callbacks.py                    All @callback functions (zero layout)
├── conftest.py                     Pytest session fixtures (offline, shared)
├── test_toolkit.py                 115 unit and integration tests (~6 s, no network)
├── requirements.txt
├── assets/
│   └── custom.css                  Responsive two-column layout styling
└── aleph_toolkit/
    ├── __init__.py
    ├── cache.py                    TTL-aware PortfolioCache (superset, EF memo, disk FF)
    ├── preprocessing.py            Pure data transforms (log-returns, weights, dates)
    ├── data_loader.py              yfinance downloads, Ken French ZIP parser, ticker CSV
    ├── feature_engineering.py      Nine technical indicators and bundle dispatcher
    ├── analytics.py                VaR methods, EF optimiser, HC3 OLS, GARCH, metrics
    ├── backtesting.py              Four signal generators, vectorised engine, walk-forward
    ├── stress_testing.py           Factor-shock scenario computation and custom parser
    ├── visualization.py            Eleven Plotly figure builders (pure, no side effects)
    └── pipeline.py                 Five orchestration functions returning structured results
```
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lazyfin.git
cd lazyfin
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the optional GARCH dependency

```bash
pip install arch
```

If `arch` is not installed, all other features remain fully functional. The GARCH chart section displays a clear installation prompt.

### 5. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:8050` in a browser.

### Production deployment

```bash
pip install gunicorn
gunicorn "app:server" --bind 0.0.0.0:8050 --workers 1 --timeout 120
```

A single worker is required because `PortfolioCache` is in-process memory. For multi-worker deployments, replace `PortfolioCache` with a Redis-backed shared cache.

---

## Usage

### Inputting tickers

There are two methods.

**Text input.** Type ticker symbols directly into the Portfolio Tickers field, separated by spaces or commas (e.g. `AAPL MSFT GOOGL`).

**File upload.** Click the upload area or drag a CSV or TXT file onto it. The accepted CSV format is semicolon-separated with a header row:

```
Symbol;Instrument Fullname
```

Blank rows are ignored. After upload, a searchable dropdown is populated from the directory. Type two or more characters to search by ticker symbol or by any part of the full instrument name. Selecting entries from the dropdown copies them into the text field.

### Running an analysis

1. Set tickers, date range, and parameters in the left-hand control panel.
2. Click the Run button for the active tab.
3. All charts and tables appear in the right-hand output panel. Sections that could not be computed (e.g. GARCH without `arch`, benchmark with no internet access) display a clear message without suppressing the other sections.

### Navigating tabs

| Tab | Button | What it does |
|---|---|---|
| VaR / CVaR | Run Full Analysis | Downloads prices, computes all risk metrics, fits FF regression and GARCH |
| Indicators | Plot Indicators | Downloads OHLCV, computes selected indicators, renders unified chart |
| Backtesting | Run Backtest | Runs vectorised backtest; Run Walk-Forward for OOS validation |
| Stress Testing | Run Stress Test | Applies factor shocks to the FF betas from the VaR tab (run that tab first) |

### Exporting results

The Export Results section at the bottom of the VaR output panel provides three download buttons: adjusted prices CSV, log-returns CSV (including portfolio column), and performance metrics CSV.

---

## Example Workflow

The following sequence illustrates a complete portfolio review session.

1. Upload tickers_master_list.csv (Symbol;Instrument Fullname format).

2. Search for and select five ETFs from the dropdown.
   The symbols populate the tickers text field automatically.

3. VaR / CVaR tab:
   - Set date range to 2020-01-01 → today.
   - Select Historical, Parametric, and Monte Carlo methods.
   - Set alpha to 0.99, benchmark to SPY, FF model to FF5.
   - Click Run Full Analysis.
   - Review the VaR comparison chart, efficient frontier, and FF attribution table.
   - Download the performance metrics CSV.

4. Indicators tab:
   - Set the same ticker (single asset) and a 3-year look-back.
   - Select Bollinger Bands, RSI, MACD, and Volume.
   - Click Plot Indicators.

5. Backtesting tab:
   - Select the same ticker.
   - Choose MACD strategy with default parameters.
   - Click Run Backtest to view in-sample results.
   - Set train window 252 days, test window 63 days.
   - Click Run Walk-Forward to see OOS Sharpe per fold.

6. Stress Testing tab:
   - The FF regression from step 3 is already stored.
   - Click Run Stress Test to apply the seven built-in scenarios.
   - Add a custom Bear Market scenario: Mkt-RF: -0.20, SMB: -0.05.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Web framework | Dash 4.x, Flask (via Dash) |
| UI components | dash-bootstrap-components |
| Data | yfinance, pandas, numpy |
| Statistical computing | scipy (SLSQP optimiser), arch (GARCH, optional) |
| Visualisation | Plotly |
| Testing | pytest, pytest-mock |
| Deployment | gunicorn |

---

## Future Improvements

- **Options pricing**: Black-Scholes and binomial tree; implied volatility surface from option chain data
- **Regime detection**: Hidden Markov Model for bull/bear market identification with state-conditional risk metrics
- **Portfolio rebalancing simulation**: threshold and calendar rebalancing with explicit transaction cost accounting
- **Multi-asset historical stress testing**: scenario replay against 2008, 2020, and user-defined historical shock dates
- **Live data streaming**: intraday price updates via WebSocket feed without full page reload
- **Broker integration**: execution layer connected to a live trading API for signal-to-order routing
- **Shared cache backend**: Redis integration to support multi-worker production deployments
- **Context-sensitive export filenames**: chart and CSV exports labelled with ticker set, date range, and analysis type

---

## License

**AGPL-3.0 License** — This project is licensed under the GNU Affero General Public License v3.0. See the full license text at: [(AGPL-3.0)](https://www.bing.com/search?q="https%3A%2F%2Fwww.gnu.org%2Flicenses%2Fagpl-3.0.en.html")

---

<p align="center">
  Developed by <a href="https://github.com/CMI-Dropout">CMI-Dropout</a><br><br>
  <a href="https://github.com/CMI-Dropout">
    <img src="https://avatars.githubusercontent.com/u/152863492" width="48" height="48" alt="CMI-Dropout" style="border-radius:50%">
  </a>
</p>
