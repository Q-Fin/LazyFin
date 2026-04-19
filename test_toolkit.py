"""
tests/test_toolkit.py
=====================
Comprehensive test suite for aleph_toolkit.

Run with:
    pytest tests/test_toolkit.py -v

All tests are fully offline — no network calls are made.
Network-dependent functions (yfinance downloads, FF factor downloads)
are patched at the pipeline module level wherever required.

Coverage
--------
  - cache.py              : TTL logic, all cache slots
  - preprocessing.py      : all transforms, date resolution, weight utils
  - feature_engineering.py: every indicator, bundle dispatch
  - analytics.py          : all VaR methods, EF, FF regression, metrics
  - backtesting.py        : all 4 strategies, walk-forward folds
  - stress_testing.py     : preset scenarios, custom scenario parsing
  - visualization.py      : figure type, trace counts, no-crash guarantee
  - pipeline.py           : integration tests via mocked data layer
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ── Fixtures ─────────────────────────────────────────────────────────────────

# Session-scoped fixtures (rng, price_index, prices, ohlcv, ff3_factors,
# portfolio, port_rets, log_rets, weights) are defined in conftest.py
# and injected automatically by pytest.

class TestCache:
    def test_price_roundtrip(self):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache(ttl_hours=8.0)
        s = pd.Series([1.0, 2.0, 3.0], name="X")
        d0, d1 = datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)
        cache.put_price("AAPL", d0, d1, s)
        result = cache.get_price("AAPL", d0, d1)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_ohlcv_roundtrip(self, ohlcv):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        cache.put_ohlcv("TSLA", 2, ohlcv)
        result = cache.get_ohlcv("TSLA", 2)
        assert result.shape == ohlcv.shape

    def test_ff_roundtrip(self, ff3_factors):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        cache.put_ff_factors("FF3", ff3_factors)
        result = cache.get_ff_factors("FF3")
        assert result.shape == ff3_factors.shape

    def test_cache_miss_returns_none(self):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        assert cache.get_ohlcv("NOPE", 1) is None
        assert cache.get_price("NOPE", datetime.date.today(), datetime.date.today()) is None
        assert cache.get_ff_factors("FF5") is None

    def test_ttl_expiry(self):
        from aleph_toolkit.cache import PortfolioCache
        import time
        cache = PortfolioCache(ttl_hours=0.0001)  # ~0.36 seconds
        s = pd.Series([1.0, 2.0])
        d0, d1 = datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)
        cache.put_price("X", d0, d1, s)
        time.sleep(0.5)
        assert cache.get_price("X", d0, d1) is None  # should have expired

    def test_clear_all(self, ohlcv):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        cache.put_ohlcv("A", 1, ohlcv)
        cache.put_ff_factors("FF3", pd.DataFrame({"RF": [0.0]}))
        cache.clear_all()
        assert cache.ohlcv_entry_count() == 0
        assert cache.get_ff_factors("FF3") is None

    def test_case_insensitive_symbol(self, ohlcv):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        cache.put_ohlcv("aapl", 1, ohlcv)
        assert cache.get_ohlcv("AAPL", 1) is not None

    def test_info_string(self, ohlcv):
        from aleph_toolkit.cache import PortfolioCache
        cache = PortfolioCache()
        cache.put_ohlcv("AAPL", 3, ohlcv)
        info = cache.info()
        assert "AAPL" in info
        assert "TTL" in info


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_flatten_multiindex(self, prices):
        from aleph_toolkit.preprocessing import flatten_multiindex_columns
        mi = pd.MultiIndex.from_arrays([["Close", "Close"], ["AAPL", "MSFT"]])
        df = pd.DataFrame(
            np.random.rand(5, 2),
            columns=mi,
        )
        result = flatten_multiindex_columns(df)
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_flatten_flat_is_noop(self, prices):
        from aleph_toolkit.preprocessing import flatten_multiindex_columns
        result = flatten_multiindex_columns(prices)
        assert list(result.columns) == list(prices.columns)

    def test_rename_adj_close(self):
        from aleph_toolkit.preprocessing import rename_adj_close
        df = pd.DataFrame({"Adj Close": [1.0, 2.0], "Open": [1.0, 2.0]})
        result = rename_adj_close(df)
        assert "Close" in result.columns
        assert "Adj Close" not in result.columns

    def test_validate_ohlcv_columns_all_present(self, ohlcv):
        from aleph_toolkit.preprocessing import validate_ohlcv_columns
        missing = validate_ohlcv_columns(ohlcv)
        assert missing == []

    def test_validate_ohlcv_columns_missing(self):
        from aleph_toolkit.preprocessing import validate_ohlcv_columns
        df = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
        missing = validate_ohlcv_columns(df)
        assert "High" in missing
        assert "Low" in missing
        assert "Volume" in missing

    def test_clean_ohlcv(self, ohlcv):
        from aleph_toolkit.preprocessing import clean_ohlcv
        result = clean_ohlcv(ohlcv)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.is_monotonic_increasing
        assert result.isna().sum().sum() == 0

    def test_compute_log_returns_shape(self, prices):
        from aleph_toolkit.preprocessing import compute_log_returns
        lr = compute_log_returns(prices)
        assert lr.shape == (len(prices) - 1, prices.shape[1])
        assert lr.isna().sum().sum() == 0

    def test_compute_log_returns_raises_on_single_row(self):
        from aleph_toolkit.preprocessing import compute_log_returns
        df = pd.DataFrame({"A": [100.0]})
        with pytest.raises(ValueError, match="need ≥ 2"):
            compute_log_returns(df)

    def test_compute_single_log_returns(self, ohlcv):
        from aleph_toolkit.preprocessing import compute_single_log_returns
        lr = compute_single_log_returns(ohlcv["Close"])
        assert len(lr) == len(ohlcv) - 1

    def test_equal_weights(self):
        from aleph_toolkit.preprocessing import equal_weights
        w = equal_weights(5)
        assert len(w) == 5
        assert abs(w.sum() - 1.0) < 1e-10
        assert all(abs(v - 0.2) < 1e-10 for v in w)

    def test_equal_weights_raises_on_zero(self):
        from aleph_toolkit.preprocessing import equal_weights
        with pytest.raises(ValueError):
            equal_weights(0)

    def test_normalise_weights(self):
        from aleph_toolkit.preprocessing import normalise_weights
        w = normalise_weights([1.0, 2.0, 7.0])
        assert abs(w.sum() - 1.0) < 1e-10
        assert abs(w[0] - 0.1) < 1e-10
        assert abs(w[2] - 0.7) < 1e-10

    def test_normalise_weights_raises_negative(self):
        from aleph_toolkit.preprocessing import normalise_weights
        with pytest.raises(ValueError, match="non-negative"):
            normalise_weights([1.0, -0.5])

    def test_normalise_weights_raises_zero_sum(self):
        from aleph_toolkit.preprocessing import normalise_weights
        with pytest.raises(ValueError, match="zero"):
            normalise_weights([0.0, 0.0])

    def test_parse_weights_valid(self):
        from aleph_toolkit.preprocessing import parse_weights
        result = parse_weights("0.5, 0.3, 0.2", 3)
        assert result == [0.5, 0.3, 0.2]

    def test_parse_weights_blank(self):
        from aleph_toolkit.preprocessing import parse_weights
        assert parse_weights("", 3) is None
        assert parse_weights("   ", 3) is None

    def test_parse_weights_wrong_count(self):
        from aleph_toolkit.preprocessing import parse_weights
        assert parse_weights("0.5, 0.5", 3) is None

    def test_parse_weights_negative(self):
        from aleph_toolkit.preprocessing import parse_weights
        assert parse_weights("-0.1, 0.6, 0.5", 3) is None

    def test_build_portfolio_returns(self, prices):
        from aleph_toolkit.preprocessing import build_portfolio_returns, equal_weights
        w = equal_weights(3)
        p = build_portfolio_returns(
            prices, w,
            datetime.date(2020, 1, 1),
            datetime.date(2022, 12, 31),
        )
        assert "port_rets" in p
        assert "log_returns" in p
        assert "tickers" in p
        assert len(p["tickers"]) == 3
        assert abs(p["weights"].sum() - 1.0) < 1e-10

    def test_build_portfolio_returns_raises_bad_dates(self, prices):
        from aleph_toolkit.preprocessing import build_portfolio_returns, equal_weights
        w = equal_weights(3)
        with pytest.raises(ValueError, match="before"):
            build_portfolio_returns(
                prices, w,
                datetime.date(2022, 12, 31),
                datetime.date(2020, 1, 1),
            )

    def test_build_portfolio_returns_raises_too_few_obs(self, prices):
        from aleph_toolkit.preprocessing import build_portfolio_returns, equal_weights
        w = equal_weights(3)
        with pytest.raises(ValueError, match="Minimum required"):
            build_portfolio_returns(
                prices, w,
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 10),
            )

    def test_resolve_date_range_defaults(self):
        from aleph_toolkit.preprocessing import resolve_date_range
        start, end = resolve_date_range(None, None, fallback_years=3)
        today = datetime.date.today()
        assert end == today
        assert (today - start).days == pytest.approx(3 * 365, abs=2)

    def test_resolve_date_range_raises_inverted(self):
        from aleph_toolkit.preprocessing import resolve_date_range
        with pytest.raises(ValueError, match="strictly before"):
            resolve_date_range(
                datetime.date(2023, 12, 31),
                datetime.date(2020, 1, 1),
            )

    def test_align_date_ranges(self, port_rets, ff3_factors):
        from aleph_toolkit.preprocessing import align_date_ranges
        bench = ff3_factors["Mkt-RF"]
        aligned_p, aligned_b = align_date_ranges(port_rets, bench)
        assert len(aligned_p) == len(aligned_b)
        assert aligned_p.index.equals(aligned_b.index)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_bollinger_keys(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_bollinger
        bb = compute_bollinger(ohlcv["Close"])
        assert set(bb.keys()) == {"upper", "mid", "lower"}

    def test_bollinger_band_ordering(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_bollinger
        bb = compute_bollinger(ohlcv["Close"], window=20)
        # Drop any row where any band is NaN (warm-up period = first window-1 rows)
        combined = pd.concat([bb["upper"], bb["mid"], bb["lower"]], axis=1).dropna()
        assert (combined.iloc[:, 0] >= combined.iloc[:, 1]).all(), "upper < mid"
        assert (combined.iloc[:, 1] >= combined.iloc[:, 2]).all(), "mid < lower"

    def test_dema_length(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_dema
        dema = compute_dema(ohlcv["Close"], window=20)
        assert len(dema) == len(ohlcv)

    def test_rsi_range(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_rsi
        rsi = compute_rsi(ohlcv["Close"])
        assert rsi.dropna().between(0, 100).all()

    def test_atr_positive(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_atr
        atr = compute_atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        assert (atr.dropna() >= 0).all()

    def test_obv_cumulative(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_obv
        obv = compute_obv(ohlcv["Close"], ohlcv["Volume"])
        assert len(obv) == len(ohlcv)

    def test_adx_keys(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_adx
        adx = compute_adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        assert set(adx.keys()) == {"plus_di", "minus_di", "dx", "adx"}

    def test_adx_range(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_adx
        adx = compute_adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        assert (adx["adx"].dropna() >= 0).all()

    def test_mfi_range(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_mfi
        mfi = compute_mfi(
            ohlcv["High"], ohlcv["Low"],
            ohlcv["Close"], ohlcv["Volume"],
        )
        assert mfi.dropna().between(0, 100).all()

    def test_psar_length(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_psar
        psar = compute_psar(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        assert len(psar) == len(ohlcv)

    def test_macd_keys(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_macd
        m = compute_macd(ohlcv["Close"])
        assert set(m.keys()) == {"macd_line", "signal_line", "histogram"}

    def test_macd_histogram_equals_diff(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_macd
        m = compute_macd(ohlcv["Close"])
        diff = m["macd_line"] - m["signal_line"]
        pd.testing.assert_series_equal(m["histogram"], diff, check_names=False)

    def test_macd_raises_fast_ge_slow(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_macd
        with pytest.raises(ValueError, match="fast"):
            compute_macd(ohlcv["Close"], fast=26, slow=12)

    def test_compute_indicator_bundle_selected_keys(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_indicator_bundle
        selected = frozenset(["RSI", "MACD", "Volume"])
        bundle = compute_indicator_bundle(ohlcv, selected, {})
        assert "rsi" in bundle
        assert "macd" in bundle
        assert "volume" in bundle
        assert "bollinger" not in bundle  # not requested

    def test_compute_indicator_bundle_unknown_raises(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_indicator_bundle
        with pytest.raises(ValueError, match="Unknown"):
            compute_indicator_bundle(ohlcv, frozenset(["NOPE"]), {})

    def test_compute_indicator_bundle_missing_column_raises(self):
        from aleph_toolkit.feature_engineering import compute_indicator_bundle
        df = pd.DataFrame({"Close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing"):
            compute_indicator_bundle(df, frozenset(["RSI"]), {})


# ─────────────────────────────────────────────────────────────────────────────
# 4. Analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_historical_var_positive(self, port_rets):
        from aleph_toolkit.analytics import compute_historical_var
        r = compute_historical_var(port_rets, 0.99)
        assert r["VaR"] > 0
        assert r["CVaR"] >= r["VaR"]

    def test_parametric_var_positive(self, port_rets):
        from aleph_toolkit.analytics import compute_parametric_var
        r = compute_parametric_var(port_rets, 0.99)
        assert r["VaR"] > 0
        assert r["CVaR"] >= r["VaR"]

    def test_montecarlo_var_shape(self, log_rets, weights):
        from aleph_toolkit.analytics import compute_montecarlo_var
        r = compute_montecarlo_var(log_rets, weights, 0.99, n_sims=2000, seed=0)
        assert r["VaR"] > 0
        assert r["sim_returns"].shape == (2000,)

    def test_cornishfisher_var_diagnostics(self, port_rets):
        from aleph_toolkit.analytics import compute_cornishfisher_var
        r = compute_cornishfisher_var(port_rets, 0.99)
        assert r["VaR"] > 0
        assert "skewness" in r
        assert "excess_kurtosis" in r
        assert "z_cf" in r

    def test_compute_var_results_all_methods(self, portfolio):
        from aleph_toolkit.analytics import compute_var_results
        vr = compute_var_results(
            portfolio,
            ["historical", "parametric", "montecarlo", "cornishfisher"],
            n_sims=1000,
        )
        assert len(vr["methods_run"]) == 4
        assert vr["primary_VaR"] > 0

    def test_compute_var_results_unknown_method(self, portfolio):
        from aleph_toolkit.analytics import compute_var_results
        with pytest.raises(ValueError, match="Unknown"):
            compute_var_results(portfolio, ["banana"])

    def test_compute_var_results_empty_methods(self, portfolio):
        from aleph_toolkit.analytics import compute_var_results
        with pytest.raises(ValueError, match="non-empty"):
            compute_var_results(portfolio, [])

    def test_performance_metrics_keys(self, port_rets):
        from aleph_toolkit.analytics import compute_performance_metrics
        m = compute_performance_metrics(port_rets, rf_annual=0.04)
        expected = {
            "ann_ret", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "calmar", "cum_ret", "best_day", "worst_day",
            "pct_positive_days", "n_days", "rf_annual",
        }
        assert expected.issubset(set(m.keys()))

    def test_performance_metrics_pct_positive_range(self, port_rets):
        from aleph_toolkit.analytics import compute_performance_metrics
        m = compute_performance_metrics(port_rets)
        assert 0.0 <= m["pct_positive_days"] <= 1.0

    def test_performance_metrics_max_drawdown_positive(self, port_rets):
        from aleph_toolkit.analytics import compute_performance_metrics
        m = compute_performance_metrics(port_rets)
        assert m["max_drawdown"] >= 0.0

    def test_format_performance_table(self, port_rets):
        from aleph_toolkit.analytics import compute_performance_metrics, format_performance_table
        m = compute_performance_metrics(port_rets, rf_annual=0.04)
        df = format_performance_table(m)
        assert isinstance(df, pd.DataFrame)
        assert "Value" in df.columns
        assert len(df) == 11

    def test_rolling_var_shape(self, port_rets):
        from aleph_toolkit.analytics import compute_rolling_var
        rv = compute_rolling_var(port_rets, 0.99, window=63)
        assert rv["rolling_df"].shape[1] == 3
        assert rv["window"] == 63

    def test_rolling_var_raises_short_series(self, port_rets):
        from aleph_toolkit.analytics import compute_rolling_var
        with pytest.raises(ValueError, match="window"):
            compute_rolling_var(port_rets.iloc[:10], 0.99, window=63)

    def test_efficient_frontier_requires_2_assets(self, log_rets):
        from aleph_toolkit.analytics import compute_efficient_frontier
        with pytest.raises(ValueError, match="≥ 2"):
            compute_efficient_frontier(log_rets[["AAPL"]])

    def test_efficient_frontier_structure(self, log_rets):
        from aleph_toolkit.analytics import compute_efficient_frontier
        ef = compute_efficient_frontier(log_rets, n_portfolios=200, seed=0)
        assert "min_var" in ef
        assert "max_sharpe" in ef
        assert len(ef["mc_vols"]) == 200
        assert abs(ef["min_var"]["weights"].sum() - 1.0) < 1e-6
        assert abs(ef["max_sharpe"]["weights"].sum() - 1.0) < 1e-6

    def test_correlation_matrix_shape(self, log_rets):
        from aleph_toolkit.analytics import compute_correlation_matrix
        corr = compute_correlation_matrix(log_rets)
        assert corr.shape == (3, 3)
        assert abs(corr.loc["AAPL", "AAPL"] - 1.0) < 1e-10

    def test_factor_regression_keys(self, port_rets, ff3_factors):
        from aleph_toolkit.analytics import compute_factor_regression
        reg = compute_factor_regression(port_rets, ff3_factors, "FF3")
        for key in ("betas", "se", "t_stats", "p_values", "r_squared", "alpha_annual"):
            assert key in reg

    def test_factor_regression_n_params(self, port_rets, ff3_factors):
        from aleph_toolkit.analytics import compute_factor_regression
        reg = compute_factor_regression(port_rets, ff3_factors, "FF3")
        # FF3: intercept + 3 factors = 4 parameters
        assert len(reg["betas"]) == 4
        assert len(reg["param_names"]) == 4

    def test_format_var_summary_table(self, portfolio):
        from aleph_toolkit.analytics import compute_var_results, format_var_summary_table
        vr = compute_var_results(portfolio, ["historical", "parametric"], n_sims=1000)
        df = format_var_summary_table(vr, portfolio)
        assert "VaR" in df.columns
        assert "CVaR" in df.columns
        assert len(df) == 2

    def test_format_factor_table(self, port_rets, ff3_factors):
        from aleph_toolkit.analytics import compute_factor_regression, format_factor_table
        reg = compute_factor_regression(port_rets, ff3_factors, "FF3")
        df = format_factor_table(reg)
        assert "Estimate" in df.columns
        assert "HC3 SE" in df.columns
        assert "Sig." in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backtesting
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktesting:
    @pytest.mark.parametrize("strategy,params", [
        ("SMA Crossover",     {"fast": 10, "slow": 50}),
        ("RSI Mean-Reversion",{"period": 14, "oversold": 30, "overbought": 70}),
        ("MACD",              {"fast": 12, "slow": 26, "signal": 9}),
        ("Bollinger Band",    {"window": 20, "k": 2.0}),
    ])
    def test_run_backtest_all_strategies(self, ohlcv, strategy, params):
        from aleph_toolkit.backtesting import run_backtest
        bt = run_backtest(ohlcv, strategy, params)
        assert bt["n_trades"] >= 0
        assert isinstance(bt["total_ret"], float)
        assert isinstance(bt["max_drawdown"], float)
        assert len(bt["strat_equity"]) == len(ohlcv)

    def test_run_backtest_equity_starts_at_one(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest
        bt = run_backtest(ohlcv, "SMA Crossover", {"fast": 10, "slow": 50})
        assert abs(bt["strat_equity"].iloc[0] - 1.0) < 0.01

    def test_run_backtest_raises_too_few_bars(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest
        with pytest.raises(ValueError, match="50 bars"):
            run_backtest(ohlcv.iloc[:20], "SMA Crossover", {"fast": 3, "slow": 5})

    def test_run_backtest_raises_no_close_column(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest
        with pytest.raises(ValueError, match="Close"):
            run_backtest(ohlcv.drop(columns=["Close"]), "SMA Crossover",
                         {"fast": 10, "slow": 50})

    def test_sma_raises_fast_ge_slow(self, ohlcv):
        from aleph_toolkit.backtesting import compute_sma_signals
        with pytest.raises(ValueError, match="fast"):
            compute_sma_signals(ohlcv["Close"], fast=50, slow=10)

    def test_generate_signals_unknown_strategy(self, ohlcv):
        from aleph_toolkit.backtesting import generate_signals
        with pytest.raises(ValueError, match="Unknown"):
            generate_signals(ohlcv["Close"], "Banana Strategy", {})

    def test_format_backtest_table_columns(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest, format_backtest_table
        bt = run_backtest(ohlcv, "SMA Crossover", {"fast": 10, "slow": 50})
        df = format_backtest_table(bt, "AAPL")
        assert "SMA Crossover" in df.columns
        assert "Buy & Hold" in df.columns
        assert "Total Return" in df.index

    def test_walk_forward_minimum_folds(self, ohlcv):
        from aleph_toolkit.backtesting import run_walkforward
        wf = run_walkforward(
            ohlcv, "SMA Crossover", {"fast": 10, "slow": 50},
            train_days=60, test_days=30,
        )
        assert wf["n_folds"] >= 1
        assert wf["agg"]["mean_total_ret"] is not None

    def test_walk_forward_raises_insufficient_data(self, ohlcv):
        from aleph_toolkit.backtesting import run_walkforward
        with pytest.raises(ValueError):
            run_walkforward(
                ohlcv.iloc[:50], "SMA Crossover", {"fast": 10, "slow": 50},
                train_days=252, test_days=126,
            )

    def test_format_walkforward_table(self, ohlcv):
        from aleph_toolkit.backtesting import run_walkforward, format_walkforward_aggregate_table
        wf = run_walkforward(
            ohlcv, "SMA Crossover", {"fast": 10, "slow": 50},
            train_days=60, test_days=30,
        )
        df = format_walkforward_aggregate_table(wf)
        assert "Value" in df.columns
        assert "Mean Fold Sharpe" in df.index


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stress Testing
# ─────────────────────────────────────────────────────────────────────────────

class TestStressTesting:
    @pytest.fixture
    def mock_ff_result(self):
        return {
            "factor_cols": ["Mkt-RF", "SMB", "HML"],
            "betas": np.array([0.0002, 1.05, 0.20, -0.10]),
            "model": "FF3",
        }

    def test_compute_stress_scenarios_count(self, mock_ff_result):
        from aleph_toolkit.stress_testing import compute_stress_scenarios, DEFAULT_STRESS_PRESETS
        result = compute_stress_scenarios(mock_ff_result)
        assert result["n_scenarios"] == len(DEFAULT_STRESS_PRESETS)

    def test_compute_stress_scenarios_columns(self, mock_ff_result):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        df = compute_stress_scenarios(mock_ff_result)["scenarios_df"]
        assert "Scenario" in df.columns
        assert "Impact %" in df.columns
        assert "Est. 1-Day Impact" in df.columns

    def test_compute_stress_sorted_by_abs_impact(self, mock_ff_result):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        df = compute_stress_scenarios(mock_ff_result)["scenarios_df"]
        impacts = df["Est. 1-Day Impact"].abs().values
        assert all(impacts[i] >= impacts[i+1] for i in range(len(impacts)-1))

    def test_extra_scenarios_prepended(self, mock_ff_result):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        extra = [{"name": "Custom", "shocks": {"Mkt-RF": -0.30}}]
        result = compute_stress_scenarios(mock_ff_result, extra_scenarios=extra)
        assert result["n_scenarios"] > 7
        assert "Custom" in result["scenarios_df"]["Scenario"].values

    def test_empty_presets(self, mock_ff_result):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        extra = [{"name": "Only", "shocks": {"SMB": -0.05}}]
        result = compute_stress_scenarios(mock_ff_result, presets=(), extra_scenarios=extra)
        assert result["n_scenarios"] == 1

    def test_ff5_only_factor_reported_correctly(self, mock_ff_result):
        """RMW shock on FF3 model should show '(no applicable factors in model)'."""
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        extra = [{"name": "RMW shock", "shocks": {"RMW": -0.08}}]
        df = compute_stress_scenarios(mock_ff_result, presets=(), extra_scenarios=extra)["scenarios_df"]
        assert "(no applicable factors in model)" in df["Factor Shocks"].values

    def test_missing_key_raises(self):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        with pytest.raises(ValueError, match="missing required key"):
            compute_stress_scenarios({"model": "FF3", "betas": np.array([])})

    def test_parse_custom_scenario_valid(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        sc = parse_custom_scenario("Bear", "Mkt-RF: -0.15\nSMB: -0.05")
        assert sc["name"] == "Bear"
        assert sc["shocks"]["Mkt-RF"] == pytest.approx(-0.15)
        assert sc["shocks"]["SMB"] == pytest.approx(-0.05)

    def test_parse_custom_scenario_comments_ignored(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        sc = parse_custom_scenario("Test", "# comment\nMkt-RF: -0.10\n# another")
        assert sc["shocks"] == {"Mkt-RF": pytest.approx(-0.10)}

    def test_parse_custom_scenario_empty_name_raises(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        with pytest.raises(ValueError, match="name"):
            parse_custom_scenario("", "Mkt-RF: -0.10")

    def test_parse_custom_scenario_empty_shocks_raises(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        with pytest.raises(ValueError, match="no shocks"):
            parse_custom_scenario("Test", "")

    def test_parse_custom_scenario_unknown_factor_raises(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        with pytest.raises(ValueError, match="Unknown factor"):
            parse_custom_scenario("Test", "BANANA: -0.10")

    def test_parse_custom_scenario_bad_value_raises(self):
        from aleph_toolkit.stress_testing import parse_custom_scenario
        with pytest.raises(ValueError, match="Cannot parse value"):
            parse_custom_scenario("Test", "Mkt-RF: abc")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    @pytest.fixture(scope="class")
    def var_inputs(self, portfolio, request):
        from aleph_toolkit.analytics import compute_var_results
        request.cls.portfolio = portfolio
        vr = compute_var_results(portfolio, ["historical", "montecarlo"], n_sims=1000)
        request.cls.var_result = vr
        return portfolio, vr

    def test_plot_var_comparison_is_figure(self, portfolio, var_inputs):
        from aleph_toolkit.visualization import plot_var_comparison
        _, vr = var_inputs
        fig = plot_var_comparison(
            portfolio["port_rets"], vr["method_results"], 0.99, ["AAPL","MSFT","GOOGL"]
        )
        assert isinstance(fig, go.Figure)
        # histogram + MC overlay + VaR/CVaR lines (2 methods × 2 lines = 4 vlines)
        assert len(fig.data) >= 2

    def test_plot_rolling_var_two_rows(self, portfolio):
        from aleph_toolkit.analytics import compute_rolling_var
        from aleph_toolkit.visualization import plot_rolling_var
        rv = compute_rolling_var(portfolio["port_rets"], 0.99, window=63)
        fig = plot_rolling_var(rv["rolling_df"], 0.99, 63, ["AAPL"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # fill, hist line, param line, bar chart

    def test_plot_correlation_heatmap(self, log_rets):
        from aleph_toolkit.analytics import compute_correlation_matrix
        from aleph_toolkit.visualization import plot_correlation_heatmap
        corr = compute_correlation_matrix(log_rets)
        fig = plot_correlation_heatmap(corr)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single Heatmap trace

    def test_plot_efficient_frontier(self, log_rets):
        from aleph_toolkit.analytics import compute_efficient_frontier
        from aleph_toolkit.visualization import plot_efficient_frontier
        ef = compute_efficient_frontier(log_rets, n_portfolios=200, seed=0)
        fig = plot_efficient_frontier(ef)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # scatter, assets, min_var, max_sharpe

    def test_plot_benchmark_comparison(self, portfolio, ff3_factors):
        from aleph_toolkit.visualization import plot_benchmark_comparison
        pr = portfolio["port_rets"]
        bench = ff3_factors["Mkt-RF"].rename("bench")
        common = pr.index.intersection(bench.index)
        pr_a, b_a = pr.loc[common], bench.loc[common]
        p_wealth = np.exp(pr_a.cumsum())
        b_wealth = np.exp(b_a.cumsum())
        p_peak   = p_wealth.cummax()
        b_peak   = b_wealth.cummax()
        p_dd     = (p_wealth - p_peak) / p_peak
        b_dd     = (b_wealth - b_peak) / b_peak
        fig = plot_benchmark_comparison(
            port_wealth=p_wealth, bench_wealth=b_wealth,
            port_drawdown=p_dd, bench_drawdown=b_dd,
            port_label="Portfolio", bench_symbol="SPY",
            port_ann_ret=0.10, port_ann_vol=0.15, port_mdd=0.20,
            bench_ann_ret=0.09, bench_ann_vol=0.14, bench_mdd=0.25,
        )
        assert isinstance(fig, go.Figure)

    def test_plot_factor_attribution(self, port_rets, ff3_factors):
        from aleph_toolkit.analytics import compute_factor_regression
        from aleph_toolkit.visualization import plot_factor_attribution
        reg = compute_factor_regression(port_rets, ff3_factors, "FF3")
        fig = plot_factor_attribution(reg)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single Bar trace

    def test_plot_backtest_results_trace_count(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest
        from aleph_toolkit.visualization import plot_backtest_results
        bt = run_backtest(ohlcv, "SMA Crossover", {"fast": 10, "slow": 50})
        fig = plot_backtest_results(bt, "AAPL")
        assert isinstance(fig, go.Figure)
        # price, entries, exits, 2 equity, 2 drawdown = 5–7 depending on trades
        assert len(fig.data) >= 5

    def test_plot_walkforward_results(self, ohlcv):
        from aleph_toolkit.backtesting import run_walkforward
        from aleph_toolkit.visualization import plot_walkforward_results
        wf = run_walkforward(
            ohlcv, "SMA Crossover", {"fast": 10, "slow": 50},
            train_days=60, test_days=30,
        )
        fig = plot_walkforward_results(wf, "AAPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= wf["n_folds"] * 2  # strategy + b&h per fold

    def test_plot_stress_results_trace_count(self):
        from aleph_toolkit.stress_testing import compute_stress_scenarios
        from aleph_toolkit.visualization import plot_stress_results
        mock_ff = {
            "factor_cols": ["Mkt-RF", "SMB", "HML"],
            "betas": np.array([0.0002, 1.05, 0.20, -0.10]),
            "model": "FF3",
        }
        stress = compute_stress_scenarios(mock_ff)
        fig = plot_stress_results(stress["scenarios_df"], "FF3")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single horizontal Bar trace

    def test_plot_indicator_chart(self, ohlcv):
        from aleph_toolkit.feature_engineering import compute_indicator_bundle
        from aleph_toolkit.visualization import plot_indicator_chart
        selected = frozenset(["Bollinger Bands", "RSI", "Volume"])
        bundle = compute_indicator_bundle(ohlcv, selected, {})
        fig = plot_indicator_chart("AAPL", ohlcv, bundle, selected)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 3

    def test_build_chart_config_filename(self):
        from aleph_toolkit.visualization import build_chart_config
        cfg = build_chart_config("my_custom_chart")
        assert cfg["toImageButtonOptions"]["filename"] == "my_custom_chart"
        assert cfg["displaylogo"] is False

    def test_plotly_template_override(self, ohlcv):
        from aleph_toolkit.backtesting import run_backtest
        from aleph_toolkit.visualization import plot_backtest_results
        bt = run_backtest(ohlcv, "SMA Crossover", {"fast": 10, "slow": 50})
        fig = plot_backtest_results(bt, "AAPL", plotly_template="plotly_dark")
        assert fig.layout.template.layout.plot_bgcolor != "white"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pipeline integration (mocked network)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def pipeline_result(self, prices, ff3_factors, request):
        import aleph_toolkit.pipeline as pl_mod
        from aleph_toolkit.cache import PortfolioCache
        from aleph_toolkit import pipeline as pl
        cache = PortfolioCache()
        with patch.object(pl_mod, "fetch_multi_price_series", return_value=prices), \
             patch.object(pl_mod, "fetch_ff_factors", return_value=ff3_factors), \
             patch.object(pl_mod, "fetch_price_series", return_value=None):
            result = pl.run_var_analysis(
                tickers=["AAPL","MSFT","GOOGL"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2022, 12, 31),
                alpha=0.99,
                weights=None,
                methods=["historical","parametric"],
                rf_annual=0.04,
                rolling_window=63,
                bench_symbol="",
                ff_model="FF3",
                garch_p=1,
                garch_q=1,
                cache=cache,
                n_sims=1000,
            )
        request.cls.result = result
        request.cls.cache  = cache
        return result

    def test_var_analysis_figures_present(self, pipeline_result):
        r = pipeline_result
        for key in ("var_comparison", "rolling_var", "correlation",
                    "efficient_frontier", "factor_attribution"):
            assert key in r.figures, f"Missing figure: {key}"

    def test_var_analysis_tables_present(self, pipeline_result):
        r = pipeline_result
        for key in ("var_summary", "performance", "factor_regression"):
            assert key in r.tables
            assert not r.tables[key].empty

    def test_var_analysis_portfolio_dict(self, pipeline_result):
        p = pipeline_result.portfolio
        assert "port_rets" in p
        assert "tickers" in p
        assert len(p["tickers"]) == 3

    def test_var_analysis_garch_error_captured(self, pipeline_result):
        # arch not installed in sandbox → should be in errors, not raised
        if "garch" in pipeline_result.errors:
            assert "arch" in pipeline_result.errors["garch"].lower()

    def test_ff_result_storable(self, pipeline_result):
        assert pipeline_result.ff_result is not None
        assert "betas" in pipeline_result.ff_result

    def test_run_backtest_via_pipeline(self, ohlcv):
        import aleph_toolkit.pipeline as pl_mod
        from aleph_toolkit.cache import PortfolioCache
        from aleph_toolkit import pipeline as pl
        cache = PortfolioCache()
        mock_ohlcv = {"df": ohlcv, "symbol": "AAPL", "years": 3, "from_cache": False}
        with patch.object(pl_mod, "fetch_ohlcv", return_value=mock_ohlcv):
            result = pl.run_backtest(
                symbol="AAPL",
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2022, 12, 31),
                strategy="SMA Crossover",
                strategy_params={"fast": 10, "slow": 50},
                commission=0.001,
                cache=cache,
            )
        assert isinstance(result.figure, go.Figure)
        assert not result.bt_table.empty
        assert result.bt["n_trades"] >= 0

    def test_run_walkforward_via_pipeline(self, ohlcv):
        import aleph_toolkit.pipeline as pl_mod
        from aleph_toolkit.cache import PortfolioCache
        from aleph_toolkit import pipeline as pl
        cache = PortfolioCache()
        mock_ohlcv = {"df": ohlcv, "symbol": "AAPL", "years": 3, "from_cache": False}
        with patch.object(pl_mod, "fetch_ohlcv", return_value=mock_ohlcv):
            result = pl.run_walkforward(
                symbol="AAPL",
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2022, 12, 31),
                strategy="SMA Crossover",
                strategy_params={"fast": 10, "slow": 50},
                commission=0.001,
                train_days=63,
                test_days=30,
                cache=cache,
            )
        assert result.wf["n_folds"] >= 1
        assert isinstance(result.figure, go.Figure)

    def test_run_stress_test_via_pipeline(self, pipeline_result):
        from aleph_toolkit import pipeline as pl
        ff_result = pipeline_result.ff_result
        result = pl.run_stress_test(ff_result=ff_result)
        assert isinstance(result.figure, go.Figure)
        assert result.scenarios_df.shape[0] >= 7

    def test_run_indicator_analysis_via_pipeline(self, ohlcv):
        import aleph_toolkit.pipeline as pl_mod
        from aleph_toolkit.cache import PortfolioCache
        from aleph_toolkit import pipeline as pl
        cache = PortfolioCache()
        mock_ohlcv = {"df": ohlcv, "symbol": "AAPL", "years": 2, "from_cache": False}
        with patch.object(pl_mod, "fetch_ohlcv", return_value=mock_ohlcv):
            result = pl.run_indicator_analysis(
                symbol="AAPL",
                years=2,
                selected_indicators=frozenset(["RSI", "MACD", "Volume"]),
                indicator_params={"rsi_period": 14, "macd_fast": 12,
                                  "macd_slow": 26, "macd_signal": 9},
                cache=cache,
            )
        assert isinstance(result.figure, go.Figure)
        assert "rsi" in result.indicators
