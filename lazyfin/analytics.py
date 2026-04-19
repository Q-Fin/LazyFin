"""
analytics.py — All pure quantitative computation functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _sp_minimize
from scipy.stats import norm as _sp_norm
from scipy.stats import t    as _sp_t_dist

__all__ = [
    "TRADING_DAYS_PER_YEAR", "VALID_VAR_METHODS", "FF_FACTOR_COLS",
    "compute_historical_var", "compute_parametric_var",
    "compute_montecarlo_var", "compute_cornishfisher_var",
    "compute_var_results",
    "compute_performance_metrics", "format_performance_table",
    "compute_rolling_var",
    "compute_efficient_frontier",
    "compute_correlation_matrix",
    "compute_benchmark_comparison",
    "compute_factor_regression", "format_factor_table",
    "compute_garch_model", "format_garch_table",
    "format_var_summary_table", "format_cornishfisher_diagnostics",
]

TRADING_DAYS_PER_YEAR: int = 252

VALID_VAR_METHODS: frozenset[str] = frozenset(
    {"historical", "parametric", "montecarlo", "cornishfisher"}
)

FF_FACTOR_COLS: dict[str, list[str]] = {
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_max_drawdown(port_rets: pd.Series) -> float:
    cum_wealth = np.exp(port_rets.cumsum())
    peak       = cum_wealth.cummax()
    drawdown   = (cum_wealth - peak) / peak
    return float(-drawdown.min())


def _portfolio_stats(
    weights: np.ndarray,
    mean_rets: np.ndarray,
    cov_matrix: np.ndarray,
    *,
    rf_annual: float = 0.0,
) -> tuple[float, float, float]:
    daily_ret = float(weights @ mean_rets)
    daily_var = float(weights @ cov_matrix @ weights)
    ann_ret   = float(np.exp(TRADING_DAYS_PER_YEAR * daily_ret) - 1.0)
    ann_vol   = float(np.sqrt(TRADING_DAYS_PER_YEAR * daily_var))
    sharpe    = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-12 else np.nan
    return ann_ret, ann_vol, sharpe


# ---------------------------------------------------------------------------
# VaR / CVaR — individual methods
# ---------------------------------------------------------------------------

def compute_historical_var(port_rets: pd.Series, alpha: float) -> dict:
    tail_prob   = 1.0 - alpha
    q           = port_rets.quantile(tail_prob)
    VaR         = -float(q)
    tail_losses = port_rets[port_rets <= q]
    CVaR        = float(-tail_losses.mean()) if len(tail_losses) > 0 else VaR
    return {"VaR": VaR, "CVaR": CVaR}


def compute_parametric_var(port_rets: pd.Series, alpha: float) -> dict:
    mu    = float(port_rets.mean())
    sigma = float(port_rets.std())
    z     = float(_sp_norm.ppf(1.0 - alpha))
    VaR   = -(mu + sigma * z)
    phi_z = float(_sp_norm.pdf(z))
    CVaR  = -(mu - sigma * phi_z / (1.0 - alpha))
    return {"VaR": float(VaR), "CVaR": float(CVaR)}


def compute_montecarlo_var(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float,
    *,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    mu_vec    = log_returns.mean().values.astype(float)
    cov_mat   = log_returns.cov().values.astype(float)
    rng       = np.random.default_rng(seed)
    sim_asset = rng.multivariate_normal(mu_vec, cov_mat, size=n_sims)
    port_sim  = sim_asset @ weights
    q_mc      = float(np.quantile(port_sim, 1.0 - alpha))
    VaR_mc    = -q_mc
    tail      = port_sim[port_sim <= q_mc]
    CVaR_mc   = float(-tail.mean()) if len(tail) > 0 else VaR_mc
    return {"VaR": float(VaR_mc), "CVaR": float(CVaR_mc), "sim_returns": port_sim}


def compute_cornishfisher_var(port_rets: pd.Series, alpha: float) -> dict:
    mu  = float(port_rets.mean())
    sig = float(port_rets.std())
    s   = float(port_rets.skew())
    k   = float(port_rets.kurtosis())
    z   = float(_sp_norm.ppf(1.0 - alpha))

    z_cf = (z
            + (z**2 - 1)       * s / 6.0
            + (z**3 - 3*z)     * k / 24.0
            - (2*z**3 - 5*z)   * s**2 / 36.0)

    VaR   = -(mu + sig * z_cf)

    rng    = np.random.default_rng(42)
    z_sim  = rng.standard_normal(200_000)
    z_cf_s = (z_sim
              + (z_sim**2 - 1)         * s / 6.0
              + (z_sim**3 - 3*z_sim)   * k / 24.0
              - (2*z_sim**3 - 5*z_sim) * s**2 / 36.0)
    sim_rets  = mu + sig * z_cf_s
    threshold = mu + sig * z_cf
    tail      = sim_rets[sim_rets <= threshold]
    CVaR      = float(-tail.mean()) if len(tail) > 0 else float(VaR)

    return {
        "VaR": float(VaR), "CVaR": float(CVaR),
        "skewness": s, "excess_kurtosis": k,
        "z_gaussian": float(z), "z_cf": float(z_cf),
    }


# ---------------------------------------------------------------------------
# VaR aggregator
# ---------------------------------------------------------------------------

def compute_var_results(
    portfolio: dict,
    methods: list[str],
    *,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    if not methods:
        raise ValueError("methods must be a non-empty list.")
    bad = [m for m in methods if m not in VALID_VAR_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}.")

    port_rets   = portfolio["port_rets"]
    log_returns = portfolio["log_returns"]
    weights     = portfolio["weights"]
    alpha       = portfolio.get("alpha", 0.99)

    results: dict[str, dict] = {}
    for m in methods:
        if m == "historical":
            results[m] = compute_historical_var(port_rets, alpha)
        elif m == "parametric":
            results[m] = compute_parametric_var(port_rets, alpha)
        elif m == "montecarlo":
            results[m] = compute_montecarlo_var(log_returns, weights, alpha,
                                                n_sims=n_sims, seed=seed)
        elif m == "cornishfisher":
            results[m] = compute_cornishfisher_var(port_rets, alpha)

    first = results[methods[0]]
    return {
        "method_results": results,
        "primary_VaR":    first["VaR"],
        "primary_CVaR":   first["CVaR"],
        "alpha":          alpha,
        "methods_run":    list(results.keys()),
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    port_rets: pd.Series,
    *,
    rf_annual: float = 0.0,
) -> dict:
    ann      = TRADING_DAYS_PER_YEAR
    rf_daily = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    ann_ret  = float(np.exp(port_rets.mean() * ann) - 1.0)
    ann_vol  = float(port_rets.std(ddof=1) * np.sqrt(ann))
    sharpe   = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    excess   = port_rets - rf_daily
    neg_exc  = excess.clip(upper=0.0)
    semi_std = float(np.sqrt((neg_exc ** 2).mean()) * np.sqrt(ann))
    sortino  = (ann_ret - rf_annual) / semi_std if semi_std > 0 else np.nan
    mdd      = _compute_max_drawdown(port_rets)
    calmar   = (ann_ret / mdd) if mdd > 1e-8 else np.nan
    cum_ret  = float(np.exp(port_rets.sum()) - 1.0)
    return {
        "ann_ret":           ann_ret,
        "ann_vol":           ann_vol,
        "sharpe":            sharpe,
        "sortino":           sortino,
        "max_drawdown":      mdd,
        "calmar":            calmar,
        "cum_ret":           cum_ret,
        "best_day":          float(port_rets.max()),
        "worst_day":         float(port_rets.min()),
        "pct_positive_days": float((port_rets > 0).mean()),
        "n_days":            len(port_rets),
        "rf_annual":         rf_annual,
    }


def format_performance_table(metrics: dict) -> pd.DataFrame:
    rf_pct = f"{metrics['rf_annual']:.2%}"
    def _f(v, fmt=".3f"):
        return f"{v:{fmt}}" if (v is not None and not np.isnan(float(v))) else "N/A"
    rows = [
        ("Annualised Return",                    f"{metrics['ann_ret']:+.2%}"),
        ("Annualised Volatility",                f"{metrics['ann_vol']:.2%}"),
        (f"Sharpe Ratio  (Rf={rf_pct})",         _f(metrics["sharpe"])),
        (f"Sortino Ratio (Rf={rf_pct})",         _f(metrics["sortino"])),
        ("Max Drawdown",                         f"{-metrics['max_drawdown']:.2%}"),
        ("Calmar Ratio",                         _f(metrics["calmar"])),
        ("Cumulative Return",                    f"{metrics['cum_ret']:+.2%}"),
        ("Best Single Day",                      f"{metrics['best_day']:+.5f}"),
        ("Worst Single Day",                     f"{metrics['worst_day']:+.5f}"),
        ("% Positive Days",                      f"{metrics['pct_positive_days']:.1%}"),
        ("Trading Days in Sample",               f"{metrics['n_days']:,}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


# ---------------------------------------------------------------------------
# Rolling VaR
# ---------------------------------------------------------------------------

def compute_rolling_var(
    port_rets: pd.Series,
    alpha: float,
    *,
    window: int = 252,
) -> dict:
    if len(port_rets) < window:
        raise ValueError(
            f"len(port_rets)={len(port_rets)} < window={window}."
        )
    tail_prob = 1.0 - alpha
    z_alpha   = float(_sp_norm.ppf(tail_prob))

    rolling = port_rets.rolling(window=window, min_periods=window)
    hist_var  = rolling.apply(lambda arr: -np.quantile(arr, tail_prob), raw=True)
    def _param(arr):
        mu, sigma = arr.mean(), arr.std(ddof=1)
        return -(mu + sigma * z_alpha)
    param_var = rolling.apply(_param, raw=True)

    df = pd.DataFrame({
        "Historical VaR":   hist_var,
        "Parametric VaR":   param_var,
        "Portfolio Return":  port_rets,
    })
    tickers = [str(port_rets.name)] if port_rets.name else ["portfolio"]
    return {"rolling_df": df, "alpha": alpha, "window": window, "tickers": tickers}


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

def compute_efficient_frontier(
    log_returns: pd.DataFrame,
    *,
    rf_annual: float = 0.0,
    n_portfolios: int = 3_000,
    seed: int = 42,
    cache=None,
) -> dict:
    """
    Adaptive portfolio count:  scales n_portfolios down for large ticker
    sets so computation stays under ~150 ms even with 50 assets.

    Memoisation:  if a PortfolioCache is supplied, the result is keyed on a
    hash of the covariance matrix + parameters.  Identical data → instant
    return on subsequent calls within the same session.
    """
    # Adaptive n_portfolios: cap at 500 for > 20 assets, 1000 for > 10
    n_assets = log_returns.shape[1]
    if n_assets > 20 and n_portfolios > 500:
        n_portfolios = 500
    elif n_assets > 10 and n_portfolios > 1_000:
        n_portfolios = 1_000

    # Cache lookup
    if cache is not None:
        cached = cache.get_ef(log_returns, rf_annual, n_portfolios, seed)
        if cached is not None:
            return cached
    if log_returns.shape[1] < 2:
        raise ValueError("Efficient frontier requires ≥ 2 assets.")
    tickers    = list(log_returns.columns)
    n          = len(tickers)
    mean_rets  = log_returns.mean().values.astype(float)
    cov_matrix = log_returns.cov().values.astype(float)

    rng        = np.random.default_rng(seed)
    raw        = rng.dirichlet(np.ones(n), size=n_portfolios)
    mc_vols    = np.empty(n_portfolios)
    mc_rets    = np.empty(n_portfolios)
    mc_sharpes = np.empty(n_portfolios)

    for i, w in enumerate(raw):
        r, v, s       = _portfolio_stats(w, mean_rets, cov_matrix, rf_annual=rf_annual)
        mc_rets[i]    = r
        mc_vols[i]    = v
        mc_sharpes[i] = s if not np.isnan(s) else 0.0

    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    w0          = np.repeat(1.0 / n, n)

    # Minimum-variance
    res_mv = _sp_minimize(
        lambda w: float(w @ cov_matrix @ w), w0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w_mv = res_mv.x / res_mv.x.sum()
    r_mv, v_mv, s_mv = _portfolio_stats(w_mv, mean_rets, cov_matrix, rf_annual=rf_annual)
    min_var = {"weights": w_mv, "ann_ret": r_mv, "ann_vol": v_mv, "sharpe": s_mv}

    # Maximum-Sharpe
    def _neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mean_rets, cov_matrix, rf_annual=rf_annual)
        return -(r - rf_annual) / v if v > 1e-12 else 1e6

    res_ms = _sp_minimize(
        _neg_sharpe, w0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w_ms = res_ms.x / res_ms.x.sum()
    r_ms, v_ms, s_ms = _portfolio_stats(w_ms, mean_rets, cov_matrix, rf_annual=rf_annual)
    max_sharpe = {"weights": w_ms, "ann_ret": r_ms, "ann_vol": v_ms, "sharpe": s_ms}

    asset_rets = np.array([float(np.exp(TRADING_DAYS_PER_YEAR * mu) - 1.0)
                           for mu in mean_rets])
    asset_vols = np.array([float(np.sqrt(TRADING_DAYS_PER_YEAR * cov_matrix[i, i]))
                           for i in range(n)])

    result = {
        "mc_vols":    mc_vols,
        "mc_rets":    mc_rets,
        "mc_sharpes": mc_sharpes,
        "mc_weights": raw,
        "min_var":    min_var,
        "max_sharpe": max_sharpe,
        "asset_rets": asset_rets,
        "asset_vols": asset_vols,
        "tickers":    tickers,
        "rf_annual":  rf_annual,
    }
    if cache is not None:
        cache.put_ef(log_returns, rf_annual, n_portfolios, seed, result)
    return result


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def compute_correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    clean = log_returns.dropna()
    if len(clean) < 2:
        raise ValueError("Need ≥ 2 non-NaN rows to compute correlation.")
    return clean.corr()


# ---------------------------------------------------------------------------
# Benchmark comparison  (computation only — no figures)
# ---------------------------------------------------------------------------

def compute_benchmark_comparison(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    *,
    bench_symbol: str,
    port_label: str = "Portfolio",
) -> dict:
    common = port_rets.index.intersection(bench_rets.index)
    if common.empty:
        raise ValueError("No overlapping dates between portfolio and benchmark.")
    p, b      = port_rets.loc[common], bench_rets.loc[common]
    ann       = TRADING_DAYS_PER_YEAR
    p_wealth  = np.exp(p.cumsum())
    b_wealth  = np.exp(b.cumsum())
    p_peak    = p_wealth.cummax()
    b_peak    = b_wealth.cummax()
    p_dd      = (p_wealth - p_peak) / p_peak
    b_dd      = (b_wealth - b_peak) / b_peak
    return {
        "port_wealth":   p_wealth,
        "bench_wealth":  b_wealth,
        "port_drawdown": p_dd,
        "bench_drawdown":b_dd,
        "port_ann_ret":  float(np.exp(p.mean() * ann) - 1.0),
        "port_ann_vol":  float(p.std() * np.sqrt(ann)),
        "port_mdd":      float(-p_dd.min()),
        "bench_ann_ret": float(np.exp(b.mean() * ann) - 1.0),
        "bench_ann_vol": float(b.std() * np.sqrt(ann)),
        "bench_mdd":     float(-b_dd.min()),
        "bench_symbol":  bench_symbol,
        "port_label":    port_label,
    }


# ---------------------------------------------------------------------------
# Fama-French factor regression
# ---------------------------------------------------------------------------

def compute_factor_regression(
    port_rets: pd.Series,
    ff_factors: pd.DataFrame,
    model: str,
) -> dict:
    model      = model.upper()
    factor_cols = FF_FACTOR_COLS.get(model, FF_FACTOR_COLS["FF3"])
    common     = port_rets.index.intersection(ff_factors.index)
    if len(common) < len(factor_cols) + 10:
        raise ValueError(
            f"Only {len(common)} overlapping dates. Extend the date range."
        )
    r_p = port_rets.loc[common]
    ff  = ff_factors.loc[common]
    r_f = ff["RF"]
    y   = (r_p - r_f).values.astype(float)

    X_fac = ff[factor_cols].values.astype(float)
    X     = np.column_stack([np.ones(len(y)), X_fac])
    n, k  = X.shape

    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    betas   = XtX_inv @ (X.T @ y)
    y_hat   = X @ betas
    e       = y - y_hat

    h_diag  = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h_diag  = np.clip(h_diag, 0.0, 1.0 - 1e-10)
    e_tilde = e / (1.0 - h_diag)
    meat    = (X * e_tilde[:, np.newaxis]).T @ (X * e_tilde[:, np.newaxis])
    V_hc3   = XtX_inv @ meat @ XtX_inv
    se      = np.sqrt(np.diag(V_hc3))

    t_stats  = betas / se
    p_values = 2.0 * _sp_t_dist.sf(np.abs(t_stats), df=n - k)

    ss_res  = float(e @ e)
    ss_tot  = float(((y - y.mean()) ** 2).sum())
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2  = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if ss_tot > 0 else np.nan

    return {
        "model":         model,
        "param_names":   ["Alpha (daily)"] + factor_cols,
        "betas":         betas,
        "se":            se,
        "t_stats":       t_stats,
        "p_values":      p_values,
        "r_squared":     float(r2),
        "adj_r_squared": float(adj_r2),
        "n_obs":         n,
        "factor_cols":   factor_cols,
        "alpha_annual":  float(betas[0] * 252),
        "excess_ret":    r_p - r_f,
        "ff_factors":    ff,
        "common_dates":  common,
    }


def format_factor_table(reg: dict) -> pd.DataFrame:
    def sig(p):
        if p < 0.001: return "***"
        if p < 0.010: return "**"
        if p < 0.050: return "*"
        if p < 0.100: return "."
        return ""
    rows = []
    for i, name in enumerate(reg["param_names"]):
        rows.append({
            "Parameter": name,
            "Estimate":  f"{reg['betas'][i]:+.6f}",
            "HC3 SE":    f"{reg['se'][i]:.6f}",
            "t-stat":    f"{reg['t_stats'][i]:+.3f}",
            "p-value":   f"{reg['p_values'][i]:.4f}",
            "Sig.":      sig(reg["p_values"][i]),
        })
    return pd.DataFrame(rows).set_index("Parameter")


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

def compute_garch_model(
    port_rets: pd.Series,
    *,
    p: int = 1,
    q: int = 1,
) -> dict:
    try:
        from arch import arch_model as _arch_model
    except ImportError:
        raise RuntimeError(
            "The 'arch' package is required for GARCH modelling.\n"
            "Install it with:  pip install arch\n"
            "Then restart and re-run."
        )
    if len(port_rets) < 50:
        raise ValueError(f"Need ≥ 50 observations for GARCH; got {len(port_rets)}.")

    rets_pct = port_rets * 100.0
    model    = _arch_model(rets_pct, vol="Garch", p=p, q=q,
                           dist="normal", rescale=False)
    result   = model.fit(disp="off", show_warning=False)

    params   = dict(result.params)
    cond_vol = result.conditional_volatility / 100.0
    cond_vol.name = "conditional_volatility"

    fcast     = result.forecast(horizon=22, reindex=False)
    fcast_var = fcast.variance.values[-1]
    fcast_vol = np.sqrt(fcast_var) / 100.0

    alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta_sum  = sum(v for k, v in params.items() if k.startswith("beta"))
    persistence = float(alpha_sum + beta_sum)

    omega = float(params.get("omega", np.nan)) / 10_000.0
    if persistence < 1.0 and omega > 0:
        lr_var_daily  = omega / (1.0 - persistence)
        lr_vol_daily  = float(np.sqrt(lr_var_daily))
        lr_vol_annual = float(lr_vol_daily * np.sqrt(252))
    else:
        lr_vol_daily  = np.nan
        lr_vol_annual = np.nan

    return {
        "p": p, "q": q,
        "params":              params,
        "cond_vol":            cond_vol,
        "fcast_vol":           fcast_vol,
        "fcast_horizon":       22,
        "persistence":         persistence,
        "long_run_vol_daily":  lr_vol_daily,
        "long_run_vol_annual": lr_vol_annual,
        "aic":                 float(result.aic),
        "bic":                 float(result.bic),
        "n_obs":               len(port_rets),
    }


def format_garch_table(garch: dict) -> pd.DataFrame:
    p, q     = garch["p"], garch["q"]
    pers     = garch["persistence"]
    lr_ann   = garch["long_run_vol_annual"]
    fcast_ann = float(np.mean(garch["fcast_vol"]) * np.sqrt(252))

    def _fv(v):
        return f"{v:.6f}" if (v is not None and not np.isnan(v)) else "N/A"

    params = garch["params"]
    param_rows = [
        (f"  {k}", _fv(params[k]))
        for k in ["omega", "alpha[1]", "beta[1]", "alpha[2]", "beta[2]"]
        if k in params
    ]
    rows = (
        [(f"Model", f"GARCH({p},{q})"), ("Observations", f"{garch['n_obs']:,}")]
        + param_rows
        + [
            ("Persistence (α+β)",
             f"{pers:.6f}" + (" [⚠ non-stationary]" if pers >= 1 else "")),
            ("Long-run vol (daily)",
             f"{garch['long_run_vol_daily']*100:.4f}%" if not np.isnan(lr_ann) else "N/A"),
            ("Long-run vol (ann.)",
             f"{lr_ann*100:.2f}%" if not np.isnan(lr_ann) else "N/A"),
            ("22-day fcast vol (ann.)", f"{fcast_ann*100:.2f}%"),
            ("AIC", f"{garch['aic']:.4f}"),
            ("BIC", f"{garch['bic']:.4f}"),
        ]
    )
    return pd.DataFrame(rows, columns=["Statistic", "Value"]).set_index("Statistic")


# ---------------------------------------------------------------------------
# Table formatters for VaR
# ---------------------------------------------------------------------------

_METHOD_LABELS: dict[str, str] = {
    "historical":    "Historical",
    "parametric":    "Parametric (Gaussian)",
    "montecarlo":    "Monte Carlo",
    "cornishfisher": "Cornish-Fisher",
}


def format_var_summary_table(var_result: dict, portfolio: dict) -> pd.DataFrame:
    rows = []
    for m in var_result["methods_run"]:
        r = var_result["method_results"][m]
        rows.append({
            "Method": _METHOD_LABELS.get(m, m),
            "VaR":    round(r["VaR"],  6),
            "CVaR":   round(r["CVaR"], 6),
            "VaR %":  f"{r['VaR']  * 100:.3f}%",
            "CVaR %": f"{r['CVaR'] * 100:.3f}%",
        })
    return pd.DataFrame(rows).set_index("Method")


def format_cornishfisher_diagnostics(cf_result: dict) -> pd.DataFrame:
    rows = [
        ("Skewness",         f"{cf_result['skewness']:+.4f}"),
        ("Excess Kurtosis",  f"{cf_result['excess_kurtosis']:+.4f}"),
        ("Gaussian z",       f"{cf_result['z_gaussian']:.4f}"),
        ("CF-adjusted z",    f"{cf_result['z_cf']:.4f}"),
        ("z shift",          f"{cf_result['z_cf'] - cf_result['z_gaussian']:+.4f}"),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"]).set_index("Statistic")
