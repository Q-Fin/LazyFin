"""
stress_testing.py — Portfolio stress testing via FF factor shocks.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = [
    "VALID_FACTOR_NAMES", "DEFAULT_STRESS_PRESETS", "StressScenario",
    "parse_custom_scenario", "compute_stress_scenarios",
]


VALID_FACTOR_NAMES: frozenset[str] = frozenset(
    {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}
)

# Type alias
StressScenario = dict  # {'name': str, 'shocks': dict[str, float]}

DEFAULT_STRESS_PRESETS: tuple[StressScenario, ...] = (
    {"name": "Equity Crash (−20%)",      "shocks": {"Mkt-RF": -0.20}},
    {"name": "Flight to Quality",         "shocks": {"Mkt-RF": -0.10, "HML": +0.05}},
    {"name": "Small-Cap Selloff",         "shocks": {"SMB": -0.15}},
    {"name": "Value Rally",               "shocks": {"HML": +0.12}},
    {"name": "Growth Selloff",            "shocks": {"HML": -0.12, "SMB": -0.05}},
    {"name": "Profitability Shock (FF5)", "shocks": {"RMW": -0.08}},
    {"name": "Investment Shock (FF5)",    "shocks": {"CMA": -0.08}},
)


def parse_custom_scenario(name: str, shocks_text: str) -> StressScenario:
    name_raw   = name.strip()
    shocks_raw = shocks_text.strip()

    if not name_raw:
        raise ValueError("Custom scenario: please supply a scenario name.")
    if not shocks_raw:
        raise ValueError(f"Custom scenario '{name_raw}': no shocks entered.")

    shocks: dict[str, float] = {}
    for line in shocks_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Cannot parse line '{line}'. "
                "Expected format: FactorName: decimal_value"
            )
        factor_str, val_str = line.split(":", 1)
        factor = factor_str.strip()
        if factor not in VALID_FACTOR_NAMES:
            raise ValueError(
                f"Unknown factor '{factor}'. "
                f"Valid names: {sorted(VALID_FACTOR_NAMES)}"
            )
        try:
            shocks[factor] = float(val_str.strip())
        except ValueError:
            raise ValueError(
                f"Cannot parse value for {factor}: '{val_str.strip()}'. "
                "Enter a decimal (e.g. -0.05 for −5%)."
            )

    if not shocks:
        raise ValueError(f"Custom scenario '{name_raw}': no valid shocks parsed.")
    return {"name": name_raw, "shocks": shocks}


def compute_stress_scenarios(
    reg: dict,
    *,
    presets: tuple[StressScenario, ...] = DEFAULT_STRESS_PRESETS,
    extra_scenarios: Optional[list[StressScenario]] = None,
) -> dict:
    for key in ("factor_cols", "betas", "model"):
        if key not in reg:
            raise ValueError(
                f"ff_result missing required key '{key}'. "
                "Run analytics.compute_factor_regression() first."
            )

    factor_cols = reg["factor_cols"]
    # betas[0] = alpha; betas[1:] = factor loadings in factor_cols order
    betas = {fc: float(reg["betas"][i + 1]) for i, fc in enumerate(factor_cols)}

    all_scenarios: list[StressScenario] = []
    if extra_scenarios:
        all_scenarios.extend(extra_scenarios)
    all_scenarios.extend(presets)

    rows = []
    for sc in all_scenarios:
        name   = sc["name"]
        shocks = sc["shocks"]
        impact = sum(
            betas.get(factor, 0.0) * shock
            for factor, shock in shocks.items()
        )
        shock_str = ",  ".join(
            f"{f}: {v:+.1%}" for f, v in shocks.items() if f in factor_cols
        )
        if not shock_str:
            shock_str = "(no applicable factors in model)"
        rows.append({
            "Scenario":          name,
            "Factor Shocks":     shock_str,
            "Est. 1-Day Impact": round(impact, 6),
            "Impact %":          f"{impact * 100:+.3f}%",
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("Est. 1-Day Impact", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    return {
        "scenarios_df": df,
        "model":        reg["model"],
        "n_scenarios":  len(df),
    }
