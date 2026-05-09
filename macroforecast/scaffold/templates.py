"""Starter recipe templates -- jumpstart for new users.

Each template returns a fully-populated :class:`RecipeBuilder` that the
caller can either use directly (``b.run("out/")``) or override fields
on (``b.l1.target = "INDPRO"``) before running.

Five templates ship in v1.0:

* ``ridge_baseline`` -- single-target ridge on a custom panel.
* ``horse_race_md`` -- ridge / random-forest / lasso on FRED-MD.
* ``regime_conditional`` -- Hamilton MS regime + per-regime fit.
* ``fred_md_replication`` -- McCracken-Ng (2016) FRED-MD baseline.
* ``fred_sd_geographic`` -- state-level forecasting + choropleth output.
"""
from __future__ import annotations

from typing import Any, Callable

from .builder import RecipeBuilder


def ridge_baseline(*, target: str = "y") -> RecipeBuilder:
    """Ridge regression on a tiny inline custom panel.

    Smallest end-to-end recipe -- runs in under a second on any laptop.
    Useful as the wizard's quick-start path and as a smoke test.
    """

    panel: dict[str, list[Any]] = {
        "date": [f"2018-{m:02d}-01" for m in range(1, 13)],
        "y": [float(v) for v in range(1, 13)],
        "x1": [0.5 * v for v in range(1, 13)],
    }
    b = RecipeBuilder()
    b.l0(random_seed=0)
    b.l1.custom_panel(target=target, panel=panel)
    b.l2.no_op()
    b.l3.lag_only(n_lag=1)
    b.l4.fit("ridge", alpha=0.1, min_train_size=4)
    b.l5.standard()
    return b


def horse_race_md(*, target: str = "CPIAUCSL") -> RecipeBuilder:
    """Three-family horse race on FRED-MD with the ridge baseline marked
    as benchmark for relative-MSE reporting."""

    b = RecipeBuilder()
    b.l0(random_seed=42)
    b.l1.fred_md(target=target)
    b.l2.standard()
    b.l3.lag_only(n_lag=6)
    b.l4.fit("ridge", alpha=1.0).is_benchmark()
    b.l4.fit("random_forest", n_estimators=100, max_depth=6)
    b.l4.fit("lasso", alpha=0.05)
    b.l5.standard(point_metrics=["mse", "rmse", "mae"])
    return b


def regime_conditional(*, target: str = "INDPRO") -> RecipeBuilder:
    """Hamilton (1989) Markov-switching regime + ridge fit. Demonstrates
    the L1.G regime axis and the cross-layer reference into L4 / L5.
    """

    b = RecipeBuilder()
    b.l0(random_seed=42)
    b.l1.fred_md(
        target=target,
        regime_definition="estimated_markov_switching",
        regime_estimation_temporal_rule="expanding_window_per_origin",
        n_regimes=2,
    )
    b.l2.standard()
    b.l3.lag_only(n_lag=6)
    b.l4.fit("ridge", alpha=1.0)
    b.l5.standard()
    return b


def fred_md_replication(*, target: str = "CPIAUCSL", horizon_set: str = "standard_md") -> RecipeBuilder:
    """McCracken-Ng (2016) FRED-MD baseline configuration.

    Apply official t-codes, IQR outliers, EM-factor imputation, expanding
    walk-forward ridge with the bundled standard horizons. Ships as the
    canonical "did we install macroforecast correctly?" smoke test on real
    macro data.
    """

    b = RecipeBuilder()
    b.l0(random_seed=2016)  # year of the McCracken-Ng paper
    b.l1.fred_md(target=target, horizon_set=horizon_set)
    b.l2.standard()
    b.l3.lag_only(n_lag=12)
    b.l4.fit("ridge", alpha=1.0).is_benchmark()
    b.l4.fit("ar_p", n_lag=12)
    b.l5.standard(point_metrics=["mse", "rmse", "mae"])
    b.l6.set_axis(enabled=True)
    return b


def fred_sd_geographic(*, target: str = "PAYEMS") -> RecipeBuilder:
    """FRED-SD state-level setup. Pairs with the L7 ``us_state_choropleth``
    figure type for geographic-importance visualisation."""

    b = RecipeBuilder()
    b.l0(random_seed=42)
    b.l1(
        custom_source_policy="official_only",
        dataset="fred_sd",
        frequency="monthly",
        horizon_set="standard_md",
        target_structure="single_target",
        target=target,
        target_geography_scope="all_states",
        predictor_geography_scope="all_states",
        fred_sd_state_group="all_states",
    )
    b.l2.standard()
    b.l3.lag_only(n_lag=6)
    b.l4.fit("ridge", alpha=1.0)
    b.l5.standard()
    b.l7.set_axis(enabled=True, figure_type="us_state_choropleth")
    return b


_TEMPLATES: dict[str, Callable[..., RecipeBuilder]] = {
    "ridge_baseline": ridge_baseline,
    "horse_race_md": horse_race_md,
    "regime_conditional": regime_conditional,
    "fred_md_replication": fred_md_replication,
    "fred_sd_geographic": fred_sd_geographic,
}


def from_template(name: str, **overrides: Any) -> RecipeBuilder:
    """Look up a starter template by name and return its
    :class:`RecipeBuilder`. ``overrides`` are forwarded to the
    template factory."""

    if name not in _TEMPLATES:
        raise KeyError(
            f"unknown template {name!r}; available: {sorted(_TEMPLATES)}"
        )
    return _TEMPLATES[name](**overrides)


def list_templates() -> tuple[str, ...]:
    """Return the registered template names in insertion order."""

    return tuple(_TEMPLATES)


__all__ = [
    "from_template",
    "fred_md_replication",
    "fred_sd_geographic",
    "horse_race_md",
    "list_templates",
    "regime_conditional",
    "ridge_baseline",
]
