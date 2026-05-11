"""Phase B-9 paper-9 (Goulet Coulombe / Frenette / Klieber 2025 JAE --
"Hemisphere Neural Networks", HNN) helper-rewrite tests.

Round 1 audit found that the paper's distributional head was dead
code from ``macroforecast.run``:

* **F1 (CRITICAL)** -- ``runtime._emit_quantile_intervals`` routed by
  family string via ``_native_quantile_engine``. The ``mlp`` family had
  no native quantile engine, so HNN forecasts fell through to the
  sklearn ``LinearRegression`` Gaussian-residual sigma path. HNN's
  ``predict_quantiles`` / ``predict_distribution`` / ``predict_variance``
  -- the paper's Eq. 10 reality-checked variance head -- were never
  invoked on the public path.
* **F2 (CRITICAL)** -- ``forecast_object='density'`` produced empty
  intervals. Only ``forecast_object='quantile'`` triggered an interval
  emission; the density branch fell through silently.
* **F5 (Medium)** -- the helper exposed only ``target / horizon /
  panel / seed``; paper hyperparameters (``B``, ``neurons``, ``lc/lm/lv``,
  ``nu``, ``lambda_emphasis``, ``n_epochs``, ``dropout``, ``lr``,
  ``sub_rate``, ``quantile_levels``, ``forecast_object``) were not
  surfaceable, and the docstring still claimed "Status: pre-promotion".

The five tests below close F1 + F2 + F5. Reference: Goulet Coulombe /
Frenette / Klieber (2025) "Hemisphere Neural Networks", JAE; Eq. 1
(Gaussian density), Eq. 8 (blocked-OOB ensemble), Eq. 9-10 (variance
reality check), §3.2 (volatility emphasis).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast.recipes.paper_methods import hemisphere_neural_network


# ----------------------------------------------------------------------
# Heteroscedastic DGP fixture (paper §4 simulation regime)
# ----------------------------------------------------------------------


def _build_hetero_panel(T: int = 80, K: int = 3, seed: int = 0):
    """Heteroscedastic panel: σ_t depends on x1 so HNN's variance head
    has a real signal to learn. ``LinearRegression`` Gaussian-residual
    sigma would NOT pick this up (its sigma is a single scalar)."""

    rng = np.random.default_rng(seed)
    import datetime

    dates: list[str] = []
    d = datetime.date(2014, 1, 1)
    for _ in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y_yr = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y_yr, m, 1)

    x1 = rng.normal(0.0, 1.0, T)
    x2 = rng.normal(0.0, 1.0, T)
    x3 = rng.normal(0.0, 1.0, T)
    # Heteroscedastic noise: variance grows with |x1|.
    sigma_t = 0.1 + 1.5 * np.abs(x1)
    y = 0.5 * x1 + 0.3 * x2 + rng.normal(0.0, sigma_t)
    panel = {
        "date": dates,
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
        "x3": x3.tolist(),
    }
    return panel


# ----------------------------------------------------------------------
# Test 1 -- F5 helper exposes paper hyperparameters with paper defaults
# ----------------------------------------------------------------------


def test_hnn_helper_exposes_paper_hyperparameters():
    """Phase B-9 F5: helper signature includes the paper hyperparameters
    with paper-faithful defaults (B=1000 from paper p.12, neurons=400
    from paper §3, hemisphere depths 2-2-2, density forecast-object)."""

    sig = inspect.signature(hemisphere_neural_network)
    params = sig.parameters

    for name in (
        "B",
        "neurons",
        "lc",
        "lm",
        "lv",
        "nu",
        "lambda_emphasis",
        "n_epochs",
        "dropout",
        "lr",
        "sub_rate",
        "quantile_levels",
        "forecast_object",
    ):
        assert name in params, f"helper missing paper hyperparameter '{name}'"

    assert params["B"].default == 1000, "paper p.12 B=1000"
    assert params["neurons"].default == 400, "paper §3 neurons=400"
    assert params["lc"].default == 2, "paper §3 lc=2"
    assert params["lm"].default == 2, "paper §3 lm=2"
    assert params["lv"].default == 2, "paper §3 lv=2"
    assert params["nu"].default is None
    assert params["lambda_emphasis"].default == 1.0
    assert params["n_epochs"].default == 200
    assert params["dropout"].default == 0.2
    assert params["lr"].default == 0.01
    assert params["sub_rate"].default == 0.80, "paper Eq. 8 sub_rate=0.80"
    assert tuple(params["quantile_levels"].default) == (0.05, 0.16, 0.84, 0.95)
    assert params["forecast_object"].default == "density"

    # And the recipe must forward them into fit_node params.
    recipe = hemisphere_neural_network(
        B=10,
        neurons=16,
        n_epochs=20,
        forecast_object="quantile",
    )
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit["params"]["B"] == 10
    assert fit["params"]["neurons"] == 16
    assert fit["params"]["n_epochs"] == 20
    assert fit["params"]["forecast_object"] == "quantile"
    assert fit["params"]["architecture"] == "hemisphere"
    assert fit["params"]["loss"] == "volatility_emphasis"


# ----------------------------------------------------------------------
# Test 2 -- F1 procedure-level: HNN dispatch invokes predict_quantiles
# ----------------------------------------------------------------------


def test_hnn_l4_dispatch_invokes_predict_quantiles_not_linear_regression():
    """Phase B-9 F1: with a heteroscedastic DGP, HNN's per-row variance
    head should make band widths VARY across origins. The
    ``LinearRegression`` Gaussian-residual fallback would yield
    near-constant widths (one scalar sigma applied to every origin)."""

    pytest.importorskip("torch")

    panel = _build_hetero_panel(T=64, K=3, seed=0)
    recipe = hemisphere_neural_network(
        target="y",
        horizon=1,
        panel=panel,
        B=10,
        neurons=16,
        n_epochs=20,
        lc=1,
        lm=1,
        lv=1,
        forecast_object="quantile",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    assert result.cells, "recipe should produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts
    forecasts_artifact = artifacts["l4_forecasts_v1"]
    intervals = getattr(forecasts_artifact, "forecast_intervals", {}) or {}
    assert intervals, "F1 fix: forecast_intervals must be populated for HNN"

    # Group bands by origin: width = q0.84 - q0.16.
    widths_by_origin: dict[object, float] = {}
    grouped: dict[object, dict[float, float]] = {}
    for (model_id, target, horizon, origin, q), value in intervals.items():
        # Skip density sentinel keys (negative floats).
        if q < 0:
            continue
        grouped.setdefault(origin, {})[float(q)] = float(value)
    for origin, qmap in grouped.items():
        # Use the closest available high/low pair.
        levels = sorted(qmap.keys())
        if not levels:
            continue
        lo = qmap[levels[0]]
        hi = qmap[levels[-1]]
        widths_by_origin[origin] = hi - lo

    assert len(widths_by_origin) >= 5, (
        f"need multiple origins to assess heteroscedasticity, got "
        f"{len(widths_by_origin)}"
    )
    widths = np.asarray(list(widths_by_origin.values()), dtype=float)
    # The ``LinearRegression`` Gaussian-residual fallback would produce
    # widths within ~1e-12 of each other (single scalar sigma). HNN's
    # per-row variance head must yield positive cross-origin variation.
    assert float(np.std(widths)) > 1e-6, (
        f"HNN per-row variance head should produce varying widths; "
        f"std(widths)={float(np.std(widths))}"
    )


# ----------------------------------------------------------------------
# Test 3 -- F2: forecast_object='density' populates intervals with
#               mean + variance
# ----------------------------------------------------------------------


def test_hnn_forecast_object_density_populates_intervals():
    """Phase B-9 F2: ``forecast_object='density'`` must populate
    ``forecast_intervals`` with the paper's predictive distribution
    (per-row mean + variance + Gaussian quantile bands). Without F2 the
    density branch fell through and intervals stayed empty."""

    pytest.importorskip("torch")

    from macroforecast.core.runtime import (
        DENSITY_MEAN_KEY,
        DENSITY_VARIANCE_KEY,
    )

    panel = _build_hetero_panel(T=64, K=3, seed=1)
    recipe = hemisphere_neural_network(
        target="y",
        horizon=1,
        panel=panel,
        B=10,
        neurons=16,
        n_epochs=20,
        lc=1,
        lm=1,
        lv=1,
        forecast_object="density",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    assert result.cells
    artifacts = result.cells[0].runtime_result.artifacts
    forecasts_artifact = artifacts["l4_forecasts_v1"]
    intervals = getattr(forecasts_artifact, "forecast_intervals", {}) or {}
    assert intervals, "F2 fix: density forecast_intervals must be populated"

    # Pick an origin and check mean / variance keys are present.
    origins = {key[3] for key in intervals}
    assert origins, "intervals should cover at least one origin"

    means_by_origin: dict[object, float] = {}
    vars_by_origin: dict[object, float] = {}
    for (model_id, target, horizon, origin, q), value in intervals.items():
        if q == DENSITY_MEAN_KEY:
            means_by_origin[origin] = float(value)
        elif q == DENSITY_VARIANCE_KEY:
            vars_by_origin[origin] = float(value)

    assert means_by_origin, "density branch must emit mean per origin"
    assert vars_by_origin, "density branch must emit variance per origin"
    assert set(means_by_origin) == set(vars_by_origin), (
        "mean + variance must be co-emitted at every origin"
    )
    # Variances strictly positive (paper softplus head + Eq. 10 floor).
    assert all(v > 0 for v in vars_by_origin.values()), (
        "Eq. 10 reality-checked variance must be strictly positive"
    )
    # Means finite.
    assert all(np.isfinite(m) for m in means_by_origin.values())

    # The forecast_object on the artifact should advertise 'density'.
    assert forecasts_artifact.forecast_object == "density"


# ----------------------------------------------------------------------
# Test 4 -- F1 procedure-level: Eq. 10 reality-check active on public
#               path (mutation propagates through ``macroforecast.run``)
# ----------------------------------------------------------------------


def test_hnn_eq10_reality_check_active_on_public_path():
    """Phase B-9 F1 procedure-level: mutate a fitted HNN's
    ``_reality_check_intercept`` and confirm that the change propagates
    through the public ``_emit_quantile_intervals`` dispatch (i.e.
    ``predict_quantiles`` is the live path, not the LinearRegression
    fallback). Without F1 the LinearRegression sigma would mask the
    mutation entirely."""

    pytest.importorskip("torch")

    from macroforecast.core.runtime import (
        _emit_quantile_intervals,
        _HemisphereNN,
    )
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(7)
    T, K = 60, 3
    X = pd.DataFrame(
        rng.standard_normal((T, K)),
        columns=list("abc"),
    )
    y = pd.Series(rng.standard_normal(T))
    model = _HemisphereNN(
        B=2,
        n_epochs=8,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=21,
    ).fit(X, y)

    # Pin reality-check coefficients to baseline.
    model._reality_check_intercept = 0.0
    model._reality_check_slope = 1.0

    forecasts = {
        ("hnn", "y", 1, idx): float(model.predict(X.iloc[[i]])[0])
        for i, idx in enumerate(X.index)
    }
    fit_nodes = [
        {
            "op": "fit_model",
            "params": {
                "forecast_object": "quantile",
                "family": "mlp",
                "architecture": "hemisphere",
                "loss": "volatility_emphasis",
                "quantile_levels": [0.1, 0.5, 0.9],
            },
        }
    ]
    artifact = ModelArtifact(
        model_id="hnn",
        family="mlp",
        fitted_object=model,
        framework="torch",
        fit_metadata={},
        feature_names=tuple(X.columns),
    )
    artifacts = {"hnn": artifact}

    intervals_baseline = _emit_quantile_intervals(
        forecasts,
        fit_nodes,
        X=X,
        y=y,
        artifacts=artifacts,
    )

    # Multiplicative ×4 on the variance head: q-bands should widen.
    model._reality_check_intercept = float(np.log(4.0))
    intervals_shifted = _emit_quantile_intervals(
        forecasts,
        fit_nodes,
        X=X,
        y=y,
        artifacts=artifacts,
    )

    # Pick the first origin; compare q=0.9 - q=0.1 widths.
    sample_origin = next(iter(forecasts))[3]
    width_base = (
        intervals_baseline[("hnn", "y", 1, sample_origin, 0.9)]
        - intervals_baseline[("hnn", "y", 1, sample_origin, 0.1)]
    )
    width_shift = (
        intervals_shifted[("hnn", "y", 1, sample_origin, 0.9)]
        - intervals_shifted[("hnn", "y", 1, sample_origin, 0.1)]
    )
    # exp(intercept)=4 multiplied on variance => sqrt-scale ×2 on sigma.
    assert width_shift == pytest.approx(2.0 * width_base, rel=1e-5), (
        f"Eq. 10 mutation did not propagate via predict_quantiles: "
        f"baseline width={width_base}, shifted width={width_shift}"
    )


# ----------------------------------------------------------------------
# Test 5 -- F5 docstring regression (no longer "pre-promotion")
# ----------------------------------------------------------------------


def test_hnn_helper_docstring_no_longer_pre_promotion():
    """Phase B-9 F5: the helper docstring used to claim "Status:
    pre-promotion". After Phase-B9 fixes the paper distributional head
    is reachable from ``macroforecast.run``, so the docstring must drop
    the pre-promotion claim and the ``NotImplementedError`` warning."""

    doc = inspect.getdoc(hemisphere_neural_network) or ""
    assert "pre-promotion" not in doc.lower(), (
        f"helper docstring still claims pre-promotion: {doc!r}"
    )
    assert "notimplementederror" not in doc.lower()
    # Forward-looking claim: should mention operational status.
    assert "operational" in doc.lower(), (
        "helper docstring must document operational status post-Phase-B9"
    )
