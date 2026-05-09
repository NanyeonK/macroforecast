"""Phase B-5 paper-5 (Goulet Coulombe 2024 JAE
"The Macroeconomy as a Random Forest") helper-expansion guards.

Round 1 audit identified two in-scope findings for paper 5:

* **F5 (LOW helper-vs-core seam)** -- ``macroeconomic_random_forest()``
  helper exposed only ``n_estimators`` and ``block_size``. Paper-relevant
  hyperparameters (``ridge_lambda`` λ=0.1 paper p.9, ``rw_regul`` ζ=0.75
  paper p.10, ``mtry_frac`` 1/3 paper p.7, ``subsampling_rate`` 0.75
  paper p.10, ``quantile_rate``, ``trend_push``) were reachable only by
  hand-editing the ``fit_params`` dict, not via the public helper API.

* **Procedure-test gap** -- existing tests cover bit-exact equivalence to
  the vendored MRF, but no DGP-recovery test verifies the per-leaf
  GTVP coefficient path on a known time-varying DGP.

Three tests below close those gaps. Reference: arXiv:2006.12724.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast.core.runtime import _MRFExternalWrapper
from macroforecast.recipes.paper_methods import macroeconomic_random_forest


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _fit_node_params(recipe: dict) -> dict:
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    return fit["params"]


# ----------------------------------------------------------------------
# Test 1 -- helper signature exposes the paper hyperparameters
# ----------------------------------------------------------------------


def test_mrf_helper_exposes_paper_hyperparameters():
    """Phase B-5 paper-5 F5: ``macroeconomic_random_forest()`` must
    expose the paper-relevant hyperparameters as first-class keyword
    arguments with paper-spec defaults, and forward them into the L4
    ``fit_model`` ``params`` dict.

    Defaults verified:

    * ``ridge_lambda = 0.1`` (paper p.9)
    * ``rw_regul = 0.75`` (paper p.10)
    * ``mtry_frac = 1/3`` (paper p.7)
    * ``subsampling_rate = 0.75`` (paper p.10)
    * ``quantile_rate = 0.3`` (vendored default; paper §3.2)
    * ``trend_push = 1.0`` (vendored default; paper §3.2)
    """

    sig = inspect.signature(macroeconomic_random_forest)
    expected_defaults = {
        "ridge_lambda": 0.1,
        "rw_regul": 0.75,
        "mtry_frac": 1.0 / 3.0,
        "subsampling_rate": 0.75,
        "quantile_rate": 0.3,
        "trend_push": 1.0,
    }
    for name, expected in expected_defaults.items():
        assert name in sig.parameters, f"helper missing kwarg {name!r}"
        param = sig.parameters[name]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name!r} must be keyword-only (matches existing helper signature)"
        )
        assert param.default == pytest.approx(expected), (
            f"{name!r} default {param.default!r} != paper spec {expected!r}"
        )

    # Recipe params reflect the helper defaults.
    recipe = macroeconomic_random_forest()
    params = _fit_node_params(recipe)
    for name, expected in expected_defaults.items():
        assert name in params, f"fit_params missing forwarded {name!r}"
        assert params[name] == pytest.approx(expected), (
            f"recipe params[{name!r}] {params[name]!r} != paper default {expected!r}"
        )


# ----------------------------------------------------------------------
# Test 2 -- helper kwargs plumb through to the vendored MRF
# ----------------------------------------------------------------------


def test_mrf_helper_args_plumb_to_vendored_wrapper():
    """Phase B-5 paper-5 F5: helper kwargs must propagate end-to-end
    into the ``_MRFExternalWrapper`` instance that the runtime hands to
    the vendored ``MacroRandomForest`` constructor.

    Procedure: invoke the helper with non-default values across all
    six new kwargs, run via ``macroforecast.run`` on a small synthetic
    panel, and inspect the fitted ``_MRFExternalWrapper`` to confirm
    every value reached the wrapper attributes (and therefore the
    ``MacroRandomForest`` constructor invoked inside ``predict``).
    """

    rng = np.random.default_rng(0)
    t = 120
    dates = (
        pd.date_range("2010-01-01", periods=t, freq="MS").strftime("%Y-%m-%d").tolist()
    )
    y = np.zeros(t)
    y[0] = 0.0
    for tt in range(1, t):
        y[tt] = 0.5 * y[tt - 1] + rng.standard_normal()
    panel = {
        "date": dates,
        "y": list(y),
        "x1": list(rng.standard_normal(t)),
        "x2": list(rng.standard_normal(t)),
        "x3": list(rng.standard_normal(t)),
    }

    overrides = dict(
        ridge_lambda=0.5,
        rw_regul=0.25,
        mtry_frac=0.5,
        subsampling_rate=0.5,
        quantile_rate=0.4,
        trend_push=2.0,
    )
    # Tiny ensemble + modest panel keeps the fit fast (<5s) while giving
    # the bootstrap enough rows to populate every leaf. Match the
    # ``test_paper_05_macroeconomic_random_forest`` smoke test convention
    # (``tests/core/test_paper_helpers_e2e.py``): force ``min_train_size``
    # high enough that early origins skip the fit and only the
    # late-origin walk-forward range exercises the bootstrap.
    recipe = macroeconomic_random_forest(
        target="y",
        horizon=1,
        n_estimators=4,
        block_size=12,
        panel=panel,
        seed=0,
        **overrides,
    )
    for node in recipe["4_forecasting_model"]["nodes"]:
        if node.get("op") == "fit_model":
            node["params"]["min_train_size"] = 80
            node["params"]["B"] = 2

    # Recipe params carry the overrides verbatim.
    params = _fit_node_params(recipe)
    for name, expected in overrides.items():
        assert params[name] == pytest.approx(expected), (
            f"recipe override {name!r} {params[name]!r} != requested {expected!r}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    cell = result.cells[0]
    model_artifact = cell.runtime_result.artifacts["l4_model_artifacts_v1"]
    fitted = next(iter(model_artifact.artifacts.values())).fitted_object
    assert isinstance(fitted, _MRFExternalWrapper), (
        f"Expected fitted_object to be _MRFExternalWrapper, got {type(fitted).__name__}"
    )
    for name, expected in overrides.items():
        actual = getattr(fitted, name)
        assert actual == pytest.approx(expected), (
            f"_MRFExternalWrapper.{name} = {actual!r} != requested {expected!r}"
        )


# ----------------------------------------------------------------------
# Test 3 -- DGP recovery on a smooth time-varying-coefficient path
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_mrf_recovers_smooth_tvp_dgp():
    """Phase B-5 paper-5 procedure test: MRF must recover a known
    smooth time-varying-coefficient (TVP) path on a synthetic DGP.

    DGP: ``y[t] = β[t] * x[t] + ε[t]`` with ``β[t] = sin(t / T * π)``
    (smooth, 0 → 1 → 0), ``x[t] ~ N(0, 1)``, ``ε[t] ~ N(0, 0.5)``.
    Paper §3.1 uses a SETAR DGP morphing into AR(2); a smooth sine
    path is a strictly easier recovery test (no regime jumps) and
    therefore an apt smoke check that the per-leaf local linear
    regression machinery produces sensible per-period coefficients.

    The vendored MRF caches the GTVP β path on
    ``_MRFExternalWrapper._cached_betas`` after each predict call.
    Asserts ``corr(β̂[t, x_col], β_true[t]) >= 0.3`` over the training
    period (generous threshold to absorb finite-sample noise from a
    small ensemble; paper Fig 3 reports much tighter recovery on
    larger ensembles).
    """

    rng = np.random.default_rng(0)
    t = 200
    # x1 is the active TVP regressor; x2..x3 are noise distractors;
    # ``trend`` is a normalized time index that the vendored MRF can use
    # as a splitting feature to discover the smooth time-varying β path
    # (paper §3.2 explicitly relies on a trend-axis split column,
    # surfaced via ``trend_push``).
    k = 4

    # True smooth TVP path on a 0..π half-cycle, 0 → 1 → 0.
    grid = np.linspace(0.0, np.pi, t)
    beta_true = np.sin(grid)

    x1 = rng.standard_normal(t)
    x2 = rng.standard_normal(t)
    x3 = rng.standard_normal(t)
    trend = np.linspace(0.0, 1.0, t)
    eps = 0.5 * rng.standard_normal(t)
    y = beta_true * x1 + eps

    X = pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "trend": trend},
    )
    assert X.shape[1] == k
    y_ser = pd.Series(y, name="y")

    # Small ensemble + small block keeps runtime <10s on a laptop. Use
    # the wrapper directly (not via macroforecast.run) so the test
    # hits exactly the procedure under audit and we can read
    # ``_cached_betas`` straight after predict.
    n_train = t - 4
    train_X, test_X = X.iloc[:n_train], X.iloc[n_train:]
    train_y = y_ser.iloc[:n_train]

    wrapper = _MRFExternalWrapper(
        B=20,
        ridge_lambda=0.1,
        rw_regul=0.75,
        mtry_frac=1.0 / 3.0,
        subsampling_rate=0.75,
        quantile_rate=0.3,
        trend_push=4.0,
        fast_rw=True,
        resampling_opt=2,
        parallelise=False,
        n_cores=1,
        block_size=24,
        random_state=0,
    )
    wrapper.fit(train_X, train_y)
    np.random.seed(0)
    _ = wrapper.predict(test_X)

    betas = wrapper._cached_betas
    assert betas is not None, "wrapper did not cache GTVP betas after predict"
    # Vendored MRF returns betas with shape (T, K+1): column 0 is the
    # intercept and columns 1..K correspond to the feature order passed
    # to ``MacroRandomForest`` (x1 first → column 1 here).
    assert betas.shape == (t, k + 1), (
        f"unexpected betas shape {betas.shape}, expected {(t, k + 1)}"
    )
    beta_hat = betas[:n_train, 1]
    beta_truth = beta_true[:n_train]

    # Replace any NaN (vendored MRF can emit NaN at the very edges of
    # leaves with too few obs) with the local mean so the correlation
    # is well-defined; if the DGP recovery is real the bulk of the
    # path is finite and informative.
    finite_mask = np.isfinite(beta_hat)
    assert finite_mask.sum() >= int(0.8 * n_train), (
        f"only {finite_mask.sum()}/{n_train} betas are finite -- "
        "MRF caching path may be broken"
    )
    beta_hat_clean = beta_hat[finite_mask]
    beta_truth_clean = beta_truth[finite_mask]

    corr = float(np.corrcoef(beta_hat_clean, beta_truth_clean)[0, 1])
    # Generous lower bound; paper Fig 3 shows much tighter recovery on
    # larger ensembles. The 0.3 floor catches obvious regressions
    # (sign flip, constant-coefficient collapse, unrelated noise) while
    # absorbing the high finite-sample variance of an 8-tree ensemble.
    assert corr >= 0.3, (
        f"MRF GTVP recovery correlation {corr:.3f} < 0.3 -- "
        "TVP path estimation may be broken"
    )
