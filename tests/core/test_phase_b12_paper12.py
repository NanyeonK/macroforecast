"""Phase B-12 paper-12 (Goulet Coulombe / Goebel / Klieber 2024 -- "A
Dual Interpretation of Machine Learning Forecasts") helper-rewrite +
KRR-branch tests.

Round 1 audit found three gaps that left the paper's central
interpretive object — the per-test-row ``(n_test, n_train)`` dual
weight matrix and §2.8 portfolio diagnostics (FC / FSP / FL / FT) —
either unreachable from the helper or unsupported for the paper's
headline estimator (Kernel Ridge Regression):

* **F1/F2 (HIGH)** -- ``dual_interpretation`` returned an L4 ridge
  baseline only. NO ``7_interpretation`` block. The
  ``dual_decomposition`` op (operational since v0.8.9) NEVER fired
  when the helper drove ``macroforecast.run``; users got
  ``forecasts`` but no ``dual_weights`` matrix and no
  ``portfolio_metrics`` frame.

* **F4 (HIGH)** -- the ``coef_`` branch in
  ``_dual_decomposition_frame`` does not match sklearn ``KernelRidge``
  (which carries ``dual_coef_`` / ``X_fit_`` and no ``coef_``), so
  KRR fell through to the unsupported-family branch and raised
  ``NotImplementedError``. KRR is the paper's §2.2 headline
  application (Fig 2 / Table 1) and the canonical setting in which
  the representer-theorem form ``w_j = K_j (K + αI)⁻¹`` (paper Eqs.
  5-6) is exact.

The five tests below close F1/F2/F4. Reference: Goulet Coulombe /
Goebel / Klieber (2024) "A Dual Interpretation of Machine Learning
Forecasts", arXiv:2412.13076; §2 Eqs. 5-6 (representer identity),
§2.2 (KRR headline application), §2.8 (portfolio metrics).
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd

import macroforecast
from macroforecast.recipes.paper_methods import dual_interpretation


# ----------------------------------------------------------------------
# DGP fixture: T=80, K=4 linear-Gaussian panel
# ----------------------------------------------------------------------


def _build_dual_panel(T: int = 80, K: int = 4, seed: int = 0):
    """Linear-Gaussian DGP with K predictors so the representer-theorem
    identity holds at machine precision (linear closed-form). Same
    fixture style as Phase B-10 / B-11 sibling tests."""

    rng = np.random.default_rng(seed)
    dates: list[str] = []
    d = datetime.date(2014, 1, 1)
    for _ in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y_yr = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y_yr, m, 1)

    X = rng.normal(0.0, 1.0, size=(T, K))
    beta = rng.normal(0.0, 1.0, size=K)
    noise = rng.normal(0.0, 0.1, size=T)
    y = X @ beta + noise

    panel: dict[str, list[object]] = {"date": dates, "y": y.tolist()}
    for j in range(K):
        panel[f"x{j + 1}"] = X[:, j].tolist()
    return panel


def _locate_dual_frame(global_importance: dict) -> pd.DataFrame | None:
    """Pick the ``dual_decomposition`` frame from an L7 importance
    sink. The ``method`` attribute on the frame is one of
    ``'linear_closed_form'`` / ``'rf_leaf_cooccurrence_kernel'`` /
    ``'krr_representer'`` (etc.); the canonical L7 sink-key tuple
    carries the op name (``'dual_decomposition'``) at position 3."""

    for key, frame in global_importance.items():
        op_name = key[3] if isinstance(key, tuple) and len(key) >= 4 else None
        if op_name == "dual_decomposition":
            return frame
        if isinstance(frame, pd.DataFrame):
            method = frame.attrs.get("method")
            if method in {
                "linear_closed_form",
                "rf_leaf_cooccurrence_kernel",
                "krr_representer",
            }:
                return frame
    return None


# ----------------------------------------------------------------------
# Test 1 -- F1/F2 (recipe shape): helper attaches a 7_interpretation
# block whose step op is ``dual_decomposition``
# ----------------------------------------------------------------------


def test_dual_interpretation_helper_attaches_l7_block():
    """Phase B-12 F1/F2 fix: ``dual_interpretation`` must append a
    ``7_interpretation`` block whose interpretation step is
    ``dual_decomposition``. Previously the helper returned an L4
    ridge baseline only, so ``macroforecast.run`` never reached the
    paper's central object (the dual-weight matrix + portfolio
    metrics)."""

    recipe = dual_interpretation()

    assert "7_interpretation" in recipe, (
        "helper must attach a 7_interpretation block by default"
    )
    block = recipe["7_interpretation"]
    assert block.get("enabled") is True

    nodes = block.get("nodes", [])
    step = next((n for n in nodes if n.get("type") == "step"), None)
    assert step is not None, "L7 block must contain a step node"
    assert step["op"] == "dual_decomposition", (
        f"helper step op must be dual_decomposition, got {step['op']!r}"
    )

    # The step must be wired to the L4 model artifact (so the runtime
    # picks up the fitted ridge) and the L3 X_final feature panel.
    input_ids = set(step.get("inputs", []))
    assert "src_model" in input_ids
    assert "src_X" in input_ids
    # L7 sink must publish the dual frame on l7_importance_v1.
    sinks = block.get("sinks", {})
    assert sinks.get("l7_importance_v1", {}).get("global") == "dual_explain"


# ----------------------------------------------------------------------
# Test 2 -- F1/F2 procedure-level closure: end-to-end via
# macroforecast.run, dual_weights matrix + portfolio_metrics frame
# reach the public L7 sink
# ----------------------------------------------------------------------


def test_dual_interpretation_e2e_emits_dual_weights():
    """Phase B-12 F1/F2 procedure-level closure: build a small T=80,
    K=4 DGP, call ``dual_interpretation()``, run via
    ``macroforecast.run``, and confirm the L7 sink carries
    ``frame.attrs['dual_weights']`` (non-empty matrix) and
    ``frame.attrs['portfolio_metrics']`` (non-empty frame)."""

    panel = _build_dual_panel(T=80, K=4, seed=0)
    recipe = dual_interpretation(target="y", horizon=1, panel=panel)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    assert result.cells, "dual_interpretation must produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l7_importance_v1" in artifacts, (
        "F1/F2 fix: helper must wire an L7 sink so the dual frame "
        f"reaches macroforecast.run output (got {list(artifacts)})"
    )
    importance = artifacts["l7_importance_v1"]
    global_importance = getattr(importance, "global_importance", {}) or {}
    assert global_importance, (
        "L7 global_importance must contain the dual_decomposition frame"
    )

    dual_frame = _locate_dual_frame(global_importance)
    assert dual_frame is not None, (
        f"dual_decomposition frame not found in L7 sink "
        f"(keys={list(global_importance)})"
    )

    weights = dual_frame.attrs.get("dual_weights")
    assert isinstance(weights, pd.DataFrame), (
        "frame.attrs['dual_weights'] must be a DataFrame"
    )
    assert not weights.empty, "dual_weights matrix must be non-empty"
    assert weights.shape[0] > 0 and weights.shape[1] > 0

    portfolio = dual_frame.attrs.get("portfolio_metrics")
    assert isinstance(portfolio, pd.DataFrame), (
        "frame.attrs['portfolio_metrics'] must be a DataFrame"
    )
    assert not portfolio.empty, "portfolio_metrics frame must be non-empty"


# ----------------------------------------------------------------------
# Test 3 -- F4 procedure-level: KernelRidge representer identity
# (paper Eqs. 5-6: max |W @ y_train - ŷ_test| < 1e-6)
# ----------------------------------------------------------------------


def test_dual_decomposition_krr_representer_identity():
    """Phase B-12 F4 procedure-level closure: fit
    ``KernelRidge(kernel='rbf', alpha=1.0)`` on a small DGP, invoke
    ``_dual_decomposition_frame``, and confirm the recovered ``W``
    matrix satisfies the representer identity ``W @ y_train ≈ ŷ_test``
    to tolerance 1e-6 (paper Eqs. 5-6).

    Pre-fix this branch raised ``NotImplementedError`` because
    ``KernelRidge`` has ``dual_coef_`` instead of ``coef_`` and the
    dispatcher's ``hasattr(fitted, 'coef_')`` short-circuit therefore
    fell through to the unsupported-family fallback."""

    from sklearn.kernel_ridge import KernelRidge
    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    n, p = 30, 4
    X_arr = rng.normal(size=(n, p))
    # Non-linear DGP so the rbf-kernel fit isn't trivially equivalent
    # to a linear baseline — the representer identity is what we're
    # testing, not the closed-form linear path.
    y_arr = X_arr[:, 0] ** 2 + np.sin(X_arr[:, 1]) + 0.1 * rng.normal(size=n)
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(p)])
    y = pd.Series(y_arr)

    fitted = KernelRidge(kernel="rbf", alpha=1.0).fit(X.to_numpy(), y.to_numpy())
    artifact = ModelArtifact(
        model_id="m_krr",
        family="kernel_ridge",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )

    frame = _dual_decomposition_frame(artifact, X, y)
    weights_full = frame.attrs["dual_weights"]
    assert isinstance(weights_full, pd.DataFrame)

    W = weights_full.to_numpy()
    yhat_krr = fitted.predict(X.to_numpy())
    yhat_dual = W @ y.to_numpy()

    max_dev = float(np.max(np.abs(yhat_dual - yhat_krr)))
    assert max_dev < 1e-6, (
        f"KRR representer identity (paper Eqs. 5-6) violated: "
        f"max|W @ y_train - ŷ_test| = {max_dev}"
    )

    # Sanity: method label tracks the new branch.
    assert frame.attrs.get("method") == "krr_representer", (
        f"KRR fit must dispatch to the krr_representer branch, "
        f"got method={frame.attrs.get('method')!r}"
    )


# ----------------------------------------------------------------------
# Test 4 -- F4: KRR no longer raises NotImplementedError
# ----------------------------------------------------------------------


def test_dual_decomposition_krr_branch_does_not_raise_unsupported():
    """Phase B-12 F4: ``_dual_decomposition_frame`` called with a
    fitted ``KernelRidge`` artifact must produce a frame, NOT raise
    ``NotImplementedError`` as it did pre-fix. The frame must carry a
    non-empty ``dual_weights`` matrix and a ``portfolio_metrics``
    frame with the standard (signed) FC/FSP/FL/FT columns."""

    from sklearn.kernel_ridge import KernelRidge
    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(1)
    n, p = 25, 3
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(X.sum(axis=1).to_numpy() + 0.1 * rng.normal(size=n))
    fitted = KernelRidge(kernel="rbf", alpha=0.5).fit(X.to_numpy(), y.to_numpy())
    artifact = ModelArtifact(
        model_id="m_krr2",
        family="kernel_ridge",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )

    # Pre-fix: this raised NotImplementedError. Post-fix: produces a frame.
    frame = _dual_decomposition_frame(artifact, X, y)
    assert isinstance(frame, pd.DataFrame)

    weights = frame.attrs["dual_weights"]
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == (n, n), (
        f"KRR W must be (n_test, n_train) = ({n}, {n}); got {weights.shape}"
    )

    portfolio = frame.attrs["portfolio_metrics"]
    assert isinstance(portfolio, pd.DataFrame)
    # paper §2.8 columns plus the legacy magnitudes carried for
    # backward-compatible plotting.
    assert {"hhi", "short", "turnover", "leverage"}.issubset(set(portfolio.columns))


# ----------------------------------------------------------------------
# Test 5 -- procedure-level: helper end-to-end emits paper §2.8
# portfolio metrics (FC, FSP, FL, FT) on the L7 sink
# ----------------------------------------------------------------------


def test_dual_interpretation_e2e_returns_paper_metrics():
    """Phase B-12 procedure-level: when the helper drives
    ``macroforecast.run``, the L7 sink's ``portfolio_metrics`` frame
    must contain the four paper §2.8 columns (FC = forecast
    concentration ``hhi``, FSP = forecast short position ``short``,
    FL = forecast leverage ``leverage``, FT = forecast turnover
    ``turnover``). The signed conventions (FSP ≤ 0, FL signed sum)
    follow the v0.9.0F audit-fix; absolute-value variants
    (``leverage_l1``, ``short_abs``) remain available as legacy
    plotting magnitudes but are not required by this acceptance
    test."""

    panel = _build_dual_panel(T=60, K=4, seed=42)
    recipe = dual_interpretation(target="y", horizon=1, panel=panel)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    artifacts = result.cells[0].runtime_result.artifacts
    importance = artifacts["l7_importance_v1"]
    global_importance = getattr(importance, "global_importance", {}) or {}
    dual_frame = _locate_dual_frame(global_importance)
    assert dual_frame is not None

    portfolio = dual_frame.attrs.get("portfolio_metrics")
    assert isinstance(portfolio, pd.DataFrame)
    # paper §2.8: four canonical columns, all present.
    required = {"hhi", "short", "turnover", "leverage"}
    missing = required.difference(set(portfolio.columns))
    assert not missing, (
        f"portfolio_metrics frame missing paper §2.8 columns "
        f"{sorted(missing)}; got {sorted(portfolio.columns)}"
    )
    # Frame must have one row per test observation.
    assert len(portfolio) > 0
    # FC = HHI ≥ 0 invariant (squared weights).
    assert (portfolio["hhi"] >= 0).all()
    # First-row turnover is 0 by construction (no t-1).
    assert float(portfolio["turnover"].iloc[0]) == 0.0
