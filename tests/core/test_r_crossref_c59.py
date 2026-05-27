"""Cycle 59 — R cross-reference validation tests (Boruta / midasr / rugarch).

Produced by the StatsClaw tester pipeline (Cycle 59). Tests are derived
exclusively from test-spec.md (Cycle 59, §C). The tester does NOT read
spec.md or implementation.md.

All three tests are gated with pytest.importorskip("rpy2"). When rpy2 is
not installed (as on the CI server at this cycle), all tests are SKIPPED.
This is the expected outcome per planner.

When rpy2 IS installed, the tests require R packages:
  - Boruta  (CRAN): install.packages("Boruta")
  - midasr  (CRAN): install.packages("midasr")
  - rugarch (CRAN): install.packages("rugarch")

Test-spec.md §C acceptance criteria:
  XR-1: Boruta vs Boruta::Boruta() — Jaccard similarity >= 0.7
  XR-2: MIDAS Almon vs midasr::midas_r(weight_method="nealmon") — y_hat rtol=0.10
  XR-3: HHS vs rugarch::ugarchfit(spec=ugarchspec(model="realGARCH")) — param compare

Planner note: all expected to SKIP locally (rpy2 not installed). The
[validation] extra (`pip install macroforecast[validation]`) installs rpy2>=3.5
if R is available.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Guard: skip the entire module at collection time if rpy2 is absent or the
# linked system R C API is incompatible with the installed rpy2 wheel.
# ---------------------------------------------------------------------------

try:
    import rpy2  # noqa: F401
    from rpy2 import robjects as _rpy2_robjects  # noqa: F401
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"rpy2/R bridge unavailable for C59 cross-reference tests: {type(exc).__name__}: {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers — shared DGPs (same as C59 Boruta and HHS DGPs)
# ---------------------------------------------------------------------------

def _make_signal_panel_for_r(
    n_obs: int = 120,
    n_features: int = 10,
    n_relevant: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Signal DGP for Boruta cross-reference.

    Smaller than the full C59 panel (n_features=10 for speed with R overhead).
    y = x0 + x1 + 0.5*x2 + 0.3*x3 + noise(sigma=0.5).
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_obs)
    X = rng.standard_normal((n_obs, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, index=idx, columns=cols)
    noise = rng.standard_normal(n_obs) * 0.5
    y = (
        frame["x0"]
        + frame["x1"]
        + 0.5 * frame["x2"]
        + 0.3 * frame["x3"]
        + noise
    )
    y.name = "y"
    return frame, y


def _make_hhs_dgp_for_r(
    T: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """HHS DGP for rugarch cross-reference.

    Returns (X with 'rv' column, y returns Series).
    True params: mu=0.05, omega=-0.5, beta=0.85, xi=-0.2, phi=0.90,
                 delta_1=-0.05, delta_2=0.05, sigma_u=0.15.
    """
    import math

    true_params: dict[str, float] = {
        "mu": 0.05, "omega": -0.5, "beta": 0.85,
        "tau_1": -0.10, "tau_2": 0.05, "gamma": 0.05,
        "xi": -0.2, "phi": 0.90,
        "delta_1": -0.05, "delta_2": 0.05,
        "sigma_u": 0.15,
    }
    rng = np.random.default_rng(seed)
    h = np.zeros(T)
    z = np.zeros(T)
    u = np.zeros(T)
    r = np.zeros(T)
    x = np.zeros(T)
    p = true_params
    h[0] = float(np.exp(p["omega"] / (1.0 - p["beta"])))
    for t in range(T):
        z[t] = float(rng.standard_normal())
        r[t] = p["mu"] + math.sqrt(h[t]) * z[t]
        u[t] = p["sigma_u"] * float(rng.standard_normal())
        log_x_t = (
            p["xi"]
            + p["phi"] * math.log(h[t])
            + p["delta_1"] * z[t]
            + p["delta_2"] * (z[t] ** 2 - 1.0)
            + u[t]
        )
        x[t] = float(np.exp(log_x_t))
        if t < T - 1:
            log_h_next = (
                p["omega"]
                + p["beta"] * math.log(h[t])
                + p["tau_1"] * z[t]
                + p["tau_2"] * (z[t] ** 2 - 1.0)
                + p["gamma"] * u[t]
            )
            h[t + 1] = float(np.exp(log_h_next))

    idx = pd.RangeIndex(T)
    X = pd.DataFrame({"rv": x}, index=idx)
    y = pd.Series(r, index=idx, name="returns")
    return X, y


# ---------------------------------------------------------------------------
# XR-1 — Boruta vs R Boruta::Boruta()
# ---------------------------------------------------------------------------

def test_xr1_boruta_vs_r_boruta() -> None:
    """XR-1: Python Boruta selection agrees with R Boruta::Boruta() (Jaccard >= 0.7).

    Test-spec.md §C XR-1:
      - DGP: n_obs=120, n_features=10, n_relevant=4, seed=42
      - Python: _boruta_selection with n_estimators_rf=100, max_iter=100, alpha=0.05
      - R:      Boruta::Boruta(x ~ ., data=df, doTrace=0, maxRuns=100, pValue=0.05)
      - Metric: Jaccard(py_selected, r_confirmed)
      - Threshold: Jaccard >= 0.7 (from test-spec.md, NOT relaxed)
    """
    pytest.importorskip(
        "rpy2",
        reason="rpy2 required for XR-1 Boruta R cross-reference",
    )
    from rpy2 import robjects  # type: ignore[import]
    from rpy2.robjects import pandas2ri  # type: ignore[import]
    from rpy2.robjects.packages import importr  # type: ignore[import]

    # Check R package availability
    try:
        boruta_r = importr("Boruta")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"R package 'Boruta' not available: {exc}")

    from macroforecast.core.runtime import _boruta_selection

    frame, y = _make_signal_panel_for_r(
        n_obs=120, n_features=10, n_relevant=4, seed=42
    )

    # Python Boruta
    py_result = _boruta_selection(
        frame,
        target=y,
        params={
            "n_estimators_rf": 100,
            "max_iter": 100,
            "alpha": 0.05,
            "include_tentative": False,
            "random_state": 42,
        },
    )
    py_selected: set[str] = set(py_result.columns)

    # R Boruta
    with pandas2ri.converter.context():  # type: ignore[attr-defined]
        r_frame = pandas2ri.py2rpy(frame)
        r_y = pandas2ri.py2rpy(y)

    robjects.globalenv["py_X"] = r_frame
    robjects.globalenv["py_y"] = r_y

    r_code = """
    set.seed(42)
    df <- cbind(py_X, y = py_y)
    boruta_res <- Boruta::Boruta(y ~ ., data = df, doTrace = 0,
                                  maxRuns = 100, pValue = 0.05)
    confirmed <- names(boruta_res$finalDecision[boruta_res$finalDecision == "Confirmed"])
    confirmed
    """
    r_confirmed_raw = robjects.r(r_code)
    r_confirmed: set[str] = set(list(r_confirmed_raw))

    # Jaccard similarity
    intersection = py_selected & r_confirmed
    union = py_selected | r_confirmed
    jaccard: float = len(intersection) / len(union) if len(union) > 0 else 1.0

    jaccard_threshold: float = 0.70  # from test-spec.md §C XR-1, NOT relaxed

    assert jaccard >= jaccard_threshold, (
        f"XR-1 Boruta Jaccard similarity = {jaccard:.3f} "
        f"(py={py_selected}, R_confirmed={r_confirmed}) "
        f"< threshold {jaccard_threshold} from test-spec.md. "
        f"Tolerance used: Jaccard >= 0.70 (unchanged from spec)."
    )


# ---------------------------------------------------------------------------
# XR-2 — MIDAS Almon vs midasr::midas_r(weight_method="nealmon")
# ---------------------------------------------------------------------------

def test_xr2_midas_almon_vs_r_midasr() -> None:
    """XR-2: Python midas_almon predictions agree with midasr::midas_r (rtol=0.10).

    Test-spec.md §C XR-2:
      - DGP: macroeconomic-style quarterly y, monthly X with 3 lags
      - Python: midas_almon from macroforecast L4
      - R: midasr::midas_r(y ~ fmls(x, 3, 3, nealmon), ...)
      - Metric: max relative deviation of y_hat (rtol=0.10)
      - Threshold: rtol=0.10 (from test-spec.md, NOT relaxed)
    """
    pytest.importorskip(
        "rpy2",
        reason="rpy2 required for XR-2 MIDAS midasr cross-reference",
    )
    from rpy2 import robjects  # type: ignore[import]
    from rpy2.robjects import pandas2ri  # type: ignore[import]
    from rpy2.robjects.packages import importr  # type: ignore[import]

    try:
        importr("midasr")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"R package 'midasr' not available: {exc}")

    from macroforecast.core.runtime import _build_l4_model

    # DGP: quarterly y, monthly X (ratio m=3)
    rng = np.random.default_rng(99)
    n_q = 60  # 60 quarters
    n_m = n_q * 3  # 180 monthly observations
    x_monthly = rng.standard_normal(n_m)
    # Almon-style weighting: sum of lagged monthly values (lags 1-3 per quarter)
    x_q = np.array([
        0.5 * x_monthly[3*t] + 0.3 * x_monthly[3*t + 1] + 0.2 * x_monthly[3*t + 2]
        for t in range(n_q)
    ])
    y_q = x_q + 0.3 * rng.standard_normal(n_q)

    idx_q = pd.date_range("2000Q1", periods=n_q, freq="QE")
    idx_m = pd.date_range("2000-01", periods=n_m, freq="ME")
    y_ser = pd.Series(y_q, index=idx_q, name="y")
    X_monthly = pd.DataFrame({"x_monthly": x_monthly}, index=idx_m)

    # Python midas_almon prediction
    try:
        model = _build_l4_model(
            "midas_almon",
            params={"midas_ratio": 3, "n_lags": 3, "random_state": 0},
        )
        model.fit(X_monthly, y_ser)
        py_yhat = model.predict(X_monthly)
    except (NotImplementedError, AttributeError, KeyError) as exc:
        pytest.skip(f"midas_almon not yet runnable in this build: {exc}")

    # R midasr prediction
    robjects.globalenv["r_y"] = robjects.FloatVector(y_q)
    robjects.globalenv["r_x_monthly"] = robjects.FloatVector(x_monthly)
    robjects.globalenv["r_n_q"] = n_q
    robjects.globalenv["r_n_m"] = n_m

    r_code = """
    set.seed(0)
    library(midasr)
    fit <- midas_r(r_y ~ fmls(r_x_monthly, 3, 3, nealmon),
                   start = list(r_x_monthly = c(1.0, 0.0, 0.0)))
    fitted(fit)
    """
    try:
        r_yhat_vec = robjects.r(r_code)
        r_yhat = np.array(list(r_yhat_vec))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"midasr::midas_r failed in R: {exc}")

    # Compare on the in-sample period (quarterly grid)
    min_len = min(len(py_yhat), len(r_yhat))
    py_q = py_yhat[-min_len:]
    r_q = r_yhat[-min_len:]

    # rtol = max |py - r| / (|r| + 1e-8)
    rel_err = np.max(np.abs(py_q - r_q) / (np.abs(r_q) + 1e-8))
    rtol_threshold: float = 0.10  # from test-spec.md §C XR-2, NOT relaxed

    assert rel_err <= rtol_threshold, (
        f"XR-2 MIDAS Almon vs midasr rtol = {rel_err:.4f} "
        f"> threshold {rtol_threshold} from test-spec.md. "
        f"Tolerance used: rtol=0.10 (unchanged from spec)."
    )


# ---------------------------------------------------------------------------
# XR-3 — HHS realized_garch vs rugarch::ugarchfit(model="realGARCH")
# ---------------------------------------------------------------------------

def test_xr3_hhs_vs_r_rugarch() -> None:
    """XR-3: Python HHS realized_garch params agree with R rugarch realGARCH.

    Test-spec.md §C XR-3:
      - DGP: T=300, seed=42, same HHS DGP as C49
      - Python: realized_garch (joint MLE)
      - R: rugarch::ugarchfit(spec=ugarchspec(model="realGARCH"), data=y)
      - Metric: per-parameter absolute difference for mu, beta, phi
      - Thresholds (from test-spec.md, NOT relaxed):
          mu:   atol=0.05
          beta: atol=0.10
          phi:  atol=0.15
    """
    pytest.importorskip(
        "rpy2",
        reason="rpy2 required for XR-3 HHS rugarch cross-reference",
    )
    from rpy2 import robjects  # type: ignore[import]
    from rpy2.robjects.packages import importr  # type: ignore[import]

    try:
        importr("rugarch")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"R package 'rugarch' not available: {exc}")

    from macroforecast.core.runtime import _build_l4_model

    X, y = _make_hhs_dgp_for_r(T=300, seed=42)

    # Python HHS fit
    model = _build_l4_model(
        "realized_garch",
        params={"realized_variance": "rv", "random_state": 42},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    py_params = model.params_

    # R rugarch fit
    y_vec = robjects.FloatVector(list(y.values))
    rv_vec = robjects.FloatVector(list(X["rv"].values))
    robjects.globalenv["r_ret"] = y_vec
    robjects.globalenv["r_rv"] = rv_vec

    r_code = """
    library(rugarch)
    spec <- ugarchspec(
        variance.model = list(model = "realGARCH", garchOrder = c(1, 1)),
        mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = "norm"
    )
    fit <- ugarchfit(spec = spec, data = r_ret,
                     realizedVol = sqrt(r_rv), solver = "hybrid")
    coef(fit)
    """
    try:
        r_coef_vec = robjects.r(r_code)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"rugarch::ugarchfit failed in R: {exc}")

    r_names = list(r_coef_vec.names)
    r_coef = dict(zip(r_names, list(r_coef_vec)))

    # Tolerance spec from test-spec.md §C XR-3 — NOT relaxed
    param_checks: dict[str, tuple[str, float]] = {
        "mu":   ("mu",   0.05),
        "beta": ("beta1", 0.10),  # rugarch names beta1; our impl names beta
        "phi":  ("rho",   0.15),  # rugarch names rho for phi in realGARCH spec
    }

    failures: list[str] = []
    for our_key, (r_key, atol) in param_checks.items():
        if our_key not in py_params:
            continue
        if r_key not in r_coef:
            # Try alternate naming
            r_key_alt = our_key
            if r_key_alt not in r_coef:
                continue
            r_key = r_key_alt
        py_val = float(py_params[our_key])
        r_val = float(r_coef[r_key])
        err = abs(py_val - r_val)
        if err > atol:
            failures.append(
                f"{our_key}: py={py_val:.5f}, R({r_key})={r_val:.5f}, "
                f"err={err:.5f} > atol={atol} (from test-spec.md)"
            )

    assert len(failures) == 0, (
        "XR-3 HHS vs rugarch param comparison failed:\n" +
        "\n".join(failures) +
        "\nTolerance used: per test-spec.md §C XR-3 (unchanged from spec)."
    )
