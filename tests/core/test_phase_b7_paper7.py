"""Phase B-7 paper-7 (Goulet Coulombe & Klieber 2025 AlbaMA --
"An Adaptive Moving Average for Macroeconomic Monitoring",
arXiv:2501.13222) helper-rewrite tests.

Round 1 audit identified four findings on this paper:

* **F3 (HIGH)** -- ``paper_methods.adaptive_ma()`` piped ``src_X``
  (the predictor panel) into ``adaptive_ma_rf``. AlbaMA is
  ``RF(y_t ~ t)``; predictors are NOT inputs. The op was fitting
  time-trend RFs on each predictor column instead of smoothing the
  target.
* **F4 (LOW)** -- helper docstring + scaffold ``option_docs/l3.py``
  claimed "Status: pre-promotion" / "schema-only" / "runtime in
  v0.9.x"; the op IS operational since v0.9.
* **F5 (MEDIUM)** -- helper exposed only ``n_estimators`` (default
  ``100``); ``min_samples_leaf`` and ``sided`` were not in the
  signature, and ``n_estimators=100`` silently overrode the op-level
  paper-faithful default of ``500``.
* **F6 (LOW)** -- weights extraction (out of scope for B-7).

The five tests below close F3-F5. Reference: arXiv:2501.13222
§2.1-2.3 + p.7-8 + §3.3.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from macroforecast.recipes.paper_methods import adaptive_ma


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _alba_node(recipe: dict) -> dict:
    """Return the L3 ``adaptive_ma_rf`` step node from the recipe."""

    nodes = recipe["3_feature_engineering"]["nodes"]
    return next(n for n in nodes if n.get("op") == "adaptive_ma_rf")


# ----------------------------------------------------------------------
# Test 1 -- helper routes src_y (target) into adaptive_ma_rf, not src_X
# ----------------------------------------------------------------------


def test_albama_helper_routes_src_y_not_src_x():
    """Phase B-7 F3: AlbaMA is ``RF(y_t ~ t)`` (paper §2.1-2.3); the
    L3 ``adaptive_ma_rf`` step must consume the target source
    (``src_y``), not the predictor panel (``src_X``)."""

    recipe = adaptive_ma()
    alba = _alba_node(recipe)

    assert "src_y" in alba["inputs"], (
        f"adaptive_ma_rf inputs {alba['inputs']!r} must contain 'src_y' "
        "(target series) -- AlbaMA is RF(y_t ~ t)"
    )
    assert "src_X" not in alba["inputs"], (
        f"adaptive_ma_rf inputs {alba['inputs']!r} must NOT contain "
        "'src_X' -- predictors are not inputs to AlbaMA"
    )


# ----------------------------------------------------------------------
# Test 2 -- helper default n_estimators = 500 (paper p.8)
# ----------------------------------------------------------------------


def test_albama_helper_default_n_estimators_500():
    """Phase B-7 F5: helper default ``n_estimators`` = 500 (paper p.8
    line 351-352: "All forests use B = 500 trees"). The recipe's L3
    ``adaptive_ma_rf`` params dict must reflect this."""

    sig = inspect.signature(adaptive_ma)
    assert "n_estimators" in sig.parameters
    assert sig.parameters["n_estimators"].default == 500

    recipe = adaptive_ma()
    alba = _alba_node(recipe)
    assert alba["params"]["n_estimators"] == 500


# ----------------------------------------------------------------------
# Test 3 -- helper default min_samples_leaf = 40 (paper p.8)
# ----------------------------------------------------------------------


def test_albama_helper_default_min_samples_leaf_40():
    """Phase B-7 F5: helper default ``min_samples_leaf`` = 40 (paper
    p.8: rule-of-thumb lower bound on the realised window). The
    recipe's L3 ``adaptive_ma_rf`` params dict must reflect this."""

    sig = inspect.signature(adaptive_ma)
    assert "min_samples_leaf" in sig.parameters
    assert sig.parameters["min_samples_leaf"].default == 40

    recipe = adaptive_ma()
    alba = _alba_node(recipe)
    assert alba["params"]["min_samples_leaf"] == 40


# ----------------------------------------------------------------------
# Test 4 -- helper default sided = "two" (paper §2.2 / §3.3)
# ----------------------------------------------------------------------


def test_albama_helper_default_sided_is_two():
    """Phase B-7 F5: helper default ``sided`` = ``"two"`` (paper §2.2
    / §3.3: two-sided is the retrospective-smoother variant; the
    one-sided expanding-window variant is the real-time nowcasting
    mode). The recipe's L3 ``adaptive_ma_rf`` params dict must
    reflect this."""

    sig = inspect.signature(adaptive_ma)
    assert "sided" in sig.parameters
    assert sig.parameters["sided"].default == "two"

    recipe = adaptive_ma()
    alba = _alba_node(recipe)
    assert alba["params"]["sided"] == "two"


# ----------------------------------------------------------------------
# Test 5 -- e2e: AlbaMA smooths the target, not the predictor panel
# ----------------------------------------------------------------------


def test_albama_e2e_smooths_target_not_predictors():
    """Phase B-7 F3 procedure-level guard: build a T=120 panel where
    the target ``y`` lives on a small scale (smooth signal + noise,
    range ~[-1, 1]) and four unrelated predictor columns ``x1..x4``
    live on a large scale (range ~[100, 200]). Run the helper through
    ``_adaptive_ma_rf`` directly with the *target* series as input
    (mirroring the recipe wiring after the F3 fix), and assert the
    smoothed output's amplitude matches the y-scale (not the X-scale).
    A pre-fix call (passing the X panel) would have produced output
    in the [100, 200] band; the F3 fix forces smoothing of y in
    [-1, 1]."""

    from macroforecast.core.runtime import _adaptive_ma_rf

    rng = np.random.default_rng(0)
    T = 120
    t = np.arange(T)
    # Target: smooth low-amplitude signal + noise, range ~[-1, 1].
    y_signal = 0.5 * np.sin(2 * np.pi * t / 24.0)
    y = y_signal + 0.1 * rng.standard_normal(T)
    # Four unrelated predictors on a large scale, range ~[100, 200].
    X = 100.0 + 100.0 * rng.uniform(size=(T, 4))

    panel = {
        "date": pd.date_range("2010-01-01", periods=T, freq="MS")
        .strftime("%Y-%m-%d")
        .tolist(),
        "y": list(y),
    }
    for k in range(4):
        panel[f"x{k + 1}"] = list(X[:, k])

    # Build the recipe and confirm the L3 alba step indeed wires
    # src_y. The actual smoothing test below operates on a frame
    # mirroring the runtime call site.
    recipe = adaptive_ma(panel=panel, n_estimators=50, min_samples_leaf=15)
    alba = _alba_node(recipe)
    assert alba["inputs"] == ["src_y"]

    # Mirror the runtime call: smooth the y frame.
    y_frame = pd.DataFrame({"y": y})
    smoothed_y = _adaptive_ma_rf(
        y_frame,
        n_estimators=50,
        min_samples_leaf=15,
        sided="two",
        random_state=0,
    )
    smoothed_col = next(c for c in smoothed_y.columns)
    smoothed_vals = smoothed_y[smoothed_col].dropna().to_numpy()

    # Amplitude check: smoothed y must live in the y-scale, not the
    # predictor-scale. Pre-fix wiring (smoothing X panel) would push
    # output well above 50; post-fix smoothing of y stays near [-1, 1].
    assert np.max(np.abs(smoothed_vals)) < 5.0, (
        f"smoothed amplitude {np.max(np.abs(smoothed_vals)):.2f} too "
        "large -- suggests the smoother ran on the predictor panel "
        "(scale [100, 200]) rather than the target (scale [-1, 1])"
    )
    # Also confirm it actually tracks y (rather than a constant 0):
    # within the leaf-bound interior, output should not be all-zero.
    assert np.std(smoothed_vals) > 0.0, "smoothed output is constant"
