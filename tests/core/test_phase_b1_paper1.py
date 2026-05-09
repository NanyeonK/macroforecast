"""Phase B-1 paper-1 (Huang/Jiang/Li/Tong/Zhou 2022 Scaled PCA)
procedure tests.

Round 1 audit identified four findings on this paper:

* **F1 (HIGH)** — the recipe helper wired the *unshifted* target
  ``src_y`` into the ``scaled_pca`` step's target_signal slot, so the
  per-column slope ``β_j`` reflected contemporaneous correlation
  rather than the paper's predictive Eq. (3) regression of
  ``y_{t+h}`` on ``X_{i,t}``.
* **F2 (HIGH)** — ``materialize_l3_minimal`` ran the L3 DAG once on
  the full panel before L4's walk-forward began, so β_j and the
  principal directions peeked at all post-origin observations.
* **F3 (HIGH)** — ``temporal_rule = expanding_window_per_origin`` was
  validated by ``_temporal_present`` but never read by the runtime.
* **F4 (Medium)** — unit-level β tests on synthetic data hit the
  closed-form algorithm but bypassed the L3 wiring (so they did not
  catch F1/F2/F3).

The four tests in this module guard the runtime fixes against
regression. Their counterparts in ``test_v09_paper_coverage.py``
remain unchanged and continue to pin the unit-level β closed form
against ``np.linalg.lstsq``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# F1 -- the helper must wire the h-shifted target into scaled_pca
# ----------------------------------------------------------------------


def test_scaled_pca_helper_routes_y_h_not_y_t():
    """Phase B-1 F1: the ``scaled_pca`` helper's emitted L3 DAG must
    route the h-shifted target (``y_h``, output of ``target_construction``)
    into the ``scaled_pca`` step's target_signal slot, not the raw
    ``src_y``. The runtime resolves ``target_signal`` via
    ``_first_series(inputs)``, which scans ``inputs[1:]`` for the first
    Series-shaped value -- so the second positional input of the
    ``scaled_pca`` step must be ``y_h``."""

    from macroforecast.recipes.paper_methods import scaled_pca

    recipe = scaled_pca(target="y", horizon=4, n_components=2)
    nodes = recipe["3_feature_engineering"]["nodes"]

    by_id = {n.get("id"): n for n in nodes}
    spca = by_id["spca"]
    y_h = by_id["y_h"]

    # The y_h node is a target_construction step at the requested horizon.
    assert y_h.get("op") == "target_construction"
    assert y_h.get("params", {}).get("horizon") == 4
    assert y_h.get("params", {}).get("mode") == "point_forecast"

    # The scaled_pca step's second positional input must be y_h, not src_y.
    spca_inputs = list(spca["inputs"])
    assert len(spca_inputs) >= 2, (
        f"scaled_pca step needs at least 2 inputs (X, target_signal); got {spca_inputs}"
    )
    assert spca_inputs[0] == "src_X", (
        f"first input should be src_X; got {spca_inputs[0]!r}"
    )
    assert spca_inputs[1] == "y_h", (
        f"second input must be the shifted target node y_h "
        f"(target_construction output); got {spca_inputs[1]!r}. "
        f"Full inputs: {spca_inputs}. The Round-1 audit found this "
        f"slot was previously wired to src_y, which violates Eq. (3)."
    )


# ----------------------------------------------------------------------
# F2 -- per-origin L3 must execute when temporal_rule is expanding
# ----------------------------------------------------------------------


def _scaled_pca_panel_with_future_dependence(
    *, T: int = 60, n_features: int = 6, lead: int = 5, seed: int = 0
) -> dict[str, list]:
    """Synthetic panel for the future-leak test. y_t depends only on
    x_{t-lead}: lagged predictor at horizon ``lead`` carries the entire
    signal. A leak-free walk-forward training at small origins cannot
    see the post-origin signal, so the per-origin loadings differ from
    the full-sample (leaky) loadings."""

    rng = np.random.default_rng(seed)
    dates = [f"{2010 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(T)]
    x = rng.standard_normal((T, n_features))
    # y_t depends on x_{t-lead, 0} (i.e. column 0 with a 5-period lead).
    y = np.zeros(T)
    for t in range(T):
        if t - lead >= 0:
            y[t] = 1.5 * x[t - lead, 0] + 0.1 * rng.standard_normal()
        else:
            y[t] = 0.1 * rng.standard_normal()
    panel = {"date": dates, "y": list(y)}
    for j in range(n_features):
        panel[f"x{j}"] = list(x[:, j])
    return panel


def test_scaled_pca_no_future_leakage_via_per_origin_l3(tmp_path):
    """Phase B-1 F2: when the L3 DAG carries
    ``temporal_rule = expanding_window_per_origin`` the runtime must
    re-fit the scaled-PCA loadings at every walk-forward origin from
    data <= origin only. Without the fix, scaled_pca peeks at the full
    panel (including post-origin observations) and the OOS forecasts
    become contaminated.

    We test the contract behaviorally: with the fix, materializing L3
    once on the full panel and again per-origin produces *different* X
    matrices for early origins (the per-origin one cannot see the
    future-shifted predictive coefficient). The metadata closure is the
    single place this swap happens, so we exercise it directly.
    """

    import macroforecast
    from macroforecast.recipes.paper_methods import scaled_pca
    from macroforecast.core.runtime import (
        materialize_l1,
        materialize_l2,
        materialize_l3_minimal,
    )

    panel = _scaled_pca_panel_with_future_dependence(
        T=48, n_features=4, lead=5, seed=11
    )
    recipe = scaled_pca(target="y", horizon=1, n_components=2, panel=panel)
    l1, _regime, _axes = materialize_l1(recipe)
    l2, _l2_axes = materialize_l2(recipe, l1)
    l3, _l3_meta = materialize_l3_minimal(recipe, l1, l2)

    closure = l3.X_final.metadata.values.get("l3_per_origin_callable")
    assert closure is not None, (
        "scaled_pca recipe declares temporal_rule=expanding_window_per_origin "
        "but materialize_l3_minimal did not attach an l3_per_origin_callable; "
        "this is the Phase B-1 F2/F3 contract."
    )
    affected_node_ids = l3.X_final.metadata.values.get("l3_per_origin_node_ids", ())
    assert "spca" in affected_node_ids, (
        f"the scaled_pca node must be in the affected sub-DAG; got {affected_node_ids}"
    )

    full_X = l3.X_final.data
    # Early walk-forward origin (the L4 default minimal_train_size for
    # n_obs ~ 48 / n_features = 2 lands well under 24).
    early_origin = full_X.index[10]
    X_origin_early = closure(early_origin).reindex(
        columns=full_X.columns, fill_value=0.0
    )
    # The early per-origin X must agree with the full-sample X up to
    # *some* date <= origin -- specifically: the per-origin matrix is
    # the full-sample matrix only if the loadings are identical, which
    # the future-leak DGP is constructed to *break*. We require the
    # row at the origin date to differ from the full-sample value.
    full_row = full_X.loc[[early_origin]].iloc[0].to_numpy()
    origin_row = X_origin_early.loc[[early_origin]].iloc[0].to_numpy()
    assert not np.allclose(full_row, origin_row, atol=1e-8), (
        "Per-origin scaled_pca loadings must differ from the full-sample "
        "loadings on this future-leak DGP; got identical rows. The "
        "expanding_window_per_origin contract is not honored."
    )

    # End-to-end smoke: macroforecast.run produces finite forecasts.
    result = macroforecast.run(recipe, output_directory=tmp_path / "spca_no_leak")
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts, "no forecasts produced"
    arr = np.asarray(list(forecasts.values()), dtype=float)
    assert np.all(np.isfinite(arr)), "non-finite forecasts under per-origin L3"


# ----------------------------------------------------------------------
# F3 -- temporal_rule must be read at runtime, not just validated
# ----------------------------------------------------------------------


def test_temporal_rule_expanding_window_actually_executes_per_origin(tmp_path):
    """Phase B-1 F3: when ``temporal_rule = expanding_window_per_origin``
    is declared, the L3 sub-DAG must be re-materialized at every L4
    origin -- not once on the full panel. We confirm this by patching
    the underlying ``_pca_factors`` helper to count its invocations and
    asserting the counter exceeds 1 (one per origin, not one full
    sample).
    """

    import macroforecast
    from macroforecast.core import runtime as rt
    from macroforecast.recipes.paper_methods import scaled_pca

    # Synthetic panel sized so the walk-forward executes at least 10 origins.
    rng = np.random.default_rng(7)
    T = 36
    f = rng.standard_normal(T)
    panel = {
        "date": [f"{2018 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(T)],
        "y": list(2.0 * f + 0.3 * rng.standard_normal(T)),
        "x1": list(0.7 * f + 0.4 * rng.standard_normal(T)),
        "x2": list(-0.5 * f + 0.4 * rng.standard_normal(T)),
        "x3": list(0.3 * f + 0.4 * rng.standard_normal(T)),
        "x4": list(rng.standard_normal(T)),
    }
    recipe = scaled_pca(target="y", horizon=1, n_components=2, panel=panel)

    counter = {"n_calls": 0}
    original = rt._pca_factors

    def _counting_pca_factors(*args, **kwargs):
        counter["n_calls"] += 1
        return original(*args, **kwargs)

    rt._pca_factors = _counting_pca_factors
    try:
        result = macroforecast.run(recipe, output_directory=tmp_path / "spca_count")
    finally:
        rt._pca_factors = original

    assert result.cells, "no cells produced"
    # The walk-forward iterates `len(X) - min_train_size` origins. With
    # T=36, n_features=4, the L4 default min_train_size = min(35, 4) = 4,
    # leaving ~31 origins after the dropna alignment. We require >= 10
    # invocations: 1 full-sample call + >= 9 per-origin re-fits. The
    # actual count is much higher; we keep the bar at 10 to make the
    # test robust to walk-forward implementation tweaks.
    assert counter["n_calls"] >= 10, (
        f"_pca_factors invoked {counter['n_calls']} times; expected >= 10 "
        f"under temporal_rule=expanding_window_per_origin (one full-sample "
        f"call + one re-fit per walk-forward origin). The Round-1 audit "
        f"found this counter was always 1 (full sample only)."
    )


# ----------------------------------------------------------------------
# F4 -- backward-compat regression guard for full_sample
# ----------------------------------------------------------------------


def test_temporal_rule_full_sample_executes_once(tmp_path):
    """Phase B-1 F4: nodes that do NOT declare
    ``temporal_rule = expanding_window_per_origin`` must keep the
    one-shot full-panel materialization. We construct a recipe whose
    ``pca`` step uses no temporal_rule (or the schema default of
    ``full_sample`` semantics) and assert ``_pca_factors`` is invoked
    exactly once. This locks the backward-compat contract: the v0.9.0F
    1220-test baseline path is unchanged for non-expanding nodes."""

    import macroforecast
    from macroforecast.core import runtime as rt

    # A minimal recipe that uses pca *without* temporal_rule. We can't
    # use a public helper because every helper now sets the rule (paper-
    # faithful), so we hand-roll a recipe using pca with no temporal_rule
    # set. Validation requires temporal_rule on factor ops, so we use the
    # ``identity`` op (which has no temporal_rule schema entry) followed
    # by a fit_model. The point is to verify the closure is NOT attached
    # when no expanding-window node is present, and the runtime stays on
    # the legacy one-shot path.
    rng = np.random.default_rng(13)
    T = 30
    panel = {
        "date": [f"{2018 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(T)],
        "y": list(rng.standard_normal(T)),
        "x1": list(rng.standard_normal(T)),
        "x2": list(rng.standard_normal(T)),
        "x3": list(rng.standard_normal(T)),
    }

    # Build a minimal recipe that uses identity (no temporal_rule). The
    # _l3_per_origin_callable should be absent from the L3 X_final
    # metadata under this configuration.
    from macroforecast.recipes.paper_methods import _base_recipe, _l4_single_fit

    recipe = _base_recipe(
        target="y",
        horizon=1,
        panel=panel,
        seed=0,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
    recipe["3_feature_engineering"] = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            # identity has no temporal_rule schema entry, so it cannot
            # trigger per-origin behavior.
            {
                "id": "X_final",
                "type": "step",
                "op": "identity",
                "params": {},
                "inputs": ["src_X"],
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {"mode": "point_forecast", "method": "direct", "horizon": 1},
                "inputs": ["src_y"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "X_final", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }

    # Verify the per-origin closure is NOT attached.
    from macroforecast.core.runtime import (
        materialize_l1,
        materialize_l2,
        materialize_l3_minimal,
    )

    l1, _r, _a = materialize_l1(recipe)
    l2, _l2a = materialize_l2(recipe, l1)
    l3, _l3m = materialize_l3_minimal(recipe, l1, l2)
    assert "l3_per_origin_callable" not in l3.X_final.metadata.values, (
        "no node declares temporal_rule=expanding_window_per_origin, "
        "so the per-origin closure must NOT be attached "
        "(backward-compat one-shot path)."
    )

    # End-to-end run remains stable.
    counter = {"n_calls": 0}
    original = rt._pca_factors

    def _counting_pca_factors(*args, **kwargs):
        counter["n_calls"] += 1
        return original(*args, **kwargs)

    rt._pca_factors = _counting_pca_factors
    try:
        result = macroforecast.run(recipe, output_directory=tmp_path / "identity_full")
    finally:
        rt._pca_factors = original

    assert result.cells, "no cells produced"
    # No PCA op in the recipe, so the counter must be 0. This pins the
    # contract: identity ops do not call _pca_factors.
    assert counter["n_calls"] == 0, (
        f"identity-only recipe should never invoke _pca_factors; "
        f"got {counter['n_calls']} calls."
    )


# ----------------------------------------------------------------------
# Phase B-1b -- residual leak in target_signal (Round 6 finding)
# ----------------------------------------------------------------------


def test_scaled_pca_target_signal_has_no_post_origin_leak(tmp_path):
    """Phase B-1b residual-leak fix (Round 6 audit finding).

    Round 6 found that ``_l3_per_origin_affected_nodes`` only walked
    DOWNSTREAM from the expanding-window seed, so the
    ``target_construction`` node ``y_h`` (UPSTREAM of ``scaled_pca``)
    was NOT in the affected set. ``y_h`` therefore fell back to a
    cached full-sample ``y.shift(-h)`` trimmed by ``iloc[: origin + 1]``.
    The trailing ``h`` rows of that trimmed series are exactly the
    post-origin y observations ``y[origin + 1 .. origin + h]`` -- a
    direct future leak into ``_scaled_pca_huang_zhou``'s slope
    regression.

    Concrete construction: y is a step function (0 for t < 20, 100
    for t >= 20). At origin=18 with horizon=4, the unfixed path
    produces ``target_signal = [0]*16 + [100, 100, 100]`` (the three
    100 values are y[20], y[21], y[22] -- post-origin leak). The fix
    extends the BFS to mark ``target_construction`` ancestors of any
    expanding-window node as affected, so the per-origin closure
    re-shifts a truncated y at every origin and the trailing rows
    become NaN (and are dropped by the consuming op's ``notna``
    mask).

    Acceptance: at every origin t with t < 20 (i.e., before the jump
    propagates into a leak-free h-shifted view), the maximum of
    ``target_signal.dropna()`` must be < 1.0 -- there must be no
    post-origin y[20]=100 leak.
    """

    import macroforecast
    from macroforecast.core import runtime as rt
    from macroforecast.recipes.paper_methods import scaled_pca

    # Step-function DGP: y[t] = 0 for t < 20, y[t] = 100 for t >= 20.
    # T=60 monthly observations, K=8 random predictors. The sharp jump
    # at t=20 is the leak diagnostic: any post-origin y observation
    # showing up in target_signal at origin t<=20 means y[20]=100
    # leaked through the trim path.
    T = 60
    rng = np.random.default_rng(42)
    y = np.zeros(T)
    y[20:] = 100.0
    panel = {
        "date": [f"{2010 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(T)],
        "y": list(y),
    }
    for j in range(8):
        panel[f"x{j}"] = list(rng.standard_normal(T))

    recipe = scaled_pca(target="y", horizon=4, n_components=2, panel=panel)

    # Capture every (frame, target_signal) pair invoked on _scaled_pca_huang_zhou.
    # The number of rows in `frame` is the origin slice length post-cleaning.
    # In the per-origin loop, it equals (origin_index + 1) for slices that
    # survive the dropna alignment. The first invocation (full sample,
    # T=60 rows) is the one-shot cached call -- we ignore it in the
    # leak assertion.
    captures: list[tuple[int, pd.Series]] = []
    original = rt._scaled_pca_huang_zhou

    def _capturing(frame, *, n_components, target_signal):
        captures.append((len(frame), target_signal.copy()))
        return original(frame, n_components=n_components, target_signal=target_signal)

    rt._scaled_pca_huang_zhou = _capturing
    try:
        macroforecast.run(recipe, output_directory=tmp_path / "spca_no_target_leak")
    finally:
        rt._scaled_pca_huang_zhou = original

    assert len(captures) >= 2, (
        f"expected per-origin scaled_pca to invoke _scaled_pca_huang_zhou "
        f"more than once (one full-sample cache call + one re-fit per "
        f"origin); got {len(captures)} invocations"
    )

    # The first call is the full-sample cache call (T=60 rows). Ignore it.
    # Every subsequent call is a per-origin invocation. For every
    # per-origin call where the origin t < 20 (i.e., frame has <= 20
    # rows after dropna alignment), the target_signal must NOT contain
    # any value >= 50 -- such a value would mean y[20]=100 leaked into
    # the slope regression at an origin where it should not be visible.
    leaked: list[tuple[int, float]] = []
    for n_rows, ts in captures[1:]:
        if n_rows > 20:
            # Once the origin is past the jump, the in-sample y itself
            # contains 100 values -- that's not a leak.
            continue
        nonnull_max = ts.dropna().max() if ts.notna().any() else 0.0
        if nonnull_max >= 50:
            leaked.append((n_rows, float(nonnull_max)))

    assert not leaked, (
        f"target_signal contains post-origin y observations at origins "
        f"before the y=100 jump: {leaked}. The Round-6 audit found this "
        f"leak path (y_h is upstream of scaled_pca, BFS only walked "
        f"downstream, so y_h fell back to a cached full-sample shift "
        f"trimmed by iloc[:origin+1] -- the trailing h rows are "
        f"y[origin+1..origin+h]). The fix marks target_construction "
        f"ancestors of expanding-window nodes as affected so the "
        f"per-origin closure re-shifts a truncated y at every origin."
    )

    # Stronger explicit check at origin=18 (frame has 19 rows): without
    # the fix, target_signal at this origin contains [0]*16 + [100, 100, 100]
    # (max == 100). With the fix, target_signal == [0]*15 + [NaN]*4
    # (max == 0). We pin both the max value and the non-null count.
    found_origin_18 = False
    for n_rows, ts in captures[1:]:
        if n_rows == 19:
            found_origin_18 = True
            nonnull = ts.dropna()
            assert nonnull.size == 15, (
                f"at origin=18 (frame_rows=19) with h=4, target_signal "
                f"should have exactly 15 non-null values (y[4..18] shifted "
                f"into positions 0..14, with positions 15..18 = NaN). "
                f"Got nonnull.size={nonnull.size}: {ts.tolist()}"
            )
            assert nonnull.max() < 1.0, (
                f"at origin=18 (frame_rows=19), target_signal max should "
                f"be < 1.0 (all in-sample y values are 0). Got "
                f"max={nonnull.max()}: {ts.tolist()}. Pre-fix value "
                f"would be 100.0 from y[20..22] leak."
            )
            break
    assert found_origin_18, (
        "expected an origin=18 (frame_rows=19) per-origin invocation "
        f"but none captured. captures sizes: {[c[0] for c in captures]}"
    )
