"""Phase B-1c L4 walk-forward training-y h-step leak tests.

Round-7 audit finding: at every walk-forward origin ``position``, the
sequential and parallel L4 fit loops sliced the cached
``y = y_h = y_orig.shift(-h)`` series via ``y.iloc[start:position]``.
For ``h >= 2`` this includes pairs whose ``y_orig`` calendar dates exceed
the origin -- a direct post-origin leak into ``model.fit``. Concrete:
``y.iloc[position - 1] = y_orig.iloc[position - 1 + h]``, which sits at
time ``position - 1 + h > position``.

The Phase B-1c fix replaces the slice end with ``position - h + 1``
(guarded by ``max(start, ...)``) at all three sites in
``macroforecast/core/runtime.py``:

* ``materialize_l4_minimal`` sequential fit-node loop (~line 1572).
* ``_run_l4_fit_node`` parallel ``_origin_step`` (~line 1848).
* ``_run_l4_fit_node`` sequential fallback (~line 1886).

The two tests below are procedure tests: they monkey-patch the model fit
to capture ``(train_X, train_y)`` per origin, then assert the leak-free
end-index contract. The h=1 case must be a no-op (last observation
remains in the slice -- no behavior change for the most common path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Step-function DGP shared by both tests
# ----------------------------------------------------------------------


def _step_panel(*, T: int = 60, K: int = 4, jump_t: int = 20, seed: int = 42):
    """y_orig is 0 for t < jump_t and 100 for t >= jump_t. Predictors are
    iid normal noise. The sharp jump is the leak diagnostic: any
    post-origin y observation showing up in train_y at an early origin
    means y[jump_t]=100 leaked through the slice."""

    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    y[jump_t:] = 100.0
    panel = {
        "date": [f"{2010 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(T)],
        "y": list(y),
    }
    for j in range(K):
        panel[f"x{j}"] = list(rng.standard_normal(T))
    return panel


def _capture_train_y_per_origin(recipe, tmp_path):
    """Patch ``sklearn.linear_model.Ridge.fit`` to capture every
    (train_X.shape, train_y.copy()) pair. Returns the captured list."""

    import macroforecast
    from sklearn.linear_model import Ridge

    captures: list[tuple[tuple[int, int], pd.Series]] = []
    original_fit = Ridge.fit

    def _capturing_fit(self, X, y, sample_weight=None):
        if isinstance(y, pd.Series):
            captures.append((tuple(np.shape(X)), y.copy()))
        elif isinstance(X, pd.DataFrame):
            captures.append(
                (tuple(np.shape(X)), pd.Series(np.asarray(y), index=X.index))
            )
        else:
            captures.append((tuple(np.shape(X)), pd.Series(np.asarray(y))))
        return original_fit(self, X, y, sample_weight=sample_weight)

    Ridge.fit = _capturing_fit
    try:
        macroforecast.run(recipe, output_directory=tmp_path / "phase_b1c_run")
    finally:
        Ridge.fit = original_fit
    return captures


# ----------------------------------------------------------------------
# Test 1: h=4 must drop the trailing 3 leaky rows
# ----------------------------------------------------------------------


def test_l4_train_y_no_h_step_post_origin_leak(tmp_path):
    """Phase B-1c: with h=4, the walk-forward training y at origin
    ``position`` must contain only ``y_orig`` observations whose calendar
    date is ``<= position``. The cached ``y_h = y_orig.shift(-h)`` puts
    the y at index i equal to ``y_orig[i + 4]``, so the leak-free end
    index of the slice is ``position - h + 1``. Without the fix, the
    slice ends at ``position`` -- including 3 post-origin observations.

    Concrete check: at origin index 18 with the step-function DGP
    (y=0 for t<20, y=100 for t>=20), the leak-free train_y has
    ``y_h.iloc[0..14]`` -- all 0 because they map to ``y_orig.iloc[4..18]``,
    none of which has crossed the jump. Pre-fix, train_y had
    ``y_h.iloc[0..17]``, the last three of which map to
    ``y_orig.iloc[19..21]`` -- two of those are 100 (post-jump leak).

    Acceptance: ``train_y.dropna().max() < 50`` -- no 100-valued leak.
    """

    from macroforecast.recipes.paper_methods import _base_recipe, _l4_single_fit

    panel = _step_panel(T=60, K=4, jump_t=20, seed=42)
    recipe = _base_recipe(
        target="y",
        horizon=4,
        panel=panel,
        seed=0,
        l4=_l4_single_fit("ridge", {"alpha": 1.0, "min_train_size": 6}),
    )

    captures = _capture_train_y_per_origin(recipe, tmp_path)
    assert captures, "no Ridge.fit invocations captured"

    # Drop the final full-sample fit (used for the model artifact, not
    # the walk-forward) -- it has T=60 rows AFTER the dropna alignment
    # removes the trailing h NaNs from y_h, so n_rows == T - h == 56.
    walk_captures = [(shape, ts) for (shape, ts) in captures if shape[0] < 56]

    # Find origin index 18 -- after dropna alignment of y_h (which drops
    # the trailing h=4 rows where shift produces NaN), the X/y panel has
    # T - h = 56 rows. Origin index 18 in that aligned panel corresponds
    # to position=18 and should produce a train_y of size
    # max(start, position - h + 1) = max(0, 18 - 3) = 15.
    matched = [(shape, ts) for (shape, ts) in walk_captures if shape[0] == 15]
    assert matched, (
        f"expected at least one Ridge.fit invocation with train_X shape "
        f"(15, 4) (origin=18, h=4 leak-free slice). Got captures shapes: "
        f"{[s for s, _ in walk_captures]}"
    )
    _shape, train_y = matched[0]
    assert len(train_y.dropna()) == 15
    assert train_y.dropna().max() < 50.0, (
        f"at origin=18 with h=4, train_y must not contain any post-origin "
        f"y_orig observations (y_orig[19..]=100). Got max="
        f"{train_y.dropna().max()}, values={train_y.tolist()}. Pre-fix "
        f"value would be 100.0 from y_orig[20]/y_orig[21] leaking via "
        f"y_h.iloc[16] / y_h.iloc[17]."
    )

    # Also verify NO walk-forward train_y at strictly leak-prone origins
    # contains any value >= 50. The L3 lag op drops the first row of the
    # original panel, so aligned-position s maps to original index s + 1.
    # ``y_aligned.iloc[s] = y_orig.iloc[s + 1 + h]``. The 100-valued
    # entries first appear at aligned-position s with s + 1 + 4 >= 20,
    # i.e., s >= 15. Under the leak-free fix, train_y at origin p
    # contains rows ``[0..p - h]`` (size p - h + 1 = p - 3), with last
    # admitted row at aligned-position s = p - 4. The first origin p
    # whose admitted training set legitimately contains a 100 is the
    # smallest p with p - 4 >= 15, i.e., p >= 19 (shape (16, 4)).
    # Any 100 at shape[0] < 16 is therefore a leak from the old
    # ``[start:position]`` slice. Pre-fix, p=18 (shape (18, 4) under
    # old slice) admitted rows [0..17], where row 16 = y_orig[21] = 100
    # is post-origin (origin = original-index 19). The fix removes that.
    leaks: list[tuple[tuple[int, int], float]] = []
    for shape, ts in walk_captures:
        if shape[0] < 16:
            m = ts.dropna().max() if ts.notna().any() else 0.0
            if m >= 50:
                leaks.append((shape, float(m)))
    assert not leaks, (
        f"train_y contains post-origin y_orig values at origins where "
        f"the leak-free slice should have excluded them: {leaks}. The "
        f"Round-7 leak-fix is incomplete -- one of the three slice "
        f"sites in runtime.py still uses the old ``[start:position]`` "
        f"end index."
    )


# ----------------------------------------------------------------------
# Test 2: h=1 must be unchanged (backward-compat regression guard)
# ----------------------------------------------------------------------


def test_l4_train_y_h_equals_1_unchanged(tmp_path):
    """Phase B-1c backward-compat regression guard: at h=1 the leak-fix
    end index ``position - h + 1 = position`` is identical to the
    pre-fix slice. The h=1 walk-forward must therefore produce the
    same train_y rows as before (one row per position, ending at
    ``position - 1``)."""

    from macroforecast.recipes.paper_methods import _base_recipe, _l4_single_fit

    panel = _step_panel(T=60, K=4, jump_t=20, seed=42)
    recipe = _base_recipe(
        target="y",
        horizon=1,
        panel=panel,
        seed=0,
        l4=_l4_single_fit("ridge", {"alpha": 1.0, "min_train_size": 6}),
    )

    captures = _capture_train_y_per_origin(recipe, tmp_path)
    assert captures, "no Ridge.fit invocations captured"

    # At h=1, ``y_h = y_orig.shift(-1)`` -- one trailing NaN. The aligned
    # panel has T - 1 = 59 rows. The full-sample fit at the end has
    # shape (59, 4). Walk-forward fits range over min_train_size=6 to 58.
    walk_captures = [(shape, ts) for (shape, ts) in captures if shape[0] < 59]
    assert walk_captures, "expected walk-forward Ridge.fit captures"

    # At every walk-forward origin p, train_y has shape (p, 4) (since
    # h=1 -> end index = p). The trailing position p ranges from 6
    # (min_train_size) to 58. So shapes go from (6, 4) up to (58, 4).
    shapes = sorted({s[0] for s, _ in walk_captures})
    assert 6 in shapes, f"missing the min_train_size=6 origin; shapes={shapes}"
    assert 58 in shapes, f"missing the final walk-forward origin; shapes={shapes}"

    # Verify the (p=20, train_y has shape (20, 4)) capture: train_y is
    # ``y_h.iloc[0..19] = y_orig.iloc[1..20]`` -- the LAST entry is
    # ``y_orig[20] = 100``. At h=1 this is NOT a leak (origin=20 admits
    # ``y_orig[20]`` because ``20 + 1 - 1 = 20 <= 20``). The pre-fix
    # behavior includes this row; the post-fix behavior must match.
    matched = [(shape, ts) for (shape, ts) in walk_captures if shape[0] == 20]
    assert matched, (
        f"expected a Ridge.fit invocation at origin=20 with train_y "
        f"shape (20, 4). Got shapes: {shapes}"
    )
    _shape, train_y = matched[0]
    # The last value in train_y at origin=20 is y_h[19] = y_orig[20] = 100.
    # This is admissible at h=1 -- the test pins that the h=1 path retains
    # this row (i.e., the fix is a no-op at h=1).
    assert train_y.iloc[-1] == 100.0, (
        f"at h=1 origin=20, train_y last value must be 100 "
        f"(y_orig[20]); got {train_y.iloc[-1]}. The Phase B-1c fix has "
        f"changed the h=1 behavior -- it should be a strict no-op."
    )
    assert len(train_y) == 20
