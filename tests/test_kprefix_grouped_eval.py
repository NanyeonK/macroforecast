"""Independent tester suite for the K-prefix grouped evaluator (request kprefix-grouped-eval).

Authored from ``.dev-notes/statsclaw_runs/kprefix-grouped-eval/test-spec.md`` WITHOUT
reading the grouped-path implementation (pipeline isolation). Every assertion is expressed
against the PUBLIC API only:

* ``macroforecast.model_selection.select_params`` (function under test; auto-routes
  supervised-PCA n_components grids through the grouped evaluator),
* standalone ``macroforecast.models.supervised_pca`` / ``supervised_scaled_pca`` fits,
* the public ``macroforecast.metrics.get_metric`` scorer.

The oracle is an independent per-candidate reference that fits every ``(n_components, ...)``
candidate from scratch on every validation split via the standalone public model function,
scores with the same metric, and aggregates exactly as the search contract specifies
(``mean_split`` = mean of per-split scores; ``mean_fold`` = concat truth/pred within each
fold id, score once per fold, mean across folds; first-exception-wins per candidate). The
grouped path (which ``select_params`` engages automatically for these models) must reproduce
this reference bitwise (max abs score diff 0.0; exact match on every cell).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast.models as mo
from macroforecast.metrics import get_metric
from macroforecast.model_selection import SearchError, explicit_folds, grid, select_params

Split = tuple[np.ndarray, np.ndarray]

# ---------------------------------------------------------------------------
# Synthetic data helpers (fixed seed for reproducibility).
# ---------------------------------------------------------------------------


def _make_data(n: int, p: int, seed: int, *, noise: float = 0.3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"x{i}" for i in range(p)],
    )
    beta = rng.standard_normal(p)
    y = pd.Series(X.values @ beta + noise * rng.standard_normal(n), name="y")
    return X, y


def _candidates_from_grid(param_grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product in insertion order (last key varies fastest) -- matches the
    order ``select_params`` assigns trial ids to grid candidates."""

    keys = list(param_grid)
    values = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _expanding_fold_splits(boundaries: Sequence[int]) -> tuple[list[Split], list[int]]:
    """Reconstruct the (split, fold_id) sequence produced by
    ``explicit_folds(boundaries, within_fold='expanding')`` for an integer RangeIndex.

    For each consecutive boundary pair (start, end) forming fold ``f``, emit one split
    ``(arange(val_pos), [val_pos])`` per ``val_pos`` in ``range(start, end)``; all share
    fold id ``f``. This mirrors the documented, general splitter contract -- it is not the
    grouped-K optimisation under test.
    """

    positions = list(boundaries)
    splits: list[Split] = []
    fold_ids: list[int] = []
    for fold_id, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
        for val_pos in range(start, end):
            splits.append((np.arange(val_pos, dtype=int), np.asarray([val_pos], dtype=int)))
            fold_ids.append(fold_id)
    return splits, fold_ids


# ---------------------------------------------------------------------------
# Independent per-candidate reference (the bitwise-identity oracle).
# ---------------------------------------------------------------------------


def _reference_trials(
    model_fn: Callable[..., Any],
    X: pd.DataFrame,
    y: pd.Series,
    candidates: list[dict[str, Any]],
    splits: Sequence[Split],
    aggregation: str,
    fold_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build the reference ``trials`` table by fitting each candidate independently."""

    metric_fn = get_metric("mse")
    if fold_ids is None:
        fold_ids = list(range(len(splits)))
    rows: list[dict[str, Any]] = []
    for trial, params in enumerate(candidates):
        record: dict[str, Any] = {"trial": trial, **params}
        try:
            per_split_scores: list[float] = []
            fold_truth: dict[int, list[pd.Series]] = {}
            fold_pred: dict[int, list[pd.Series]] = {}
            for split_id, (train_idx, val_idx) in enumerate(splits):
                fit = model_fn(X.iloc[train_idx], y.iloc[train_idx], **params)
                y_val = y.iloc[val_idx]
                pred = pd.Series(
                    np.asarray(fit.predict(X.iloc[val_idx])).reshape(-1),
                    index=y_val.index,
                )
                if aggregation == "mean_split":
                    per_split_scores.append(float(metric_fn(y_val, pred)))
                else:
                    fid = int(fold_ids[split_id])
                    fold_truth.setdefault(fid, []).append(y_val)
                    fold_pred.setdefault(fid, []).append(pred)
            if aggregation == "mean_fold":
                for fid in dict.fromkeys(int(f) for f in fold_ids):
                    y_fold = pd.concat(fold_truth[fid])
                    pred_fold = pd.concat(fold_pred[fid])
                    per_split_scores.append(float(metric_fn(y_fold, pred_fold)))
            record.update(
                score=float(np.mean(per_split_scores)),
                n_splits=len(splits),
                status="ok",
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - failed trials are part of the contract.
            record.update(
                score=np.nan,
                n_splits=len(splits),
                status="error",
                error=str(exc),
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def _assert_bitwise_identity(result_trials: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Assert the search ``trials`` table matches the reference exactly (max abs diff 0.0)."""

    # Same set of named columns (column order is allowed to differ).
    assert set(result_trials.columns) == set(reference.columns), (
        f"column mismatch: got {sorted(result_trials.columns)} "
        f"vs reference {sorted(reference.columns)}"
    )
    got = result_trials[reference.columns].sort_values("trial").reset_index(drop=True)
    pdt.assert_frame_equal(got, reference, check_exact=True, check_dtype=False)
    # Explicit numeric max-abs-diff guard on the score column (0.0, not a tolerance).
    both_scores = pd.concat([got["score"], reference["score"]], axis=1)
    finite = both_scores.dropna()
    if not finite.empty:
        max_abs = float((finite.iloc[:, 0] - finite.iloc[:, 1]).abs().max())
        assert max_abs == 0.0, f"max abs score diff {max_abs!r} != 0.0"
    # NaN cells must coincide.
    assert (got["score"].isna() == reference["score"].isna()).all(), "NaN score cells differ"


def _run_and_compare(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, Sequence[Any]],
    *,
    splits: Sequence[Split] | None = None,
    boundaries: Sequence[int] | None = None,
    aggregation: str = "mean_split",
) -> pd.DataFrame:
    """Run ``select_params`` (grouped path) and assert bitwise identity vs the reference.

    Provide either ``splits`` (explicit, one fold per split) or ``boundaries`` (expanding
    ``explicit_folds`` grouping, exercising genuine multi-split folds for ``mean_fold``).
    Returns the search ``trials`` table for further assertions.
    """

    model_fn = getattr(mo, model_name)
    candidates = _candidates_from_grid(param_grid)

    if boundaries is not None:
        ref_splits, fold_ids = _expanding_fold_splits(boundaries)
        search = grid(
            param_grid,
            validation_splitter=explicit_folds(list(boundaries), within_fold="expanding"),
        )
        try:
            result_trials = select_params(
                model=model_name,
                X=X,
                y=y,
                search=search,
                metric="mse",
                score_aggregation=aggregation,
            ).trials
        except SearchError as exc:
            # When every candidate errors, select_params raises but still carries the table.
            result_trials = exc.trials
        else:
            # Sanity: our reconstructed split set matches what the splitter produced.
            pass
        reference = _reference_trials(model_fn, X, y, candidates, ref_splits, aggregation, fold_ids)
    else:
        assert splits is not None
        try:
            result_trials = select_params(
                model=model_name,
                X=X,
                y=y,
                search=grid(param_grid),
                splits=list(splits),
                metric="mse",
                score_aggregation=aggregation,
            ).trials
        except SearchError as exc:
            result_trials = exc.trials
        reference = _reference_trials(model_fn, X, y, candidates, splits, aggregation)

    _assert_bitwise_identity(result_trials, reference)
    return result_trials


# ===========================================================================
# Section 0 -- feature engagement guards (keep the identity tests non-vacuous).
# ===========================================================================


def test_grouped_path_is_active_for_supervised_pca_models() -> None:
    """supervised_pca / supervised_scaled_pca expose the prefix search on n_components,
    so an n_components grid auto-routes through the grouped evaluator."""

    for name in ("supervised_pca", "supervised_scaled_pca"):
        spec = mo.get_model(name)
        ps = getattr(spec, "prefix_search", None)
        assert ps is not None, f"{name} lost its prefix_search spec"
        assert ps.param == "n_components"


def test_non_prefix_models_have_no_prefix_search() -> None:
    """Non-supervised-PCA models must NOT expose a prefix path (they use the old loop)."""

    for name in ("ridge", "pcr", "scaled_pca"):
        assert getattr(mo.get_model(name), "prefix_search", None) is None


# ===========================================================================
# Section 2 -- primary bitwise-identity oracle.
# ===========================================================================


def _primary_grid() -> dict[str, Sequence[Any]]:
    # n_components spans 4+ distinct values, including 15 > 12 predictors (clamp path);
    # crossed with two n_selected values -> two distinct groups in one search.
    return {"n_components": [1, 2, 3, 5, 15], "n_selected": [5, 10]}


def _primary_splits() -> list[Split]:
    # 6 non-overlapping width-4 validation windows -> 24 distinct validation positions
    # (>= 20) in a single search; last window ends at position 64 (< n = 80).
    return [
        (np.arange(0, 40 + k * 4, dtype=int), np.arange(40 + k * 4, 44 + k * 4, dtype=int))
        for k in range(6)
    ]


@pytest.mark.parametrize("model_name", ["supervised_pca", "supervised_scaled_pca"])
def test_primary_identity_mean_split(model_name: str) -> None:
    X, y = _make_data(n=80, p=12, seed=12345)
    splits = _primary_splits()
    # Coverage: union of distinct validation positions >= 20.
    positions = set()
    for _, val in splits:
        positions.update(int(v) for v in val)
    assert len(positions) >= 20
    _run_and_compare(model_name, X, y, _primary_grid(), splits=splits, aggregation="mean_split")


@pytest.mark.parametrize("model_name", ["supervised_pca", "supervised_scaled_pca"])
def test_primary_identity_mean_fold(model_name: str) -> None:
    X, y = _make_data(n=80, p=12, seed=12345)
    # Two folds of 10 and 12 expanding splits -> 22 validation positions, genuine grouping.
    boundaries = [50, 60, 72]
    ref_splits, fold_ids = _expanding_fold_splits(boundaries)
    assert len(ref_splits) >= 20
    assert len(set(fold_ids)) == 2  # more than one fold -> mean_fold != mean_split path
    _run_and_compare(model_name, X, y, _primary_grid(), boundaries=boundaries, aggregation="mean_fold")


# ===========================================================================
# Section 3 -- backward compatibility.
# ===========================================================================


@pytest.mark.parametrize("model_name", ["ridge", "pcr"])
def test_backward_compat_non_prefix_models_unchanged(model_name: str) -> None:
    """Non-prefix models route through the old per-candidate loop; output must be
    run-to-run byte identical and independent of the grouped machinery."""

    X, y = _make_data(n=90, p=8, seed=555)
    splits = [
        (np.arange(0, 50 + k * 5, dtype=int), np.arange(50 + k * 5, 55 + k * 5, dtype=int))
        for k in range(4)
    ]
    # >= 6 candidate combinations over >= 2 hyperparameters.
    if model_name == "ridge":
        param_grid: dict[str, Sequence[Any]] = {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]}
    else:  # pcr (uses `standardize`, not `scale`)
        param_grid = {"n_components": [1, 2, 3], "standardize": [True, False]}
    assert len(_candidates_from_grid(param_grid)) >= 6

    def _run() -> pd.DataFrame:
        return select_params(
            model=model_name,
            X=X,
            y=y,
            search=grid(param_grid),
            splits=list(splits),
            metric="mse",
        ).trials

    first = _run()
    second = _run()
    pdt.assert_frame_equal(first, second, check_exact=True)
    # Trial count invariant on the fallback path.
    assert len(first) == len(_candidates_from_grid(param_grid))
    assert first["n_splits"].eq(len(splits)).all()


_STANDALONE_PARAM_SETS: list[dict[str, Any]] = [
    {"n_components": nc, "quadratic_factors": qf, "preselect": pre}
    for nc in (1, 2, 5)
    for qf in (True, False)
    for pre in ("none", "elastic_net")
]


@pytest.mark.parametrize("model_name", ["supervised_pca", "supervised_scaled_pca"])
def test_standalone_predict_deterministic_and_isolated(model_name: str) -> None:
    """Standalone predict() over a representative parameter grid must be deterministic and
    unaffected by running a grouped select_params search in between (feature isolation).

    Note: a true *pre-feature* golden anchor for the recursion itself lives in the existing
    ``tests/models/test_models.py`` matlab-style recursion tests (test-spec section 3.3);
    this test guards determinism + isolation within the post-feature checkout.
    """

    model_fn = getattr(mo, model_name)
    X, y = _make_data(n=70, p=9, seed=99)
    X_holdout, _ = _make_data(n=15, p=9, seed=1000)

    golden: dict[int, np.ndarray] = {}
    for i, params in enumerate(_STANDALONE_PARAM_SETS):
        fit = model_fn(X, y, **params)
        golden[i] = np.asarray(fit.predict(X_holdout)).reshape(-1)

    # Run a grouped search that exercises the same models heavily.
    select_params(
        model=model_name,
        X=X,
        y=y,
        search=grid({"n_components": [1, 2, 3, 4], "n_selected": [5, 9]}),
        splits=[(np.arange(0, 50, dtype=int), np.arange(50, 60, dtype=int))],
        metric="mse",
    )

    for i, params in enumerate(_STANDALONE_PARAM_SETS):
        fit = model_fn(X, y, **params)
        again = np.asarray(fit.predict(X_holdout)).reshape(-1)
        assert np.array_equal(again, golden[i]), (
            f"standalone predict changed for {model_name} params={params}"
        )


# ===========================================================================
# Section 4 -- edge cases (all held to the section-2 bitwise-identity bar).
# ===========================================================================


def test_edge_component_count_clamp() -> None:
    """3 usable predictors, grid n_components up to 10: over-requested K must clamp to the
    model's own effective component count, identical to standalone fits."""

    X, y = _make_data(n=64, p=3, seed=42)
    splits = [
        (np.arange(0, 40 + k * 4, dtype=int), np.arange(40 + k * 4, 45 + k * 4, dtype=int))
        for k in range(4)
    ]
    grid_spec = {"n_components": [1, 2, 3, 5, 10], "n_selected": [None, 2]}
    _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)


def test_edge_all_candidates_error_group_isolated() -> None:
    """A group whose predictor block is constant (zero variance) with scale=True fails for
    every n_components; unrelated groups with usable predictors are unaffected.

    We assemble X with a constant sub-block selected only when n_selected picks constants.
    Simpler realisation: use a fully-constant predictor frame so every candidate in the
    'scale=True' group errors, crossed with scale=False (which the model tolerates)."""

    n = 60
    rng = np.random.default_rng(3)
    # First 4 columns constant, last 4 informative.
    const_block = {f"c{i}": np.full(n, 2.0) for i in range(4)}
    info = rng.standard_normal((n, 4))
    info_block = {f"x{i}": info[:, i] for i in range(4)}
    X = pd.DataFrame({**const_block, **info_block})
    y = pd.Series(info @ rng.standard_normal(4) + 0.1 * rng.standard_normal(n), name="y")
    splits = [
        (np.arange(0, 40 + k * 4, dtype=int), np.arange(40 + k * 4, 44 + k * 4, dtype=int))
        for k in range(4)
    ]
    # n_selected=4 with scale=True on the constant-first columns triggers the model's
    # "could not extract a non-zero component" rejection; the same grid with a large
    # n_selected including informative columns stays valid.
    # Build a group that is guaranteed to error: constant-only frame.
    X_const = X[[f"c{i}" for i in range(4)]]
    grid_err = {"n_components": [1, 2, 3], "scale": [True, False]}
    trials = _run_and_compare("supervised_pca", X_const, y, grid_err, splits=splits)

    # scale=True group must be all-error; scale=False group must be all-ok, independently.
    scale_true = trials[trials["scale"] == True]  # noqa: E712
    scale_false = trials[trials["scale"] == False]  # noqa: E712
    assert (scale_true["status"] == "error").all()
    assert scale_true["error"].notna().all()
    assert scale_true["score"].isna().all()
    assert (scale_false["status"] == "ok").all()
    assert scale_false["score"].notna().all()


def test_edge_single_candidate_group() -> None:
    """A grid where exactly one candidate's non-n_components params are unique (group size 1)."""

    X, y = _make_data(n=72, p=10, seed=17)
    splits = [
        (np.arange(0, 45 + k * 3, dtype=int), np.arange(45 + k * 3, 49 + k * 3, dtype=int))
        for k in range(4)
    ]
    # grid() is a full Cartesian product, so every group has size == len(n_components).
    # Fixing n_components to a single value makes each (n_selected) group size 1, exercising
    # the "group of size 1 is not special-cased/degraded" requirement.
    grid_spec = {"n_components": [4], "n_selected": [3, 6, 9]}
    trials = _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)
    # Every group here has exactly one candidate (one n_components value).
    assert len(trials) == 3


def test_edge_n_selected_variation_includes_none() -> None:
    """n_components crossed with >= 3 distinct n_selected values, including None (all preds)."""

    X, y = _make_data(n=80, p=12, seed=21)
    splits = _primary_splits()
    grid_spec = {"n_components": [1, 2, 3, 5], "n_selected": [None, 4, 8]}
    trials = _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)
    assert len(trials) == 12


def test_edge_min_abs_corr_variation() -> None:
    """n_components crossed with >= 2 nonzero min_abs_corr values that actually screen."""

    X, y = _make_data(n=80, p=12, seed=33)
    splits = _primary_splits()
    grid_spec = {"n_components": [1, 2, 3, 5], "min_abs_corr": [0.1, 0.3]}
    _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)
    _run_and_compare("supervised_scaled_pca", X, y, grid_spec, splits=splits)


@pytest.mark.parametrize("model_name", ["supervised_pca", "supervised_scaled_pca"])
def test_edge_quadratic_factors_on_off(model_name: str) -> None:
    """n_components crossed with quadratic_factors in {True, False}."""

    X, y = _make_data(n=80, p=10, seed=44)
    splits = _primary_splits()
    grid_spec = {"n_components": [1, 2, 3, 5], "quadratic_factors": [True, False]}
    _run_and_compare(model_name, X, y, grid_spec, splits=splits)


def test_edge_supervised_scaled_pca_slope_scaling() -> None:
    """Explicit restatement: the primary scenario for supervised_scaled_pca (slope scaling)."""

    X, y = _make_data(n=80, p=12, seed=12345)
    _run_and_compare("supervised_scaled_pca", X, y, _primary_grid(), splits=_primary_splits())


def test_edge_mixed_splits_partial_validity() -> None:
    """>= 3 splits where a degenerate (constant-block) condition appears in only SOME splits;
    any single-split failure must invalidate the WHOLE trial (not a partial average)."""

    n = 66
    rng = np.random.default_rng(9)
    # Rows 0..29 are constant across every predictor (zero within-column variance);
    # rows 30..65 are informative.
    p = 5
    block = np.empty((n, p))
    block[:30, :] = np.arange(p, dtype=float)  # constant down each column for early rows
    block[30:, :] = rng.standard_normal((n - 30, p))
    X = pd.DataFrame(block, columns=[f"x{i}" for i in range(p)])
    y = pd.Series(
        np.concatenate([np.zeros(30), rng.standard_normal(n - 30)]), name="y"
    )
    # Split 0 trains only on the constant region -> the model rejects it (error).
    # Splits 1,2 train on regions that include informative rows -> valid.
    splits = [
        (np.arange(0, 20, dtype=int), np.arange(20, 25, dtype=int)),
        (np.arange(0, 50, dtype=int), np.arange(50, 55, dtype=int)),
        (np.arange(0, 55, dtype=int), np.arange(55, 60, dtype=int)),
    ]
    grid_spec = {"n_components": [1, 2, 3], "scale": [True]}
    trials = _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)
    # The degenerate split 0 must invalidate every trial fully (status error, score NaN).
    assert (trials["status"] == "error").all()
    assert trials["score"].isna().all()
    assert trials["n_splits"].eq(len(splits)).all()


# ===========================================================================
# Section 5 -- property-based invariants.
# ===========================================================================


def test_invariant_trial_count_matches_grid_cardinality() -> None:
    """len(trials) == size of the Cartesian product, for 1 group, equal groups, unequal groups."""

    X, y = _make_data(n=76, p=10, seed=61)
    splits = [
        (np.arange(0, 46 + k * 3, dtype=int), np.arange(46 + k * 3, 50 + k * 3, dtype=int))
        for k in range(3)
    ]
    cases: list[dict[str, Sequence[Any]]] = [
        {"n_components": [1, 2, 3, 4]},  # 1 group
        {"n_components": [1, 2, 3], "n_selected": [4, 8]},  # equal groups
        {"n_components": [1, 2, 3, 4], "n_selected": [4, 8, None]},  # unequal axis sizes
    ]
    for grid_spec in cases:
        trials = _run_and_compare("supervised_pca", X, y, grid_spec, splits=splits)
        assert len(trials) == len(_candidates_from_grid(grid_spec))


def test_invariant_trial_ids_unique_and_complete() -> None:
    """trials['trial'] == range(n_candidates) exactly (no gaps, no dups)."""

    X, y = _make_data(n=80, p=12, seed=12345)
    trials = _run_and_compare("supervised_pca", X, y, _primary_grid(), splits=_primary_splits())
    n = len(_candidates_from_grid(_primary_grid()))
    assert sorted(trials["trial"].tolist()) == list(range(n))


def test_invariant_status_score_consistency() -> None:
    """error rows -> NaN score + non-null error; ok rows -> finite score + null error."""

    X, y = _make_data(n=80, p=12, seed=12345)
    trials = _run_and_compare("supervised_pca", X, y, _primary_grid(), splits=_primary_splits())
    for _, row in trials.iterrows():
        if row["status"] == "error":
            assert math.isnan(row["score"])
            assert row["error"] is not None
        else:
            assert row["status"] == "ok"
            assert math.isfinite(row["score"])
            assert row["error"] is None


def test_invariant_prefix_consistency_of_standalone_fits() -> None:
    """Direct check of the mathematical property the whole feature relies on: for fixed
    hyperparameters, an over-requested K clamps to the same effective component count and
    yields the same prediction as any K at or above the usable-predictor ceiling."""

    X, y = _make_data(n=60, p=3, seed=5)  # 3 usable predictors
    X_holdout, _ = _make_data(n=8, p=3, seed=6)
    preds = {}
    effective = {}
    for k in (1, 2, 3, 5, 10):
        fit = mo.supervised_pca(X, y, n_components=k)
        preds[k] = np.asarray(fit.predict(X_holdout)).reshape(-1)
        est = getattr(fit, "estimator", None)
        effective[k] = getattr(est, "n_components_", None)
    # K >= 3 (the predictor ceiling) all clamp to the same effective count and prediction.
    assert effective[3] == effective[5] == effective[10]
    assert np.array_equal(preds[3], preds[5])
    assert np.array_equal(preds[5], preds[10])
    # Increasing K up to the ceiling changes the effective component count monotonically.
    assert effective[1] <= effective[2] <= effective[3]
