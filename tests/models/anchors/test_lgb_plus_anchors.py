"""Independent correctness anchors for ``lgb_plus`` / ``lgba_plus``.

WP-V2. Per ``.dev-notes/anchor_coverage/matrix.csv``, both models' only
anchor was an internal bookkeeping identity ("prediction_total ==
init+tree+linear"), which cannot catch an algorithmically misaligned
selection rule. No executable parity vs the named public reference
("philgoucou/lgbplus") exists in-repo. This file clean-rooms the ENSEMBLE
LOGIC itself (not the LightGBM tree-fitting internals, which are a real,
separately-tested upstream library) by hand-rolling an equivalent
step-by-step loop directly against ``lightgbm``, reading the documented
algorithm from ``macroforecast/models/tree.py``'s own docstrings/comments
(competition selection for LGB+, alternating tree/linear cycles for
LGB^A+) rather than importing or copying its private helper functions.

1. ``LGBPlusRegressor`` (``lgb_plus``): at every step, fits ONE LightGBM
   residual tree (``num_boost_round=1``, the estimator's own
   ``learning_rate`` applied externally, LightGBM's internal
   ``learning_rate`` pinned to 1.0) AND one greedy univariate no-intercept
   linear residual update, then keeps whichever has lower loss on the
   sampled rows (``selection_method="training"`` -- the simplest of the
   three branches, requiring no held-out split). With ``n_ensemble=1`` and
   a fixed seed, this is reproduced by an independently-written loop that
   issues the identical sequence of ``np.random.default_rng(seed)`` calls
   (subsample draw, then linear-candidate-feature draw, per step -- the
   exact call ORDER matters for bit-identical random streams) and an
   identical sequence of ``lgb.train(..., num_boost_round=1)`` calls.

2. ``LGBAPlusRegressor`` (``lgba_plus``): with ``subsample=1.0`` (the
   default), this scheme is actually fully DETERMINISTIC given a training
   set and seed -- every cycle uses ALL rows, and picks the best-correlated
   feature by a full (not randomly-subsampled) scan. Hand-rolled
   independently below.

3. A determinism pin for both (two fits, identical seed and data ->
   bit-identical predictions) -- guards the "same seed" half of the mission
   ask independently of whether the clean-room match above is exact.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.tree import LGBAPlusRegressor, LGBPlusRegressor

lgb = pytest.importorskip("lightgbm")


def _fixture_data(n_obs: int = 60, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, n_features))
    true_beta = np.array([1.2, -0.5, 0.0, 0.3])
    y = x @ true_beta + 0.5 * np.sin(x[:, 0] * 2.0) + 0.2 * rng.normal(size=n_obs)
    return x, y


# ---------------------------------------------------------------------------
# Anchor 1: LGBPlusRegressor competition loop, hand-rolled.
# ---------------------------------------------------------------------------


def _hand_rolled_lgb_plus(x, y, *, n_steps, learning_rate, num_leaves, min_data_in_leaf,
                           lambda_l2, subsample, linear_candidate_fraction, seed):
    n_obs, n_features = x.shape
    x_std = (x - x.mean(axis=0, keepdims=True)) / np.where(
        x.std(axis=0, keepdims=True) < 1e-10, 1.0, x.std(axis=0, keepdims=True)
    )
    train_pool_idx = np.arange(n_obs)
    subsample_size = max(1, int(np.floor(n_obs * subsample)))
    n_candidates = max(1, int(np.floor(n_features * linear_candidate_fraction)))
    params = {
        "objective": "regression",
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l2": lambda_l2,
        "learning_rate": 1.0,
        "verbosity": -1,
        "force_col_wise": True,
        "num_threads": 1,
        "seed": seed,
    }
    rng = np.random.default_rng(seed)
    pred = np.full(n_obs, float(np.mean(y)), dtype=float)
    chosen_types = []

    for _ in range(n_steps):
        sample_idx = rng.choice(train_pool_idx, size=subsample_size, replace=False)
        x_sample, y_sample = x[sample_idx], y[sample_idx]
        resid_sample = y_sample - pred[sample_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtrain = lgb.Dataset(x_sample, label=resid_sample, free_raw_data=False)
            booster = lgb.train(params, dtrain, num_boost_round=1)
        tree_update_all = np.asarray(booster.predict(x), dtype=float)
        tree_candidate = pred + learning_rate * tree_update_all

        candidate_idx = rng.choice(n_features, size=n_candidates, replace=False)
        resid_std = resid_sample - np.mean(resid_sample)
        resid_scale = float(np.std(resid_sample)) or 1.0
        if resid_scale < 1e-10:
            resid_scale = 1.0
        resid_standardized = resid_std / resid_scale
        correlations = (x_std[sample_idx][:, candidate_idx].T @ resid_standardized) / len(sample_idx)
        correlations = np.nan_to_num(correlations, nan=0.0)
        feature_idx = int(candidate_idx[int(np.argmax(np.abs(correlations)))])
        x_feature = x_sample[:, feature_idx]
        coef = float((x_feature @ resid_sample) / ((x_feature**2).sum() + 1e-10))
        linear_candidate = pred + learning_rate * coef * x[:, feature_idx]

        tree_loss = float(np.mean((y_sample - tree_candidate[sample_idx]) ** 2))
        linear_loss = float(np.mean((y_sample - linear_candidate[sample_idx]) ** 2))
        if tree_loss <= linear_loss:
            pred = tree_candidate
            chosen_types.append("tree")
        else:
            pred = linear_candidate
            chosen_types.append("linear")

    return pred, chosen_types


@pytest.mark.reference
def test_lgb_plus_competition_loop_matches_hand_rolled_reference():
    x, y = _fixture_data()
    kwargs = dict(
        n_steps=6,
        learning_rate=0.05,
        num_leaves=5,
        min_data_in_leaf=5,
        lambda_l2=0.1,
        subsample=0.7,
        linear_candidate_fraction=0.5,
        seed=42,
    )
    ref_pred, ref_types = _hand_rolled_lgb_plus(x, y, **kwargs)

    est = LGBPlusRegressor(
        n_ensemble=1,
        n_steps=kwargs["n_steps"],
        learning_rate=kwargs["learning_rate"],
        subsample=kwargs["subsample"],
        num_leaves=kwargs["num_leaves"],
        min_data_in_leaf=kwargs["min_data_in_leaf"],
        lambda_l2=kwargs["lambda_l2"],
        linear_candidate_fraction=kwargs["linear_candidate_fraction"],
        selection_method="training",
        random_state=kwargs["seed"],
    )
    est.fit(pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])]), pd.Series(y))
    impl_pred = est.predict(pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])]))
    impl_types = [step["type"] for step in est.ensemble_[0]["steps"]]

    assert impl_types == ref_types
    np.testing.assert_allclose(impl_pred, ref_pred, rtol=1e-10, atol=1e-10)


@pytest.mark.reference
def test_lgb_plus_determinism_pin():
    x, y = _fixture_data()
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    series = pd.Series(y)

    def _fit_predict():
        est = LGBPlusRegressor(
            n_ensemble=2, n_steps=8, subsample=0.7, selection_method="training",
            random_state=7,
        )
        est.fit(frame, series)
        return est.predict(frame)

    pred_a = _fit_predict()
    pred_b = _fit_predict()
    np.testing.assert_array_equal(pred_a, pred_b)


# ---------------------------------------------------------------------------
# Anchor 2: LGBAPlusRegressor alternating cycle, hand-rolled (deterministic).
# ---------------------------------------------------------------------------


def _hand_rolled_lgba_plus(x, y, *, n_cycles, trees_per_cycle, lr_tree, lr_linear,
                            num_leaves, min_data_in_leaf, seed):
    n_obs, n_features = x.shape
    params = {
        "objective": "regression",
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "learning_rate": 1.0,
        "verbosity": -1,
        "force_col_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_seed": seed,
    }
    pred = np.full(n_obs, float(np.mean(y)), dtype=float)
    linear_features = []

    for _ in range(n_cycles):
        resid = y - pred
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtrain = lgb.Dataset(x, label=resid, free_raw_data=False)
            booster = lgb.train(params, dtrain, num_boost_round=trees_per_cycle)
        tree_update = np.asarray(booster.predict(x), dtype=float)
        pred = pred + lr_tree * tree_update

        resid = y - pred
        correlations = np.zeros(n_features, dtype=float)
        for feature_idx in range(n_features):
            xj = x[:, feature_idx]
            if np.std(xj) < 1e-10 or np.std(resid) < 1e-10:
                correlations[feature_idx] = 0.0
            else:
                correlations[feature_idx] = float(np.corrcoef(xj, resid)[0, 1])
        correlations = np.nan_to_num(correlations, nan=0.0)
        best_idx = int(np.argmax(np.abs(correlations)))
        x_best = x[:, best_idx]
        x_centered = x_best - np.mean(x_best)
        resid_centered = resid - np.mean(resid)
        denom = float(np.sum(x_centered**2))
        if denom > 1e-10:
            coef = float((x_centered @ resid_centered) / denom)
            intercept = float(np.mean(resid) - coef * np.mean(x_best))
        else:
            coef = 0.0
            intercept = float(np.mean(resid))
        linear_update = coef * x_best + intercept
        pred = pred + lr_linear * linear_update
        linear_features.append(best_idx)

    return pred, linear_features


@pytest.mark.reference
def test_lgba_plus_alternating_cycle_matches_hand_rolled_reference():
    x, y = _fixture_data()
    kwargs = dict(
        n_cycles=5, trees_per_cycle=4, lr_tree=0.02, lr_linear=0.1,
        num_leaves=15, min_data_in_leaf=5, seed=11,
    )
    ref_pred, ref_features = _hand_rolled_lgba_plus(x, y, **kwargs)

    est = LGBAPlusRegressor(
        n_runs=1,
        n_cycles=kwargs["n_cycles"],
        trees_per_cycle=kwargs["trees_per_cycle"],
        lr_tree=kwargs["lr_tree"],
        lr_linear=kwargs["lr_linear"],
        num_leaves=kwargs["num_leaves"],
        min_data_in_leaf=kwargs["min_data_in_leaf"],
        subsample=1.0,
        random_state=kwargs["seed"],
    )
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    est.fit(frame, pd.Series(y))
    impl_pred = est.predict(frame)
    impl_features = [step["feature_idx"] for step in est.runs_[0]["linear_steps"]]

    assert impl_features == ref_features
    np.testing.assert_allclose(impl_pred, ref_pred, rtol=1e-10, atol=1e-10)


@pytest.mark.reference
def test_lgba_plus_determinism_pin():
    x, y = _fixture_data()
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    series = pd.Series(y)

    def _fit_predict():
        est = LGBAPlusRegressor(n_runs=1, n_cycles=6, trees_per_cycle=3, random_state=3)
        est.fit(frame, series)
        return est.predict(frame)

    pred_a = _fit_predict()
    pred_b = _fit_predict()
    np.testing.assert_array_equal(pred_a, pred_b)


@pytest.mark.reference
def test_lgb_plus_public_entrypoint_smoke_with_training_selection():
    """Sanity: the public mf.models.lgb_plus entry point still works end to
    end with the selection_method used by the clean-room anchor above."""
    x, y = _fixture_data()
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    fit = mf.models.lgb_plus(
        frame, pd.Series(y), n_ensemble=1, n_steps=5, selection_method="training",
        random_state=1,
    )
    pred = fit.predict(frame)
    assert np.all(np.isfinite(pred))
