from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _xy(n: int = 36) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    index = pd.period_range("2000-01", periods=n, freq="M").to_timestamp()
    X = pd.DataFrame(
        rng.normal(size=(n, 5)),
        index=index,
        columns=[f"x{i}" for i in range(5)],
    )
    y = (0.8 * X["x0"] - 0.4 * X["x1"] + rng.normal(scale=0.2, size=n)).rename("y")
    return X, y


def test_model_ensemble_specs_are_separate_from_models() -> None:
    model_names = set(mf.models.MODEL_SPECS)
    ensemble_names = set(mf.model_ensemble.MODEL_ENSEMBLE_SPECS)

    assert {"bagging", "booging"}.issubset(ensemble_names)
    assert {"subagging", "random_subspace", "stacking", "super_learner"}.issubset(
        ensemble_names
    )
    assert not ensemble_names & model_names

    bagging_spec = mf.model_ensemble.get_model_ensemble("bagging")
    booging_spec = mf.model_ensemble.get_model_ensemble("booging")
    stacking_spec = mf.model_ensemble.get_model_ensemble("stacking")

    assert bagging_spec.family == "model_ensemble"
    assert (
        bagging_spec.backend
        == "internal member resampling + sklearn-compatible base estimators"
    )
    assert (
        booging_spec.backend
        == "internal augmentation/bagging + sklearn.ensemble.GradientBoostingRegressor"
    )
    assert "OOF stacking" in stacking_spec.backend


def test_model_ensemble_base_estimator_helper_documents_choices() -> None:
    table = mf.model_ensemble.list_model_ensemble_bases()

    assert {"name", "backend", "notes"}.issubset(table.columns)
    assert {"ridge", "lasso", "svr"}.issubset(set(table["name"]))
    assert mf.model_ensemble.MODEL_ENSEMBLE_BASE_ESTIMATORS["ridge"].endswith("Ridge")


def test_model_ensemble_callables_fit_and_predict() -> None:
    X, y = _xy()

    fits = [
        mf.model_ensemble.bagging(X, y, base="ridge", n_estimators=4, random_state=0),
        mf.model_ensemble.subagging(X, y, base="ridge", n_estimators=4, random_state=0),
        mf.model_ensemble.random_subspace(
            X, y, base="ridge", n_estimators=4, max_features=0.6, random_state=0
        ),
        mf.model_ensemble.stacking(
            X,
            y,
            models=("ridge", "lasso"),
            meta_model="ridge",
            n_splits=3,
            random_state=0,
            model_params={"lasso": {"alpha": 0.01}},
        ),
        mf.model_ensemble.super_learner(
            X,
            y,
            models=("ridge", "lasso"),
            n_splits=3,
            weight_method="nnls",
            random_state=0,
            model_params={"lasso": {"alpha": 0.01}},
        ),
    ]

    for fit in fits:
        pred = fit.predict(X.iloc[:4])
        assert len(pred) == 4
        assert np.isfinite(pred).all()

    assert fits[1].model == "subagging"
    assert fits[1].metadata["replace"] is False
    assert (
        fits[1]
        .diagnostics["model_ensemble"]["member_samples"]["replace"]
        .eq(False)
        .all()
    )

    weights = fits[-1].estimator.weights_
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert "model_ensemble" in fits[-1].diagnostics
    assert "weights" in fits[-1].diagnostics["model_ensemble"]
    assert "oof_risk" in fits[-1].diagnostics["model_ensemble"]
    assert "folds" in fits[-1].diagnostics["model_ensemble"]
    json.dumps(fits[-1].to_dict())


def test_model_ensemble_member_diagnostics_are_exposed() -> None:
    X, y = _xy()

    bag = mf.model_ensemble.bagging(
        X, y, base="ridge", n_estimators=8, max_samples=0.5, random_state=0
    )
    subspace = mf.model_ensemble.random_subspace(
        X, y, base="ridge", n_estimators=4, max_features=0.5, random_state=0
    )

    bag_diag = bag.diagnostics["model_ensemble"]
    assert bag_diag["n_members"] == 8
    assert "oob_metrics" in bag_diag
    assert bag_diag["oob_metrics"]["n"] > 0
    assert {"n_rows", "n_unique_rows", "n_oob_rows"}.issubset(
        bag_diag["member_samples"].columns
    )

    subspace_diag = subspace.diagnostics["model_ensemble"]
    assert subspace_diag["n_members"] == 4
    assert "member_features" in subspace_diag
    assert len(subspace.estimator.predict_quantiles(X.iloc[:3], levels=(0.1, 0.9))) == 2


def test_stacking_and_super_learner_reject_duplicate_model_names() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="unique"):
        mf.model_ensemble.stacking(X, y, models=("ridge", "ridge"))
    with pytest.raises(ValueError, match="unique"):
        mf.model_ensemble.super_learner(X, y, models=("ridge", "ridge"))


def test_bagging_and_booging_are_callable_with_small_budgets() -> None:
    X, y = _xy()

    bag = mf.model_ensemble.bagging(
        X, y, base="ridge", n_estimators=4, max_samples=0.7, random_state=0
    )
    boo = mf.model_ensemble.booging(
        X,
        y,
        B=3,
        inner_n_estimators=5,
        sample_frac=0.8,
        random_state=0,
    )

    assert len(bag.predict(X.iloc[:3])) == 3
    assert len(boo.predict(X.iloc[:3])) == 3
    assert {"member_features", "member_samples"}.issubset(
        boo.diagnostics["model_ensemble"]
    )
    assert set(boo.estimator.predict_quantiles(X.iloc[:3], levels=(0.25, 0.75))) == {
        0.25,
        0.75,
    }


def test_bagging_supports_member_level_feature_perturbation() -> None:
    X, y = _xy()

    fit = mf.model_ensemble.bagging(
        X,
        y,
        base="decision_tree",
        n_estimators=4,
        max_samples=0.8,
        max_features="sqrt",
        random_state=0,
    )

    member_features = fit.diagnostics["model_ensemble"]["member_features"]
    assert member_features["n_features"].eq(2).all()
    assert fit.metadata["max_features"] == "sqrt"
    assert np.isfinite(fit.predict(X.iloc[:3])).all()


def test_booging_accepts_r_style_aliases_and_records_augmentation() -> None:
    X, y = _xy()
    X = X.copy()
    X["binary_signal"] = (X["x0"] > 0).astype(float)

    fit = mf.model_ensemble.booging(
        X,
        y,
        B=3,
        sampling_rate=0.6,
        mtry=0.5,
        data_aug=True,
        noise_level=0.3,
        shuffle_rate=0.25,
        n_trees=5,
        tree_depth=2,
        nu=0.2,
        bf=0.5,
        random_state=11,
    )

    diag = fit.diagnostics["model_ensemble"]
    augmentation = diag["augmentation_summary"]
    assert fit.metadata["sampling_rate"] == pytest.approx(0.6)
    assert fit.metadata["n_trees"] == 5
    assert fit.metadata["nu"] == pytest.approx(0.2)
    assert augmentation["data_aug"] is True
    assert augmentation["n_augmented_features"] == X.shape[1] * 3
    assert augmentation["n_binary_features"] == 1
    assert any(
        name.startswith("fake1_") for name in fit.estimator.augmented_feature_names_
    )
    assert diag["member_features"]["mtry"].eq(0.5).all()
    assert np.isfinite(fit.predict(X.iloc[:4])).all()


def test_bagging_validates_strategy_base_params_and_quantile_levels() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="strategy"):
        mf.model_ensemble.bagging(X, y, strategy="stationary")

    fit = mf.model_ensemble.bagging(
        X,
        y,
        base="lasso",
        base_params={"alpha": 0.01, "max_iter": 10},
        n_estimators=2,
    )

    assert len(fit.predict(X.iloc[:2])) == 2
    with pytest.raises(ValueError, match="quantile levels"):
        fit.estimator.predict_quantiles(X.iloc[:2], levels=(1.0,))


def test_forecasting_and_model_selection_resolve_model_ensemble_names() -> None:
    X, y = _xy()

    search = mf.model_selection.search_spec("subagging", preset="small")
    assert search.metadata["model_family"] == "model_ensemble"

    result = mf.forecasting.run(
        pd.concat([y, X], axis=1),
        model="subagging",
        target="y",
        horizon=1,
        window=mf.window.last_block(validation_size=4),
        features=mf.feature_engineering.feature_spec(target="y", horizon=1, lags=1),
        params={"subagging": {"n_estimators": 3, "base": "ridge"}},
        save_models=False,
    )

    table = result.forecasts
    assert set(table["model"]) == {"subagging"}
    assert table["model_spec"].eq("subagging").all()


def test_forecasting_runner_combines_model_ensemble_aliases() -> None:
    # ``run`` is atomic: each model (a plain model and a model-ensemble alias) is
    # fitted in its own single-model run; the two forecast tables are then combined
    # with the cross-model combination primitive.
    from macroforecast.forecasting.combination import (
        apply_combinations,
        resolve_combinations,
    )

    X, y = _xy()
    panel = pd.concat([y, X], axis=1)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x0", "x1", "x2"],
        lags=(0, 1),
    )

    linear = mf.forecasting.run(
        panel,
        "ridge",
        target="y",
        horizon=1,
        window=mf.window.last_block(validation_size=6),
        features=features,
        params={"alpha": 0.1},
        model_selection=None,
        save_models=False,
    )
    bagged = mf.forecasting.run(
        panel,
        "bagging",
        target="y",
        horizon=1,
        window=mf.window.last_block(validation_size=6),
        features=features,
        params={"base": "ridge", "n_estimators": 3, "random_state": 0},
        model_selection=mf.model_selection.fixed({"n_estimators": 2, "base": "ridge"}),
        save_models=False,
    )

    linear_table = linear.forecasts.assign(model="linear")
    bagged_table = bagged.forecasts.assign(model="bagged")
    master = pd.concat([linear_table, bagged_table], ignore_index=True)

    assert set(linear.forecasts["model_spec"]) == {"ridge"}
    assert set(bagged.forecasts["model_spec"]) == {"bagging"}

    records = apply_combinations(
        master,
        resolve_combinations(
            {"linear_plus_bagged": {"method": "mean", "models": ["linear", "bagged"]}}
        ),
    )
    combined = pd.DataFrame(records)
    combined_indexed = combined.set_index(["date", "origin_pos", "horizon"])

    assert not combined.empty
    assert combined["model_spec"].eq("forecast_combination").all()
    assert combined["combined"].all()
    assert {tuple(item["models"]) for item in combined["combination"].dropna()} == {
        ("linear", "bagged")
    }
    # FIX1: explicit Arm/model params pin matching SearchSpec keys. Here every
    # requested candidate key is pinned by explicit params, so no selection runs
    # and the explicit n_estimators=3 value wins over the old fixed-search value.
    assert bagged.forecasts["model_selection"].isna().all()
    bagged_params = bagged.forecasts["params"].dropna().iloc[0]
    assert bagged_params["base"] == "ridge"
    assert bagged_params["n_estimators"] == 3
    assert bagged_params["random_state"] == 0
    base = master
    for key, group in base.groupby(["date", "origin_pos", "horizon"], sort=False):
        assert np.isclose(
            combined_indexed.loc[key, "prediction"],
            group["prediction"].mean(),
        )


def test_model_ensemble_public_signatures_match_specs() -> None:
    for name, spec in mf.model_ensemble.MODEL_ENSEMBLE_SPECS.items():
        signature = inspect.signature(spec.fit_func)
        public_params = {
            parameter
            for parameter, value in signature.parameters.items()
            if parameter not in {"X", "y"}
            and value.kind
            in {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        }
        assert set(spec.default_params).issubset(public_params), name
