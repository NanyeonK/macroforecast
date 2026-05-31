from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    x1 = np.linspace(0.0, 1.0, n)
    x2 = np.sin(np.arange(n) / 3.0)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(1.0 + 2.0 * x1 - 0.5 * x2, index=idx, name="y")
    return X, y


def _matlab_style_spca_reference(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    n_components: int,
    n_selected: int,
    slope_scale: bool,
) -> np.ndarray:
    x_train = X_train.to_numpy(dtype=float)
    x_test = X_test.to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)
    if slope_scale:
        slopes = _reference_marginal_slopes(x_train, y_values)
        x_train = x_train * slopes
        x_test = x_test * slopes

    control_train = np.ones((len(x_train), 1), dtype=float)
    control_test = np.ones((len(x_test), 1), dtype=float)
    control_coef = np.linalg.pinv(control_train) @ y_values
    y_residual = y_values - control_train @ control_coef

    work_x = x_train.copy()
    work_y = y_residual.copy()
    loadings: list[np.ndarray] = []
    factor_coefs: list[float] = []
    for _ in range(n_components):
        scores = _reference_abs_corr(work_x, work_y)
        selected = np.argsort(-scores)[:n_selected]
        _, _, vt = np.linalg.svd(work_x[:, selected], full_matrices=False)
        loading = np.zeros(work_x.shape[1], dtype=float)
        loading[selected] = vt[0]
        factor = work_x @ loading
        denom = float(factor @ factor)
        alpha = float(work_y @ factor / denom)
        lambdas = work_x.T @ factor / denom
        loadings.append(loading)
        factor_coefs.append(alpha)
        work_y = work_y - alpha * factor
        work_x = work_x - np.outer(factor, lambdas)

    loading_matrix = np.vstack(loadings)
    return (
        x_test @ loading_matrix.T @ np.asarray(factor_coefs)
        + control_test @ control_coef
    )


def _reference_marginal_slopes(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = X - X.mean(axis=0)
    centered_y = y - y.mean()
    return (centered_x * centered_y[:, None]).sum(axis=0) / (
        centered_x * centered_x
    ).sum(axis=0)


def _reference_abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = X - X.mean(axis=0)
    centered_y = y - y.mean()
    denom = np.sqrt((centered_x * centered_x).sum(axis=0)) * np.sqrt(
        float(centered_y @ centered_y)
    )
    out = np.zeros(X.shape[1], dtype=float)
    np.divide(
        (centered_x * centered_y[:, None]).sum(axis=0),
        denom,
        out=out,
        where=denom > 1e-12,
    )
    return np.abs(out)


def _huang_scaled_pca_reference(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = X_train.mean(axis=0)
    x_scale = X_train.std(axis=0, ddof=1).where(lambda s: s > 1e-12, 1.0)
    x_train = ((X_train - x_mean) / x_scale).to_numpy(dtype=float)
    x_test = ((X_test - x_mean) / x_scale).to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)
    slopes = _reference_marginal_slopes(x_train, y_values)
    scaled_train = x_train * slopes
    scaled_test = x_test * slopes

    u, _, _ = np.linalg.svd(scaled_train @ scaled_train.T, full_matrices=False)
    factors = u[:, :n_components] * np.sqrt(float(len(x_train)))
    loadings = scaled_train.T @ factors / float(len(x_train))
    projection = loadings @ np.linalg.pinv(loadings.T @ loadings)
    factor_test = scaled_test @ projection

    control_train = np.ones((len(x_train), 1), dtype=float)
    control_test = np.ones((len(x_test), 1), dtype=float)
    control_coef = np.linalg.pinv(control_train) @ y_values
    residual = y_values - control_train @ control_coef
    factor_coef = np.linalg.pinv(factors) @ residual
    return factors, control_test @ control_coef + factor_test @ factor_coef


def test_linear_models_return_model_fit_with_series_predictions() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.5)

    pred = fit.predict(X.iloc[:5])

    assert isinstance(fit, mf.models.ModelFit)
    assert list(pred.index) == list(X.index[:5])
    assert pred.name == "prediction"
    assert fit.metadata["alpha"] == 0.5


def test_ridge_variants_fit_and_record_constraints() -> None:
    X, y = _xy()

    nonneg = mf.models.nonneg_ridge(X, y, alpha=0.1)
    shrink = mf.models.shrink_to_target_ridge(
        X,
        y,
        alpha=0.1,
        prior_target={"x1": 0.7, "x2": 0.3},
        simplex=True,
        nonneg=True,
    )
    fused = mf.models.fused_difference_ridge(
        X,
        y,
        alpha=0.1,
        difference_order=1,
        nonneg=True,
    )
    random_walk = mf.models.random_walk_ridge(X, y, alpha=0.5, initial_alpha=0.1)

    assert (nonneg.diagnostics["coefficients"] >= -1e-10).all()
    assert abs(float(shrink.diagnostics["coefficients"].sum()) - 1.0) < 1e-6
    assert (fused.diagnostics["coefficients"] >= -1e-10).all()
    assert random_walk.estimator.coef_path_.shape == X.shape
    for fit in (nonneg, shrink, fused, random_walk):
        pred = fit.predict(X.iloc[:3])
        assert pred.shape == (3,)
        assert fit.metadata["alpha"] > 0.0


def test_lasso_path_is_not_public_model_family() -> None:
    assert not hasattr(mf.models, "lasso_path")
    assert "lasso_path" not in mf.models.__all__


def test_top_level_model_exports_include_new_model_families() -> None:
    assert mf.adaptive_lasso is mf.models.adaptive_lasso
    assert mf.adaptive_elastic_net is mf.models.adaptive_elastic_net
    assert mf.nonneg_ridge is mf.models.nonneg_ridge
    assert mf.shrink_to_target_ridge is mf.models.shrink_to_target_ridge
    assert mf.fused_difference_ridge is mf.models.fused_difference_ridge
    assert mf.random_walk_ridge is mf.models.random_walk_ridge
    assert mf.group_lasso is mf.models.group_lasso
    assert mf.sparse_group_lasso is mf.models.sparse_group_lasso
    assert mf.svr is mf.models.svr
    assert mf.linear_svr is mf.models.linear_svr
    assert mf.nu_svr is mf.models.nu_svr
    assert mf.nn is mf.models.nn
    assert mf.lstm is mf.models.lstm
    assert mf.gru is mf.models.gru
    assert mf.transformer is mf.models.transformer
    assert mf.hemisphere_nn is mf.models.hemisphere_nn
    assert mf.kernel_ridge is mf.models.kernel_ridge
    assert mf.knn is mf.models.knn
    assert mf.mars is mf.models.mars
    assert mf.bvar_minnesota is mf.models.bvar_minnesota
    assert mf.dfm_mixed_mariano_murasawa is mf.models.dfm_mixed_mariano_murasawa
    assert mf.dfm_unrestricted_midas is mf.models.dfm_unrestricted_midas
    assert mf.midas_almon is mf.models.midas_almon
    assert mf.unrestricted_midas is mf.models.unrestricted_midas
    assert not hasattr(mf.models, "mlp")
    assert not hasattr(mf, "mlp")


def test_model_specs_own_default_params_and_search_spaces() -> None:
    spec = mf.models.get_model("lasso", preset="small")

    assert isinstance(spec, mf.models.ModelSpec)
    assert spec.default_params["alpha"] == 1.0
    assert spec.search_space() == {"alpha": (0.01, 0.1, 1.0)}

    table = mf.models.describe_model(spec)
    assert set(table["parameter"]) == {"alpha", "max_iter"}
    assert table.loc[table["parameter"] == "alpha", "small_space"].iloc[0] == (
        0.01,
        0.1,
        1.0,
    )


def test_model_spec_parameter_metadata_is_consistent() -> None:
    for name, spec in mf.models.MODEL_SPECS.items():
        parameter_names = {parameter.name for parameter in spec.parameters}
        tunable_names = {
            parameter.name for parameter in spec.parameters if parameter.tunable
        }
        search_names = (
            set().union(*(space.keys() for space in spec.search_spaces.values()))
            if spec.search_spaces
            else set()
        )

        assert set(spec.default_params).issubset(parameter_names), name
        assert search_names.issubset(parameter_names), name
        if spec.search_spaces:
            assert tunable_names.issubset(search_names), name


def test_model_spec_can_fit_like_model_callable() -> None:
    X, y = _xy()
    spec = mf.models.get_model("ridge", params={"alpha": 0.5})

    fit = spec(X, y)

    assert isinstance(fit, mf.models.ModelFit)
    assert fit.metadata["alpha"] == 0.5


def test_kernel_knn_and_mars_models_fit_and_predict() -> None:
    X, y = _xy()
    nonlinear_y = pd.Series(
        1.0 + 3.0 * np.maximum(0.0, X["x1"] - 0.45) - 0.25 * X["x2"],
        index=X.index,
        name="y",
    )

    krr = mf.models.kernel_ridge(X, y, alpha=0.1, kernel="rbf")
    knn = mf.models.knn(X, y, n_neighbors=3)
    mars = mf.models.mars(X, nonlinear_y, max_terms=8, n_knots=6)

    assert krr.predict(X.iloc[:4]).shape == (4,)
    assert knn.predict(X.iloc[:4]).shape == (4,)
    assert mars.predict(X.iloc[:4]).shape == (4,)
    assert mars.estimator.n_terms_ > 1
    assert "Package-native" in mars.metadata["implementation_note"]


def test_bvar_and_target_timeseries_models_fit_and_predict() -> None:
    X, y = _xy(80)
    panel = pd.concat([y.rename("y"), X], axis=1)
    future = pd.DataFrame(index=pd.date_range(panel.index[-1] + pd.offsets.MonthEnd(), periods=3, freq="ME"))

    bvar = mf.models.bvar_minnesota(panel, target="y", n_lag=2)
    niw = mf.models.bvar_normal_inverse_wishart(panel, target="y", n_lag=2)
    ets = mf.models.ets(y, trend="add")
    hw = mf.models.holt_winters(y, trend="add")
    theta = mf.models.theta_method(y, deseasonalize=False)

    for fit in (bvar, niw, ets, hw, theta):
        pred = fit.predict(future)
        assert len(pred) == 3
        assert np.isfinite(pred).all()


def test_midas_variants_group_lagged_predictors() -> None:
    X, y = _xy(70)
    lagged = pd.DataFrame(
        {
            "x_lag0": X["x1"],
            "x_lag1": X["x1"].shift(1),
            "x_lag2": X["x1"].shift(2),
            "z_lag0": X["x2"],
            "z_lag1": X["x2"].shift(1),
        },
        index=X.index,
    )
    target = y.shift(-1)

    almon = mf.models.midas_almon(lagged, target, alpha=0.1)
    beta = mf.models.midas_beta(lagged, target, beta_params=(1.0, 2.0), alpha=0.1)
    step = mf.models.midas_step(lagged, target, n_steps=2, alpha=0.1)
    unrestricted = mf.models.unrestricted_midas(lagged, target, alpha=0.1)

    assert set(almon.estimator.groups_) == {"x", "z"}
    assert set(beta.estimator.weights_) == {"x", "z"}
    assert step.predict(lagged.dropna().iloc[:5]).shape == (5,)
    assert set(unrestricted.estimator.groups_) == {"x", "z"}
    assert "x_lag2" in unrestricted.metadata["lag_groups"]["x"]
    assert almon.metadata["lag_group_details"]["x"][2] == {
        "column": "x_lag2",
        "lag": 2,
    }
    assert almon.metadata["weighted_columns"] == ["x_midas", "z_midas"]
    assert almon.diagnostics["coefficients"].index.tolist() == ["x_midas", "z_midas"]
    assert set(almon.diagnostics["effective_lag_coefficients"].index) >= {
        "x_lag0",
        "x_lag1",
        "x_lag2",
    }


def test_midas_weighted_design_requires_complete_lag_blocks() -> None:
    X, y = _xy(20)
    lagged = pd.DataFrame(
        {
            "x_lag0": X["x1"],
            "x_lag1": X["x1"].shift(1),
            "x_lag2": X["x1"].shift(2),
        },
        index=X.index,
    )
    fit = mf.models.midas_almon(lagged, y, polynomial_order=0)

    with pytest.raises(ValueError, match="Input X contains NaN"):
        fit.predict(lagged.iloc[:2])


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (mf.models.midas_almon, {"polynomial_order": -1}, "polynomial_order"),
        (mf.models.midas_almon, {"polynomial_order": 2, "theta": (0.0, 0.1)}, "theta"),
        (mf.models.midas_beta, {"beta_params": (1.0, 0.0)}, "beta_params"),
        (mf.models.midas_step, {"n_steps": 0}, "n_steps"),
        (mf.models.unrestricted_midas, {"alpha": -0.1}, "alpha"),
    ],
)
def test_midas_variants_validate_parameters(factory, kwargs, message) -> None:
    X, y = _xy(20)
    lagged = pd.DataFrame({"x_lag0": X["x1"], "x_lag1": X["x1"].shift(1)}, index=X.index)

    with pytest.raises(ValueError, match=message):
        factory(lagged, y, **kwargs)


def test_mixed_frequency_dfm_uses_native_frequency_metadata() -> None:
    idx = pd.date_range("2000-01-01", periods=48, freq="MS")
    factor = np.sin(np.arange(48) / 5.0)
    monthly = mf.data.DataBundle(
        mf.data.as_panel(
            pd.DataFrame(
                {
                    "date": idx,
                    "m1": factor + 0.05 * np.cos(np.arange(48)),
                    "m2": 0.5 * factor + 0.03 * np.sin(np.arange(48)),
                }
            ),
            date="date",
            metadata={"dataset": "monthly", "frequency": "monthly"},
        ),
        {"dataset": "monthly", "frequency": "monthly"},
    )
    quarterly_idx = idx[2::3]
    quarterly = mf.data.DataBundle(
        mf.data.as_panel(
            pd.DataFrame(
                {
                    "date": quarterly_idx,
                    "q_target": [factor[max(0, i - 2) : i + 1].mean() for i in range(2, 48, 3)],
                }
            ),
            date="date",
            metadata={"dataset": "quarterly", "frequency": "quarterly"},
        ),
        {"dataset": "quarterly", "frequency": "quarterly"},
    )
    mixed = mf.data.combine(monthly, quarterly, dataset="mixed", frequency="native")

    fit = mf.models.dfm_mixed_mariano_murasawa(
        mixed,
        target="q_target",
        n_factors=1,
        factor_order=1,
        maxiter=20,
        tolerance=1e-3,
    )
    future = pd.DataFrame(index=pd.date_range(idx[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS"))

    pred = fit.predict(future)

    assert len(pred) == 3
    assert np.isfinite(pred.to_numpy(dtype=float)).all()
    assert fit.metadata["monthly_columns"] == ["m1", "m2"]
    assert fit.metadata["quarterly_columns"] == ["q_target"]
    assert "Mariano-Murasawa" in fit.metadata["implementation_note"]
    assert "factors_filtered" in fit.diagnostics


def test_dfm_unrestricted_midas_builds_composite_design() -> None:
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    factor = np.sin(np.arange(60) / 5.0)
    monthly = mf.data.DataBundle(
        mf.data.as_panel(
            pd.DataFrame(
                {
                    "date": idx,
                    "m1": factor + 0.05 * np.cos(np.arange(60)),
                    "m2": 0.5 * factor + 0.03 * np.sin(np.arange(60)),
                }
            ),
            date="date",
            metadata={"dataset": "monthly", "frequency": "monthly"},
        ),
        {"dataset": "monthly", "frequency": "monthly"},
    )
    quarterly_idx = idx[2::3]
    quarterly = mf.data.DataBundle(
        mf.data.as_panel(
            pd.DataFrame(
                {
                    "date": quarterly_idx,
                    "q_target": [factor[max(0, i - 2) : i + 1].mean() for i in range(2, 60, 3)],
                }
            ),
            date="date",
            metadata={"dataset": "quarterly", "frequency": "quarterly"},
        ),
        {"dataset": "quarterly", "frequency": "quarterly"},
    )
    mixed = mf.data.combine(monthly, quarterly, dataset="mixed", frequency="native")

    fit = mf.models.dfm_unrestricted_midas(
        mixed,
        target="q_target",
        lag_columns=["m1"],
        lags=(0, 1, 2),
        factor_lags=(0,),
        maxiter=20,
        tolerance=1e-3,
        alpha=0.1,
    )
    design = fit.estimator.design_

    assert isinstance(design, pd.DataFrame)
    assert "dfm_factor1_lag0" in design.columns
    assert "m1_lag2" in design.columns
    assert fit.metadata["prediction_contract"].startswith("predict() expects")
    pred = fit.predict(design.iloc[-2:])
    assert len(pred) == 2
    assert np.isfinite(pred.to_numpy(dtype=float)).all()


def test_model_fit_and_spec_export_metadata() -> None:
    X, y = _xy()
    spec = mf.models.get_model("ridge", preset="small", params={"alpha": 0.5})
    fit = spec(X, y)

    fit_dict = fit.to_dict()
    spec_dict = spec.to_dict()
    spec_metadata = spec.to_metadata()

    assert fit_dict["model"] == "ridge"
    assert fit_dict["feature_names"] == ["x1", "x2"]
    assert fit.to_metadata()["fit"]["n_features"] == 2
    assert set(fit.diagnostics) >= {
        "coefficients",
        "fitted_values",
        "metrics",
        "residuals",
    }
    assert fit.diagnostics["residuals"].index.equals(X.index)
    assert "rmse" in fit_dict["diagnostics"]["metrics"]
    assert spec_dict["parameters"][0]["name"] == "alpha"
    assert spec_metadata["model"] == "ridge"
    assert spec_metadata["search_space"]["alpha"] == [0.01, 0.1, 1.0]
    json.dumps({"fit": fit_dict, "spec": spec_dict, "metadata": spec_metadata})


def test_save_fit_and_load_fit_persist_model_sidecar(tmp_path) -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.5)

    saved = mf.models.save_fit(
        fit,
        tmp_path / "ridge" / "fit.pkl",
        metadata={"alias": "ridge", "params": {"alpha": 0.5}},
    )

    assert saved.model_path is not None
    assert saved.save_error is None
    loaded = mf.models.load_fit(saved.model_path)
    assert isinstance(loaded, mf.models.ModelFit)
    assert loaded.model == "ridge"

    metadata = json.loads(Path(saved.metadata_path).read_text(encoding="utf-8"))
    assert metadata["alias"] == "ridge"
    assert metadata["params"]["alpha"] == 0.5
    assert metadata["fit"]["fit"]["diagnostics"]["metrics"]["n"] == len(X)


def test_save_fit_can_write_metadata_when_pickle_fails(tmp_path) -> None:
    class LocalFit:
        pass

    saved = mf.models.save_fit(LocalFit(), tmp_path / "local.pkl")

    assert saved.model_path is None
    assert saved.save_error is not None
    metadata = json.loads(Path(saved.metadata_path).read_text(encoding="utf-8"))
    assert metadata["model_path"] is None
    assert metadata["save_error"] == saved.save_error


def test_model_specs_record_backend_extra_and_scaling_metadata() -> None:
    svr_spec = mf.models.get_model("svr")
    nn_spec = mf.models.get_model("nn")
    lstm_spec = mf.models.get_model("lstm")
    transformer_spec = mf.models.get_model("transformer")
    hnn_spec = mf.models.get_model("hemisphere_nn")
    xgb_spec = mf.models.get_model("xgboost")
    lightgbm_spec = mf.models.get_model("lightgbm")
    catboost_spec = mf.models.get_model("catboost")
    mrf_spec = mf.models.get_model("macro_random_forest")
    garch_spec = mf.models.get_model("garch11")
    egarch_spec = mf.models.get_model("egarch")
    realized_garch_spec = mf.models.get_model("realized_garch")

    assert svr_spec.backend == "sklearn.svm.SVR"
    assert svr_spec.requires_scaling is True
    assert svr_spec.to_metadata()["requires_scaling"] is True
    assert "standardize predictors" in svr_spec.recommended_preprocessing[0]

    assert nn_spec.backend == "torch.nn.Sequential"
    assert nn_spec.requires_extra == "deep"
    assert nn_spec.requires_scaling is False

    assert lstm_spec.backend == "torch.nn.LSTM"
    assert lstm_spec.requires_extra == "deep"
    assert lstm_spec.to_dict()["recommended_preprocessing"] == [
        "handled internally: X and y are standardized inside each fit"
    ]
    assert transformer_spec.backend == "torch.nn.TransformerEncoder"
    assert transformer_spec.requires_extra == "deep"
    assert hnn_spec.requires_extra == "deep"
    assert hnn_spec.default_params["B"] is None

    table = mf.models.list_model_specs(family="neural")
    row = table.loc[table["name"] == "lstm"].iloc[0]
    assert row["requires_extra"] == "deep"
    assert not bool(row["requires_scaling"])

    assert xgb_spec.backend == "xgboost.XGBRegressor"
    assert xgb_spec.requires_extra == "xgboost"
    assert lightgbm_spec.backend == "lightgbm.LGBMRegressor"
    assert lightgbm_spec.requires_extra == "lightgbm"
    assert catboost_spec.backend == "catboost.CatBoostRegressor"
    assert catboost_spec.requires_extra == "catboost"
    assert (
        mrf_spec.backend
        == "macroforecast.models._mrf_reference.MacroRandomForest"
    )
    assert mrf_spec.requires_extra == "macro_random_forest"
    assert garch_spec.backend == "arch.arch_model"
    assert garch_spec.requires_extra == "arch"
    assert egarch_spec.backend == "arch.arch_model"
    assert egarch_spec.requires_extra == "arch"
    assert realized_garch_spec.backend == "internal"
    assert realized_garch_spec.requires_extra is None


def test_pls_and_far_fit() -> None:
    X, y = _xy()

    pls_fit = mf.models.pls(X, y, n_components=1)
    far_fit = mf.models.far(X, y, n_factors=1, n_lag=1)

    assert len(pls_fit.predict(X.iloc[-3:])) == 3
    assert pls_fit.metadata["n_components"] == 1
    assert len(far_fit.predict(X.iloc[-3:])) == 3


def test_support_vector_models_fit() -> None:
    X, y = _xy()

    rbf = mf.models.svr(X, y, C=1.0, epsilon=0.01)
    linear = mf.models.linear_svr(X, y, C=1.0, epsilon=0.0, max_iter=5000)
    nu = mf.models.nu_svr(X, y, C=1.0, nu=0.5)

    assert len(rbf.predict(X.iloc[-3:])) == 3
    assert len(linear.predict(X.iloc[-3:])) == 3
    assert len(nu.predict(X.iloc[-3:])) == 3
    assert rbf.metadata["kernel"] == "rbf"
    assert linear.metadata["loss"] == "epsilon_insensitive"
    assert nu.metadata["nu"] == 0.5


def test_adaptive_and_group_penalized_models_fit() -> None:
    X, y = _xy()
    groups = ("trend", "cycle")

    adaptive = mf.models.adaptive_lasso(
        X, y, alpha=0.001, gamma=1.0, random_state=0
    )
    adaptive_enet = mf.models.adaptive_elastic_net(
        X, y, alpha=0.001, l1_ratio=0.5, gamma=1.0, random_state=0
    )
    grouped = mf.models.group_lasso(X, y, groups=groups, alpha=0.001)
    sparse_grouped = mf.models.sparse_group_lasso(
        X, y, groups=groups, alpha=0.001, l1_ratio=0.25
    )

    for fit in (adaptive, adaptive_enet, grouped, sparse_grouped):
        assert len(fit.predict(X.iloc[-3:])) == 3
        assert "coefficients" in fit.diagnostics
        assert np.isfinite(fit.diagnostics["coefficients"].to_numpy(dtype=float)).all()

    assert adaptive.metadata["gamma"] == 1.0
    assert adaptive.estimator.adaptive_weights_.shape == (X.shape[1],)
    assert grouped.estimator.groups_ == groups
    assert sparse_grouped.metadata["l1_ratio"] == 0.25


def test_group_lasso_validates_group_length() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="one entry per X column"):
        mf.models.group_lasso(X, y, groups=("only_one",))


def test_tree_model_records_feature_importance_diagnostics() -> None:
    X, y = _xy()

    fit = mf.models.random_forest(X, y, n_estimators=10, random_state=0, n_jobs=1)

    importance = fit.diagnostics["feature_importance"]
    assert list(importance.index) == ["x1", "x2"] or set(importance.index) == {
        "x1",
        "x2",
    }
    assert np.isfinite(importance.to_numpy(dtype=float)).all()


def test_composite_models_record_factor_diagnostics() -> None:
    X, y = _xy()

    scaled = mf.models.scaled_pca(X, y, n_components=2)
    supervised = mf.models.supervised_pca(X, y, n_components=1, n_selected=1)

    assert "factor_loadings" in scaled.diagnostics
    assert scaled.diagnostics["factor_loadings"].shape[1] == 2
    assert "factor_loadings" in supervised.diagnostics
    assert "component_selected_features" in supervised.diagnostics


def test_save_fit_roundtrip_for_core_model_families(tmp_path) -> None:
    X, y = _xy()
    fits = [
        mf.models.ridge(X, y, alpha=0.5),
        mf.models.random_forest(X, y, n_estimators=5, random_state=0, n_jobs=1),
        mf.models.scaled_pca(X, y, n_components=1),
        mf.models.ar(y, n_lag=2),
    ]

    for fit in fits:
        saved = mf.models.save_fit(fit, tmp_path / fit.model / "fit.pkl")
        assert saved.model_path is not None, fit.model
        loaded = mf.models.load_fit(saved.model_path)
        assert loaded.model == fit.model
        assert len(loaded.predict(X.iloc[-2:])) == 2


@pytest.mark.parametrize(
    ("module", "fit_call"),
    [
        (
            "xgboost",
            lambda X, y: mf.models.xgboost(
                X, y, n_estimators=3, max_depth=2, random_state=0
            ),
        ),
        (
            "lightgbm",
            lambda X, y: mf.models.lightgbm(
                X, y, n_estimators=3, num_leaves=7, random_state=0, verbose=-1
            ),
        ),
        (
            "catboost",
            lambda X, y: mf.models.catboost(
                X, y, n_estimators=3, max_depth=2, random_state=0, verbose=False
            ),
        ),
        (
            "torch",
            lambda X, y: mf.models.nn(
                X,
                y,
                hidden_layer_sizes=(4,),
                max_epochs=1,
                batch_size=8,
                device="cpu",
            ),
        ),
    ],
)
def test_optional_non_volatility_backends_fit_predict_and_save_when_installed(
    tmp_path,
    module: str,
    fit_call,
) -> None:
    pytest.importorskip(module)
    X, y = _xy(36)

    fit = fit_call(X, y)
    pred = fit.predict(X.iloc[-3:])
    saved = mf.models.save_fit(fit, tmp_path / fit.model / "fit.pkl")

    assert len(pred) == 3
    assert np.isfinite(pred.to_numpy(dtype=float)).all()
    assert saved.metadata_path.endswith(".json")
    if saved.model_path is not None:
        loaded = mf.models.load_fit(saved.model_path)
        assert loaded.model == fit.model


def test_realized_garch_records_volatility_diagnostics() -> None:
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    returns = pd.Series(
        0.001 + 0.01 * np.sin(np.arange(40) / 5.0),
        index=idx,
        name="returns",
    )
    realized = (returns.abs() + 0.01) ** 2

    fit = mf.models.realized_garch(
        returns,
        rv=realized,
        max_iter=20,
        n_starts=1,
        random_state=0,
    )

    assert "params" in fit.diagnostics
    assert "conditional_volatility" in fit.diagnostics
    assert len(fit.diagnostics["conditional_volatility"]) == len(returns)
    assert len(fit.predict_variance(2)) == 2


def test_pls_default_clamps_components_to_available_predictors() -> None:
    X, y = _xy()

    fit = mf.models.pls(X, y)

    assert fit.metadata["requested_n_components"] == 3
    assert fit.metadata["resolved_n_components"] == 2
    assert fit.metadata["n_components"] == 2
    assert len(fit.predict(X.iloc[-3:])) == 3


def test_nn_model_fit() -> None:
    pytest.importorskip("torch")
    X, y = _xy()

    fit = mf.models.nn(
        X,
        y,
        hidden_layer_sizes=(8,),
        max_epochs=1,
        batch_size=8,
        random_state=0,
        device="cpu",
    )
    pred = fit.predict(X.iloc[-4:])

    assert len(pred) == 4
    assert np.isfinite(pred).all()
    assert fit.metadata["hidden_layer_sizes"] == (8,)
    assert fit.metadata["device"] == "cpu"


def test_recurrent_neural_models_fit_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    X, y = _xy(24)

    lstm = mf.models.lstm(
        X,
        y,
        sequence_length=3,
        hidden_size=4,
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )
    gru = mf.models.gru(
        X,
        y,
        sequence_length=3,
        hidden_size=4,
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )
    transformer = mf.models.transformer(
        X,
        y,
        sequence_length=3,
        hidden_size=8,
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )

    assert len(lstm.predict(X.iloc[-2:])) == 2
    assert len(gru.predict(X.iloc[-2:])) == 2
    assert len(transformer.predict(X.iloc[-2:])) == 2
    assert lstm.estimator.device_ == "cpu"
    assert gru.estimator.device_ == "cpu"
    assert transformer.estimator.device_ == "cpu"
    assert lstm.diagnostics["sequence_context"]["sequence_length"] == 3
    assert lstm.diagnostics["sequence_context"]["train_tail_rows"] == 2
    assert (
        lstm.diagnostics["sequence_context"]["test_sequence_prefix"]
        == "last fitted rows only"
    )


def test_hemisphere_nn_fit_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    X, y = _xy(24)

    fit = mf.models.hemisphere_nn(
        X,
        y,
        neurons=4,
        max_epochs=1,
        n_estimators=2,
        patience=1,
        device="cpu",
        quantile_levels=(0.1, 0.5, 0.9),
    )
    pred = fit.predict(X.iloc[-3:])
    variance = fit.predict_variance(X.iloc[-3:])
    quantiles = fit.predict_quantiles(X.iloc[-3:])

    assert len(pred) == 3
    assert variance.shape == (3,)
    assert np.isfinite(variance).all()
    assert (variance > 0).all()
    assert set(quantiles) == {0.1, 0.5, 0.9}
    assert all(values.shape == (3,) for values in quantiles.values())
    assert fit.metadata["n_estimators"] == 2
    assert fit.metadata["quantile_levels"] == (0.1, 0.5, 0.9)
    assert fit.estimator.device_ == "cpu"


def test_supervised_pca_fit_records_selected_features() -> None:
    X, y = _xy()

    fit = mf.models.supervised_pca(
        X,
        y,
        n_components=1,
        n_selected=1,
        min_abs_corr=0.0,
    )
    pred = fit.predict(X.iloc[-3:])

    assert len(pred) == 3
    assert fit.metadata["n_components"] == 1
    assert fit.estimator.selected_features_
    assert set(fit.estimator.selected_features_).issubset(set(X.columns))
    assert fit.estimator.component_selected_features_


def test_scaled_pca_fit_records_huang_state() -> None:
    X, y = _xy()

    fit = mf.models.scaled_pca(X, y, n_components=2)
    pred = fit.predict(X.iloc[-3:])
    scores = fit.estimator.factor_scores_

    assert len(pred) == 3
    assert fit.metadata["source"].startswith("Huang et al.")
    assert fit.estimator.scaling_slopes_ is not None
    assert fit.estimator.n_components_ == 2
    assert scores is not None
    np.testing.assert_allclose(scores.T @ scores / len(scores), np.eye(2), atol=1e-10)


def test_supervised_scaled_pca_fit_records_slope_scaling() -> None:
    X, y = _xy()

    fit = mf.models.supervised_scaled_pca(
        X,
        y,
        n_components=2,
        n_selected=2,
        control_columns=["x1"],
        preselect="hard_tstat",
        t_threshold=0.0,
    )
    pred = fit.predict(X.iloc[-3:])

    assert len(pred) == 3
    assert fit.metadata["source"].startswith("Hounyo and Li")
    assert fit.estimator.scaling_slopes_ is not None
    assert fit.estimator.n_components_ >= 1
    assert fit.estimator.control_names_ == ("x1", "const")
    assert "x1" not in fit.estimator.factor_features_


def test_scaled_pca_matches_huang_spcaest_factor_extraction() -> None:
    X, y = _xy(42)
    X_train = X.iloc[:34]
    y_train = y.iloc[:34]
    X_test = X.iloc[34:38]

    fit = mf.models.scaled_pca(X_train, y_train, n_components=2)
    expected_factors, expected_pred = _huang_scaled_pca_reference(
        X_train,
        y_train,
        X_test,
        n_components=2,
    )

    assert fit.estimator.factor_scores_ is not None
    correlations = np.corrcoef(fit.estimator.factor_scores_.T, expected_factors.T)[
        :2, 2:
    ]
    np.testing.assert_allclose(np.abs(np.diag(correlations)), np.ones(2), atol=1e-10)
    np.testing.assert_allclose(
        fit.predict(X_test).to_numpy(), expected_pred, atol=1e-10
    )


def test_supervised_pca_matches_matlab_style_spca_recursion() -> None:
    X, y = _xy(42)
    X_train = X.iloc[:34]
    y_train = y.iloc[:34]
    X_test = X.iloc[34:38]

    fit = mf.models.supervised_pca(
        X_train,
        y_train,
        n_components=2,
        n_selected=1,
        scale=False,
        include_constant=True,
    )
    expected = _matlab_style_spca_reference(
        X_train,
        y_train,
        X_test,
        n_components=2,
        n_selected=1,
        slope_scale=False,
    )

    np.testing.assert_allclose(fit.predict(X_test).to_numpy(), expected, atol=1e-10)


def test_supervised_scaled_pca_matches_matlab_style_sspca_recursion() -> None:
    X, y = _xy(42)
    X_train = X.iloc[:34]
    y_train = y.iloc[:34]
    X_test = X.iloc[34:38]

    fit = mf.models.supervised_scaled_pca(
        X_train,
        y_train,
        n_components=2,
        n_selected=2,
        scale=False,
        include_constant=True,
    )
    expected = _matlab_style_spca_reference(
        X_train,
        y_train,
        X_test,
        n_components=2,
        n_selected=2,
        slope_scale=True,
    )

    np.testing.assert_allclose(fit.predict(X_test).to_numpy(), expected, atol=1e-10)


def test_tree_models_include_custom_macro_callables() -> None:
    X, y = _xy()

    sgt = mf.models.slow_growing_tree(X, y, max_depth=2, min_leaf_size=3)
    qrf = mf.models.quantile_regression_forest(
        X,
        y,
        n_estimators=8,
        min_samples_leaf=2,
        random_state=0,
    )

    assert len(sgt.predict(X.iloc[:4])) == 4
    quantiles = qrf.estimator.predict_quantiles(X.iloc[:4], levels=(0.1, 0.9))
    assert set(quantiles) == {0.1, 0.9}
    assert quantiles[0.1].shape == (4,)


def test_bagging_and_booging_are_callable_with_small_budgets() -> None:
    X, y = _xy()

    bag = mf.models.bagging(
        X, y, base="ridge", n_estimators=4, max_samples=0.7, random_state=0
    )
    boo = mf.models.booging(
        X,
        y,
        B=3,
        inner_n_estimators=5,
        sample_frac=0.8,
        random_state=0,
    )

    assert len(bag.predict(X.iloc[:3])) == 3
    assert len(boo.predict(X.iloc[:3])) == 3


@pytest.mark.parametrize(
    ("module", "extra", "fit_call"),
    [
        ("xgboost", "xgboost", lambda X, y: mf.models.xgboost(X, y)),
        ("lightgbm", "lightgbm", lambda X, y: mf.models.lightgbm(X, y)),
        ("catboost", "catboost", lambda X, y: mf.models.catboost(X, y)),
        ("arch", "arch", lambda X, y: mf.models.garch11(y)),
        (
            "matplotlib",
            "macro_random_forest",
            lambda X, y: mf.models.macro_random_forest(X, y),
        ),
        ("torch", "deep", lambda X, y: mf.models.nn(X, y, max_epochs=1)),
        ("torch", "deep", lambda X, y: mf.models.lstm(X, y, max_epochs=1)),
        ("torch", "deep", lambda X, y: mf.models.gru(X, y, max_epochs=1)),
        ("torch", "deep", lambda X, y: mf.models.transformer(X, y, max_epochs=1)),
        (
            "torch",
            "deep",
            lambda X, y: mf.models.hemisphere_nn(X, y, max_epochs=1, n_estimators=1),
        ),
    ],
)
def test_optional_external_models_fail_lazily_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    extra: str,
    fit_call,
) -> None:
    X, y = _xy()

    def fail_import(name: str, package: str | None = None):  # noqa: ARG001
        if name == module:
            raise ImportError("blocked")
        return __import__(name)

    monkeypatch.setattr("macroforecast.models.utils.import_module", fail_import)

    with pytest.raises(ImportError, match=f"macroforecast\\[{extra}\\]"):
        fit_call(X, y)


def test_pcr_is_not_public_model_family() -> None:
    assert not hasattr(mf.models, "pcr")
    assert "pcr" not in mf.models.__all__
    assert "pcr" not in mf.models.MODEL_SPECS


def test_mars_is_public_package_native_model_family() -> None:
    assert mf.mars is mf.models.mars
    assert "mars" in mf.models.__all__
    assert "mars" in mf.models.MODEL_SPECS


def test_macro_random_forest_adapter_wires_reference_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _xy(36)
    calls = {}

    class FakeMRF:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def _ensemble_loop(self):
            calls["n_ensemble_loop"] = calls.get("n_ensemble_loop", 0) + 1
            n = len(calls["oos_pos"])
            return {
                "pred_ensemble": pd.DataFrame(
                    {"Ensembled_Prediction": np.arange(n, dtype=float)}
                )
            }

    monkeypatch.setattr(
        "macroforecast.models.tree.MacroRandomForestRegressor._import_external",
        staticmethod(lambda: FakeMRF),
    )

    fit = mf.models.macro_random_forest(
        X.iloc[:30],
        y.iloc[:30],
        x_columns=["x1"],
        S_columns=["x1", "x2"],
        B=2,
        print_b=False,
        parallelise=False,
    )
    pred = fit.predict(X.iloc[30:33])
    pred_again = fit.predict(X.iloc[30:33])

    assert list(pred) == [0.0, 1.0, 2.0]
    assert list(pred_again) == [0.0, 1.0, 2.0]
    assert calls["n_ensemble_loop"] == 1
    assert calls["y_pos"] == 0
    assert calls["x_pos"].tolist() == [1]
    assert calls["S_pos"].tolist() == [1, 2]
    assert fit.metadata["B"] == 2
    assert fit.metadata["x_columns"] == ("x1",)


def test_macro_random_forest_vendored_backend_smoke() -> None:
    pytest.importorskip("joblib")
    pytest.importorskip("matplotlib")
    X, y = _xy(48)

    fit = mf.models.macro_random_forest(
        X.iloc[:36],
        y.iloc[:36],
        B=1,
        minsize=10,
        mtry_frac=1.0,
        S_columns=["x1", "x2"],
        parallelise=False,
        print_b=False,
    )
    pred = fit.predict(X.iloc[36:38])

    assert len(pred) == 2
    assert np.isfinite(pred).all()
    assert fit.estimator.output_ is not None
    assert "pred" in fit.estimator.output_


def test_model_metadata_records_all_fit_params() -> None:
    X, y = _xy()

    rf = mf.models.random_forest(
        X,
        y,
        n_estimators=7,
        max_depth=2,
        min_samples_leaf=3,
        bootstrap=False,
        n_jobs=1,
    )
    gb = mf.models.gradient_boosting(X, y, n_estimators=7, loss="absolute_error")

    assert rf.metadata["n_estimators"] == 7
    assert rf.metadata["min_samples_leaf"] == 3
    assert rf.metadata["bootstrap"] is False
    assert gb.metadata["loss"] == "absolute_error"


def test_tree_ensemble_public_signatures_match_specs() -> None:
    for name in (
        "slow_growing_tree",
        "quantile_regression_forest",
        "bagging",
        "booging",
        "macro_random_forest",
    ):
        spec = mf.models.get_model(name)
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


def test_ar_and_var_fit() -> None:
    X, y = _xy()
    panel = pd.concat([y, X], axis=1)

    ar_fit = mf.models.ar(y, n_lag=2)
    var_fit = mf.models.var(panel, target="y", n_lag=1)

    assert len(ar_fit.predict(pd.DataFrame(index=y.index[-4:]))) == 4
    assert len(var_fit.predict(panel.iloc[-4:])) == 4
