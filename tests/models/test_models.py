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
    quadratic_factors: bool = False,
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
    factor_square_coefs: list[float] = []
    for _ in range(n_components):
        scores = _reference_abs_corr(work_x, work_y)
        selected = np.argsort(-scores)[:n_selected]
        _, _, vt = np.linalg.svd(work_x[:, selected], full_matrices=False)
        loading = np.zeros(work_x.shape[1], dtype=float)
        loading[selected] = vt[0]
        factor = work_x @ loading
        denom = float(factor @ factor)
        alpha = float(work_y @ factor / denom)
        squared = factor**2
        denom2 = float(squared @ squared)
        alpha2 = 0.0 if not quadratic_factors else float(work_y @ squared / denom2)
        lambdas = work_x.T @ factor / denom
        loadings.append(loading)
        factor_coefs.append(alpha)
        factor_square_coefs.append(alpha2)
        work_y = work_y - alpha * factor - alpha2 * squared
        work_x = work_x - np.outer(factor, lambdas)

    loading_matrix = np.vstack(loadings)
    factors = x_test @ loading_matrix.T
    prediction = factors @ np.asarray(factor_coefs) + control_test @ control_coef
    if quadratic_factors:
        prediction = prediction + (factors**2) @ np.asarray(factor_square_coefs)
    return prediction


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
    quadratic_factors: bool = False,
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
    prediction = control_test @ control_coef + factor_test @ factor_coef
    if quadratic_factors:
        factor_square_coef = np.linalg.pinv(factors**2) @ residual
        prediction = prediction + (factor_test**2) @ factor_square_coef
    return factors, prediction


def test_linear_models_return_model_fit_with_series_predictions() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.5)

    pred = fit.predict(X.iloc[:5])

    assert isinstance(fit, mf.models.ModelFit)
    assert list(pred.index) == list(X.index[:5])
    assert pred.name == "prediction"
    assert fit.metadata["alpha"] == 0.5


def test_basic_linear_models_fit_and_predict() -> None:
    X, y = _xy()

    fits = [
        mf.models.ols(X, y),
        mf.models.elastic_net(X, y, alpha=0.01, l1_ratio=0.5),
        mf.models.bayesian_ridge(X, y),
        mf.models.huber(X, y, max_iter=100),
        mf.models.glmboost(X, y, n_iter=5),
    ]

    for fit in fits:
        pred = fit.predict(X.iloc[:4])
        assert isinstance(fit, mf.models.ModelFit)
        assert len(pred) == 4
        assert np.isfinite(pred.to_numpy(dtype=float)).all()
        assert "coefficients" in fit.diagnostics


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


def test_nonneg_ridge_matches_augmented_nnls_objective() -> None:
    from scipy.optimize import nnls

    X, y = _xy()
    alpha = 0.1
    fit = mf.models.nonneg_ridge(X, y, alpha=alpha)
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_centered = x_values - x_values.mean(axis=0)
    y_centered = y_values - y_values.mean()
    x_aug = np.vstack([x_centered, np.sqrt(alpha) * np.eye(X.shape[1])])
    y_aug = np.concatenate([y_centered, np.zeros(X.shape[1])])
    expected_coef, _ = nnls(x_aug, y_aug)

    np.testing.assert_allclose(
        fit.estimator.coef_, expected_coef, rtol=1e-10, atol=1e-10
    )
    assert fit.estimator.intercept_ == pytest.approx(
        float(y_values.mean() - x_values.mean(axis=0) @ expected_coef)
    )


def test_shrink_to_target_ridge_matches_unconstrained_target_ridge_solution() -> None:
    X, y = _xy()
    alpha = 0.25
    prior = {"x1": 0.25, "x2": -0.1}

    fit = mf.models.shrink_to_target_ridge(X, y, alpha=alpha, prior_target=prior)
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_centered = x_values - x_values.mean(axis=0)
    y_centered = y_values - y_values.mean()
    prior_vector = np.asarray([prior["x1"], prior["x2"]], dtype=float)
    lhs = x_centered.T @ x_centered + alpha * np.eye(X.shape[1])
    rhs = x_centered.T @ y_centered + alpha * prior_vector
    expected_coef = np.linalg.solve(lhs, rhs)

    np.testing.assert_allclose(fit.estimator.coef_, expected_coef, rtol=1e-8, atol=1e-8)
    assert fit.estimator.intercept_ == pytest.approx(
        float(y_values.mean() - x_values.mean(axis=0) @ expected_coef)
    )


def test_fused_difference_ridge_matches_unconstrained_difference_penalty_solution() -> (
    None
):
    X, y = _xy()
    alpha = 0.4

    fit = mf.models.fused_difference_ridge(X, y, alpha=alpha, difference_order=1)
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_centered = x_values - x_values.mean(axis=0)
    y_centered = y_values - y_values.mean()
    difference = np.diff(np.eye(X.shape[1]), axis=0)
    lhs = x_centered.T @ x_centered + alpha * (difference.T @ difference)
    rhs = x_centered.T @ y_centered
    expected_coef = np.linalg.solve(lhs, rhs)

    np.testing.assert_allclose(fit.estimator.coef_, expected_coef, rtol=1e-8, atol=1e-8)
    assert fit.estimator.intercept_ == pytest.approx(
        float(y_values.mean() - x_values.mean(axis=0) @ expected_coef)
    )


def test_random_walk_ridge_matches_augmented_path_least_squares() -> None:
    X, y = _xy(12)
    alpha = 0.5
    initial_alpha = 0.2

    fit = mf.models.random_walk_ridge(
        X,
        y,
        alpha=alpha,
        initial_alpha=initial_alpha,
    )
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_centered = x_values - x_values.mean(axis=0)
    y_centered = y_values - y_values.mean()
    n_obs, n_features = x_centered.shape
    design = np.zeros((n_obs, n_obs * n_features), dtype=float)
    for row in range(n_obs):
        start = row * n_features
        design[row, start : start + n_features] = x_centered[row]
    initial_penalty = np.zeros((n_features, n_obs * n_features), dtype=float)
    initial_penalty[:, :n_features] = np.sqrt(initial_alpha) * np.eye(n_features)
    walk_penalty = np.zeros(((n_obs - 1) * n_features, n_obs * n_features), dtype=float)
    cursor = 0
    for row in range(1, n_obs):
        previous = (row - 1) * n_features
        current = row * n_features
        walk_penalty[cursor : cursor + n_features, previous : previous + n_features] = (
            -np.sqrt(alpha) * np.eye(n_features)
        )
        walk_penalty[cursor : cursor + n_features, current : current + n_features] = (
            np.sqrt(alpha) * np.eye(n_features)
        )
        cursor += n_features
    expected_vector = np.linalg.lstsq(
        np.vstack([design, initial_penalty, walk_penalty]),
        np.concatenate(
            [y_centered, np.zeros(n_features), np.zeros((n_obs - 1) * n_features)]
        ),
        rcond=None,
    )[0]
    expected_path = expected_vector.reshape(n_obs, n_features)

    np.testing.assert_allclose(
        fit.estimator.coef_path_.to_numpy(dtype=float),
        expected_path,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(fit.estimator.coef_, expected_path[-1])


def test_tvp_z_basis_matches_r_zfun_layout() -> None:
    from macroforecast.models.tvp import _tvp_z_basis

    data = np.asarray(
        [
            [2.0, 10.0],
            [3.0, 11.0],
            [4.0, 12.0],
        ]
    )
    got = _tvp_z_basis(data)
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0],
            [1.0, 0.0, 3.0, 0.0, 11.0, 0.0, 1.0, 3.0, 11.0],
            [1.0, 1.0, 4.0, 4.0, 12.0, 12.0, 1.0, 4.0, 12.0],
        ]
    )

    np.testing.assert_allclose(got, expected)


def test_tvp_ridge_fits_and_records_reference_outputs() -> None:
    X, y = _xy(40)

    fit = mf.models.tvp_ridge(
        X,
        y,
        lambda_candidates=(0.1, 1.0),
        kfold=3,
        cv_2srr=False,
        use_garch=False,
    )

    assert isinstance(fit, mf.models.ModelFit)
    assert fit.metadata["lambda2"] == 0.1
    assert fit.estimator.betas_rr_.shape == (1, X.shape[1] + 1, len(X))
    assert fit.estimator.betas_2srr_.shape == (1, X.shape[1] + 1, len(X))
    assert fit.estimator.yhat_rr_.shape == (len(X), 1)
    assert fit.estimator.yhat_2srr_.shape == (len(X), 1)
    assert fit.estimator.sig_eps_.shape == (len(X), 1)
    assert fit.estimator.coef_path_.shape == X.shape
    assert np.isfinite(fit.predict(X.iloc[-5:]).to_numpy(dtype=float)).all()
    assert fit.diagnostics["fitted_values"].shape == (len(X),)
    assert fit.diagnostics["coefficients"].shape == (X.shape[1],)


def test_tvp_ridge_fixed_lambda_matches_dual_grr_helper() -> None:
    from macroforecast.models.tvp import _dual_generalized_ridge, _tvp_z_basis

    X, y = _xy(18)
    fit = mf.models.tvp_ridge(
        X,
        y,
        lambda_candidates=(0.5,),
        cv_2srr=False,
        sig_u_param=0.0,
        sig_eps_param=0.0,
        use_garch=False,
    )
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float).reshape(-1, 1)
    x_sd = np.std(x_values, axis=0, ddof=1)
    y_sd = float(np.std(y_values[:, 0], ddof=1))
    z_basis = _tvp_z_basis(x_values / x_sd.reshape(1, -1))
    expected = _dual_generalized_ridge(
        z_basis,
        y_values / y_sd,
        dim_x=X.shape[1] + 1,
        lambda1=0.5,
        lambda2=0.1,
    )

    np.testing.assert_allclose(
        fit.estimator.yhat_rr_.iloc[:, 0].to_numpy(dtype=float),
        expected.yhat[:, 0] * y_sd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit.estimator.betas_rr_[0, 0, :],
        expected.betas_grr[0, 0, :] * y_sd,
        rtol=1e-8,
        atol=1e-8,
    )


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
    assert mf.tvp_ridge is mf.models.tvp_ridge
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
    assert mf.density_hnn is mf.models.density_hnn
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


def test_all_registered_models_are_public_exports() -> None:
    for name in mf.models.MODEL_SPECS:
        assert hasattr(mf.models, name), name
        assert name in mf.models.__all__, name
        assert hasattr(mf, name), name
        assert getattr(mf, name) is getattr(mf.models, name), name


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


def test_model_spec_defaults_match_callable_signatures() -> None:
    for name, spec in mf.models.MODEL_SPECS.items():
        signature = inspect.signature(spec.fit_func)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        for key, default in spec.default_params.items():
            assert key in parameters or accepts_kwargs, (name, key)
            if key in parameters and parameters[key].default is not inspect._empty:
                assert parameters[key].default == default, (name, key)

        for parameter in spec.parameters:
            assert parameter.name in parameters or accepts_kwargs, (
                name,
                parameter.name,
            )
            if (
                parameter.name in parameters
                and parameters[parameter.name].default is not inspect._empty
            ):
                assert parameters[parameter.name].default == parameter.default, (
                    name,
                    parameter.name,
                )

        parameter_names = {parameter.name for parameter in spec.parameters}
        for preset, space in spec.search_spaces.items():
            for key in space:
                assert key in parameters or accepts_kwargs, (name, preset, key)
                assert key in parameter_names, (name, preset, key)


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
    assert krr.metadata["kernel"] == "rbf"
    assert "sklearn.kernel_ridge.KernelRidge" in krr.metadata["implementation_note"]
    assert knn.metadata["n_neighbors"] == 3
    assert "small-window" in knn.metadata["implementation_note"]
    assert mars.predict(X.iloc[:4]).shape == (4,)
    assert mars.estimator.n_terms_ > 1
    assert "Package-native" in mars.metadata["implementation_note"]


def test_knn_resolves_neighbors_for_small_training_windows() -> None:
    X, y = _xy(3)

    fit = mf.models.knn(X, y, n_neighbors=10)

    assert fit.metadata["n_neighbors"] == 3
    assert fit.metadata["requested_n_neighbors"] == 10
    assert fit.predict(X).shape == (3,)


def test_bvar_and_target_timeseries_models_fit_and_predict() -> None:
    X, y = _xy(80)
    panel = pd.concat([y.rename("y"), X], axis=1)
    future = pd.DataFrame(
        index=pd.date_range(
            panel.index[-1] + pd.offsets.MonthEnd(), periods=3, freq="ME"
        )
    )

    bvar = mf.models.bvar_minnesota(
        panel, target="y", n_lag=2, iter=30, burnin=10, random_state=1
    )
    niw = mf.models.bvar_normal_inverse_wishart(
        panel, target="y", n_lag=2, iter=30, burnin=10, random_state=2
    )
    ets = mf.models.ets(y, trend="add")
    hw = mf.models.holt_winters(y, trend="add")
    theta = mf.models.theta_method(y, deseasonalize=False)

    for fit in (bvar, niw, ets, hw, theta):
        pred = fit.predict(future)
        assert len(pred) == 3
        assert np.isfinite(pred).all()
    assert "posterior draw sampler" in bvar.metadata["implementation_note"]
    assert "posterior draw sampler" in niw.metadata["implementation_note"]
    assert set(bvar.diagnostics) == {
        "coef_mean",
        "coef_se",
        "coef_mcse",
        "coef_q025",
        "coef_q975",
        "sigma_mean",
    }
    assert "sigma_mean" in niw.diagnostics


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
    assert "midasr::nealmon" in almon.metadata["implementation_note"]
    assert "midasr::nbetaMT" in beta.metadata["implementation_note"]
    assert "midasr::polystep" in step.metadata["implementation_note"]


def test_midas_weights_match_midasr_shape_formulas() -> None:
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    lagged = pd.DataFrame(
        {
            "x_lag0": np.arange(10, dtype=float),
            "x_lag1": np.arange(10, dtype=float) + 1.0,
            "x_lag2": np.arange(10, dtype=float) + 2.0,
            "x_lag3": np.arange(10, dtype=float) + 3.0,
        },
        index=idx,
    )
    target = pd.Series(np.arange(10, dtype=float), index=idx, name="y")

    almon = mf.models.midas_almon(
        lagged, target, polynomial_order=2, theta=(0.1, -0.02)
    )
    positions = np.arange(1, 5, dtype=float)
    raw = 0.1 * positions + -0.02 * positions**2
    expected_almon = np.exp(raw - raw.max())
    expected_almon = expected_almon / expected_almon.sum()

    beta = mf.models.midas_beta(lagged, target, beta_params=(1.5, 2.0))
    z = (np.arange(1, 5, dtype=float) - 1.0) / 3.0
    z[0] += np.finfo(float).eps
    z[-1] -= np.finfo(float).eps
    expected_beta = z**0.5 * (1.0 - z)
    expected_beta = expected_beta / expected_beta.sum()

    step = mf.models.midas_step(
        lagged, target, step_bounds=(2,), step_weights=(1.0, 3.0)
    )
    expected_step = np.asarray([1.0, 1.0, 3.0, 3.0], dtype=float)
    expected_step = expected_step / expected_step.sum()

    np.testing.assert_allclose(almon.metadata["weights"]["x"], expected_almon)
    np.testing.assert_allclose(beta.metadata["weights"]["x"], expected_beta)
    np.testing.assert_allclose(step.metadata["weights"]["x"], expected_step)


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
        (mf.models.midas_almon, {"polynomial_order": 2, "theta": (0.0,)}, "theta"),
        (mf.models.midas_beta, {"beta_params": (1.0, 0.0)}, "beta_params"),
        (mf.models.midas_step, {"n_steps": 0}, "n_steps"),
        (mf.models.midas_step, {"step_bounds": (2, 1)}, "step_bounds"),
        (mf.models.midas_step, {"step_weights": ()}, "step_weights"),
        (mf.models.unrestricted_midas, {"alpha": -0.1}, "alpha"),
    ],
)
def test_midas_variants_validate_parameters(factory, kwargs, message) -> None:
    X, y = _xy(20)
    lagged = pd.DataFrame(
        {"x_lag0": X["x1"], "x_lag1": X["x1"].shift(1)}, index=X.index
    )

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
                    "q_target": [
                        factor[max(0, i - 2) : i + 1].mean() for i in range(2, 48, 3)
                    ],
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
    future = pd.DataFrame(
        index=pd.date_range(idx[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS")
    )

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
                    "q_target": [
                        factor[max(0, i - 2) : i + 1].mean() for i in range(2, 60, 3)
                    ],
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
    assert fit.metadata["prediction_contract"].startswith("predict() accepts")
    pred = fit.predict(design.iloc[-2:])
    assert len(pred) == 2
    assert np.isfinite(pred.to_numpy(dtype=float)).all()
    masked = mixed.panel.copy()
    masked.loc[quarterly_idx[-1:], "q_target"] = np.nan
    future_pred = fit.estimator.predict_from_panel(
        mf.data.DataBundle(masked, mixed.metadata),
        anchor_dates=quarterly_idx[-1:],
    )
    assert len(future_pred) == 1
    assert np.isfinite(future_pred).all()


def test_restricted_midas_estimates_midasr_style_nls_weights() -> None:
    idx = pd.RangeIndex(18)
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        rng.normal(size=(len(idx), 4)),
        index=idx,
        columns=["x_lag0", "x_lag1", "x_lag2", "x_lag3"],
    )
    true_weights = np.array([0.35, 0.35, 0.15, 0.15])
    y = pd.Series(1.5 + X.to_numpy() @ true_weights, index=idx, name="y")

    fit = mf.models.restricted_midas(
        X,
        y,
        weighting="step",
        step_bounds=(2,),
        start_params={"x": (0.2, 0.2)},
        fit_intercept=True,
        tolerance=1e-10,
    )

    effective = fit.diagnostics["effective_lag_coefficients"]
    assert fit.metadata["converged"] is True
    assert np.isclose(float(fit.diagnostics["intercept"]), 1.5, atol=1e-6)
    assert np.allclose(
        effective.reindex(["x_lag0", "x_lag1"]).to_numpy(dtype=float),
        [0.35, 0.35],
        atol=1e-6,
    )
    assert np.allclose(
        effective.reindex(["x_lag2", "x_lag3"]).to_numpy(dtype=float),
        [0.15, 0.15],
        atol=1e-6,
    )
    assert "midasr::midas_r" in fit.metadata["implementation_note"]


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
    ols_spec = mf.models.get_model("ols")
    ridge_spec = mf.models.get_model("ridge")
    lasso_spec = mf.models.get_model("lasso")
    tvp_spec = mf.models.get_model("tvp_ridge")
    adaptive_lasso_spec = mf.models.get_model("adaptive_lasso")
    group_lasso_spec = mf.models.get_model("group_lasso")
    glmboost_spec = mf.models.get_model("glmboost")
    pls_spec = mf.models.get_model("pls")
    kernel_ridge_spec = mf.models.get_model("kernel_ridge")
    knn_spec = mf.models.get_model("knn")
    decision_tree_spec = mf.models.get_model("decision_tree")
    random_forest_spec = mf.models.get_model("random_forest")
    extra_trees_spec = mf.models.get_model("extra_trees")
    gradient_boosting_spec = mf.models.get_model("gradient_boosting")
    svr_spec = mf.models.get_model("svr")
    linear_svr_spec = mf.models.get_model("linear_svr")
    nu_svr_spec = mf.models.get_model("nu_svr")
    nn_spec = mf.models.get_model("nn")
    lstm_spec = mf.models.get_model("lstm")
    transformer_spec = mf.models.get_model("transformer")
    hnn_spec = mf.models.get_model("hemisphere_nn")
    density_hnn_spec = mf.models.get_model("density_hnn")
    xgb_spec = mf.models.get_model("xgboost")
    lightgbm_spec = mf.models.get_model("lightgbm")
    lgb_plus_spec = mf.models.get_model("lgb_plus")
    lgba_plus_spec = mf.models.get_model("lgba_plus")
    catboost_spec = mf.models.get_model("catboost")
    qrf_spec = mf.models.get_model("quantile_regression_forest")
    mrf_spec = mf.models.get_model("macro_random_forest")
    garch_spec = mf.models.get_model("garch11")
    egarch_spec = mf.models.get_model("egarch")
    realized_garch_spec = mf.models.get_model("realized_garch")

    assert ols_spec.backend == "sklearn.linear_model.LinearRegression"
    assert ridge_spec.backend == "sklearn.linear_model.Ridge"
    assert lasso_spec.backend == "sklearn.linear_model.Lasso"
    assert "TVPRidge" in tvp_spec.backend
    assert tvp_spec.default_params["random_state"] == 1071
    assert tvp_spec.default_params["cv_2srr"] is True
    assert (
        adaptive_lasso_spec.backend
        == "internal adaptive weights + sklearn.linear_model.Lasso"
    )
    assert group_lasso_spec.backend == "internal proximal-gradient solver"
    assert glmboost_spec.backend == "internal componentwise L2 boosting"
    assert glmboost_spec.default_params["center"] is True
    assert pls_spec.backend == "sklearn.cross_decomposition.PLSRegression"
    assert pls_spec.default_params["include_constant"] is True
    assert kernel_ridge_spec.backend == "sklearn.kernel_ridge.KernelRidge"
    assert kernel_ridge_spec.family == "nonparametric"
    assert kernel_ridge_spec.requires_scaling is True
    assert "nonlinear kernels" in kernel_ridge_spec.recommended_preprocessing[0]
    assert knn_spec.backend == "sklearn.neighbors.KNeighborsRegressor"
    assert knn_spec.requires_scaling is True
    assert "distance-based" in knn_spec.recommended_preprocessing[0]

    assert decision_tree_spec.backend == "sklearn.tree.DecisionTreeRegressor"
    assert random_forest_spec.backend == "sklearn.ensemble.RandomForestRegressor"
    assert extra_trees_spec.backend == "sklearn.ensemble.ExtraTreesRegressor"
    assert (
        gradient_boosting_spec.backend == "sklearn.ensemble.GradientBoostingRegressor"
    )

    assert svr_spec.backend == "sklearn.svm.SVR"
    assert svr_spec.requires_scaling is True
    assert svr_spec.to_metadata()["requires_scaling"] is True
    assert "standardize predictors" in svr_spec.recommended_preprocessing[0]
    assert linear_svr_spec.backend == "sklearn.svm.LinearSVR"
    assert linear_svr_spec.requires_scaling is True
    assert nu_svr_spec.backend == "sklearn.svm.NuSVR"
    assert nu_svr_spec.requires_scaling is True

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
    assert density_hnn_spec.backend == "torch-native Aionx DensityHNN port"
    assert density_hnn_spec.requires_extra == "deep"
    assert density_hnn_spec.default_params["prior_estimators"] == 50
    assert density_hnn_spec.default_params["rescale_volatility"] is True

    table = mf.models.list_model_specs(family="neural")
    row = table.loc[table["name"] == "lstm"].iloc[0]
    assert row["requires_extra"] == "deep"
    assert not bool(row["requires_scaling"])

    assert xgb_spec.backend == "xgboost.XGBRegressor"
    assert xgb_spec.requires_extra == "xgboost"
    assert lightgbm_spec.backend == "lightgbm.LGBMRegressor"
    assert lightgbm_spec.requires_extra == "lightgbm"
    assert (
        lgb_plus_spec.backend == "internal philgoucou/lgbplus-aligned + lightgbm.train"
    )
    assert lgb_plus_spec.requires_extra == "lightgbm"
    assert lgb_plus_spec.default_params["selection_method"] == "oob"
    assert (
        lgba_plus_spec.backend == "internal philgoucou/lgbplus-aligned + lightgbm.train"
    )
    assert lgba_plus_spec.requires_extra == "lightgbm"
    assert lgba_plus_spec.default_params["n_cycles"] == 25
    assert catboost_spec.backend == "catboost.CatBoostRegressor"
    assert catboost_spec.requires_extra == "catboost"
    assert (
        qrf_spec.backend
        == "sklearn.ensemble.RandomForestRegressor + internal leaf quantiles"
    )
    assert mrf_spec.backend == "macroforecast.models._mrf_reference.MacroRandomForest"
    assert mrf_spec.requires_extra == "macro_random_forest"
    assert garch_spec.backend == "arch.arch_model"
    assert garch_spec.requires_extra == "arch"
    assert egarch_spec.backend == "arch.arch_model"
    assert egarch_spec.requires_extra == "arch"
    assert (
        realized_garch_spec.backend
        == "internal rugarch-realGARCH-style p=q=1 log-linear MLE"
    )
    assert realized_garch_spec.requires_extra is None


def test_pls_and_far_fit() -> None:
    X, y = _xy()

    pls_fit = mf.models.pls(X, y, n_components=1)
    far_fit = mf.models.far(X, y, n_factors=1, n_lag=1)

    assert len(pls_fit.predict(X.iloc[-3:])) == 3
    assert pls_fit.metadata["n_components"] == 1
    assert len(far_fit.predict(X.iloc[-3:])) == 3


def test_favar_model_fits_and_predicts() -> None:
    X, y = _xy(80)

    fit = mf.models.favar(X, y, n_factors=2, n_lag=2, fctmethod="BGM", nburn=5, nrep=15)
    pred = fit.predict(X.iloc[-4:])

    assert isinstance(fit, mf.models.ModelFit)
    assert len(pred) == 4
    assert np.isfinite(pred.to_numpy(dtype=float)).all()
    assert fit.metadata["n_factors"] == 2
    assert fit.metadata["fctmethod"] == "BGM"
    assert fit.metadata["backend"] == "internal FAVAR::FAVAR-aligned Bayesian sampler"
    assert "predict.favar" in fit.metadata["implementation_note"]
    assert set(fit.diagnostics) >= {
        "factorx",
        "loading_mean",
        "var_coef_mean",
        "var_sigma_mean",
    }
    assert fit.feature_names == tuple(X.columns)


def test_support_vector_models_fit() -> None:
    X, y = _xy()

    rbf = mf.models.svr(X, y, C=1.0, epsilon=0.01)
    linear = mf.models.linear_svr(
        X, y, C=1.0, epsilon=0.0, max_iter=5000, random_state=None
    )
    nu = mf.models.nu_svr(X, y, C=1.0, nu=0.5)

    assert len(rbf.predict(X.iloc[-3:])) == 3
    assert len(linear.predict(X.iloc[-3:])) == 3
    assert len(nu.predict(X.iloc[-3:])) == 3
    assert rbf.metadata["kernel"] == "rbf"
    assert linear.metadata["loss"] == "epsilon_insensitive"
    assert linear.metadata["random_state"] is None
    assert nu.metadata["nu"] == 0.5


def test_support_vector_models_reject_precomputed_kernel() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="feature-matrix ModelFit contract"):
        mf.models.svr(X, y, kernel="precomputed")
    with pytest.raises(ValueError, match="feature-matrix ModelFit contract"):
        mf.models.nu_svr(X, y, kernel="precomputed")


def test_adaptive_and_group_penalized_models_fit() -> None:
    X, y = _xy()
    groups = ("trend", "cycle")

    adaptive = mf.models.adaptive_lasso(X, y, alpha=0.001, gamma=1.0, random_state=0)
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
    assert np.isclose(float(adaptive.estimator.adaptive_weights_.mean()), 1.0)
    assert adaptive.metadata["normalize_weights"] is True
    assert grouped.estimator.groups_ == groups
    assert grouped.estimator.resolved_group_weights_["trend"] == pytest.approx(1.0)
    assert grouped.estimator.resolved_group_weights_["cycle"] == pytest.approx(1.0)
    assert sparse_grouped.metadata["l1_ratio"] == 0.25


def test_group_lasso_records_default_sqrt_group_weights() -> None:
    X, y = _xy()
    X = X.assign(x3=0.5 * X["x1"] + 0.1)
    groups = ("macro", "macro", "cycle")

    fit = mf.models.group_lasso(X, y, groups=groups, alpha=0.001)

    assert fit.estimator.groups_ == groups
    assert fit.estimator.resolved_group_weights_["macro"] == pytest.approx(np.sqrt(2.0))
    assert fit.estimator.resolved_group_weights_["cycle"] == pytest.approx(1.0)


def test_group_lasso_validates_group_length() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="one entry per X column"):
        mf.models.group_lasso(X, y, groups=("only_one",))


def test_glmboost_matches_mboost_componentwise_selection_rule() -> None:
    values = np.asarray([1.0, -1.0, 1.0, -1.0])
    orthogonal = np.asarray([1.0, 1.0, -1.0, -1.0])
    X = pd.DataFrame(
        {
            "large_low_corr": 100.0 * (values + np.sqrt(3.0) * orthogonal),
            "exact": values,
        }
    )
    y = pd.Series(values)

    fit = mf.models.glmboost(X, y, n_iter=1, learning_rate=1.0)

    assert fit.estimator.n_iter_ == 1
    assert fit.estimator.coef_[0] == pytest.approx(0.0)
    assert fit.estimator.coef_[1] == pytest.approx(1.0)
    assert fit.metadata["center"] is True


def test_tree_model_records_feature_importance_diagnostics() -> None:
    X, y = _xy()

    fit = mf.models.random_forest(X, y, n_estimators=10, random_state=0, n_jobs=1)

    importance = fit.diagnostics["feature_importance"]
    assert list(importance.index) == ["x1", "x2"] or set(importance.index) == {
        "x1",
        "x2",
    }
    assert np.isfinite(importance.to_numpy(dtype=float)).all()


def test_basic_tree_models_fit_and_predict() -> None:
    X, y = _xy()

    fits = [
        mf.models.decision_tree(X, y, max_depth=2, random_state=0),
        mf.models.extra_trees(
            X,
            y,
            n_estimators=8,
            max_depth=3,
            random_state=0,
            n_jobs=1,
        ),
    ]

    for fit in fits:
        pred = fit.predict(X.iloc[:4])
        assert isinstance(fit, mf.models.ModelFit)
        assert len(pred) == 4
        assert np.isfinite(pred.to_numpy(dtype=float)).all()
        assert "feature_importance" in fit.diagnostics


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
            "lightgbm",
            lambda X, y: mf.models.lgb_plus(
                X,
                y,
                n_ensemble=2,
                n_steps=3,
                min_data_in_leaf=3,
                early_stop_patience=None,
                random_state=0,
            ),
        ),
        (
            "lightgbm",
            lambda X, y: mf.models.lgba_plus(
                X,
                y,
                n_cycles=2,
                trees_per_cycle=2,
                min_data_in_leaf=3,
                random_state=0,
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
    assert {"alpha", "beta", "delta", "eta_1", "eta_2", "persistence"}.issubset(
        fit.diagnostics["params"]
    )
    assert "rugarch-style" in fit.metadata["implementation_note"]
    assert len(fit.diagnostics["conditional_volatility"]) == len(returns)
    assert len(fit.predict_variance(2)) == 2


def test_volatility_models_validate_orders_and_horizon() -> None:
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    returns = pd.Series(0.01 * np.sin(np.arange(40)), index=idx, name="returns")
    realized = (returns.abs() + 0.01) ** 2

    with pytest.raises(ValueError, match="p must be positive"):
        mf.models.GARCHEstimator(variant="garch11", p=0)
    with pytest.raises(ValueError, match="q must be positive"):
        mf.models.GARCHEstimator(variant="garch11", q=0)
    with pytest.raises(ValueError, match="o must be non-negative"):
        mf.models.GARCHEstimator(variant="egarch", o=-1)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        mf.models.realized_garch(returns, rv=realized, max_iter=0)
    with pytest.raises(ValueError, match="n_starts must be positive"):
        mf.models.realized_garch(returns, rv=realized, n_starts=0)

    fit = mf.models.realized_garch(
        returns,
        rv=realized,
        max_iter=20,
        n_starts=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="horizon must be positive"):
        fit.predict_variance(0)


def test_pls_default_clamps_components_to_available_predictors() -> None:
    X, y = _xy()

    fit = mf.models.pls(X, y)

    assert fit.metadata["requested_n_components"] == 3
    assert fit.metadata["resolved_n_components"] == 2
    assert fit.metadata["n_components"] == 2
    assert len(fit.predict(X.iloc[-3:])) == 3
    assert fit.estimator.control_names_ == ("const",)


def test_pls_supports_hounyo_li_control_residualization() -> None:
    X, y = _xy(48)
    X = X.assign(control=y.shift(1).bfill())

    fit = mf.models.pls(
        X,
        y,
        n_components=2,
        control_columns=["control"],
        include_constant=True,
        drop_control_columns=True,
        quadratic_factors=True,
    )

    assert fit.metadata["resolved_n_components"] == 2
    assert fit.metadata["quadratic_factors"] is True
    assert fit.estimator.control_names_ == ("control", "const")
    assert "control" not in fit.estimator.factor_features_
    assert fit.estimator.factor_square_coefs_.shape == (2,)
    assert fit.diagnostics["factor_loadings"].shape[1] == 2
    assert "PLS_emp002.m" in fit.metadata["implementation_note"]
    assert len(fit.predict(X.iloc[-4:])) == 4


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


def test_neural_models_validate_learning_rate_before_backend_import() -> None:
    X, y = _xy()

    factories = [
        lambda: mf.models.nn(X, y, learning_rate=0.0),
        lambda: mf.models.lstm(X, y, learning_rate=0.0),
        lambda: mf.models.gru(X, y, learning_rate=0.0),
        lambda: mf.models.transformer(X, y, learning_rate=0.0),
        lambda: mf.models.hemisphere_nn(X, y, learning_rate=0.0),
    ]

    for factory in factories:
        with pytest.raises(ValueError, match="learning_rate"):
            factory()


def test_nn_spec_documents_all_supported_activations() -> None:
    spec = mf.models.get_model("nn")
    activation = next(
        parameter for parameter in spec.parameters if parameter.name == "activation"
    )

    for name in ("identity", "logistic", "sigmoid", "tanh", "relu", "gelu"):
        assert name in activation.description


def test_neural_prediction_scaling_maps_nonfinite_values_to_fit_mean() -> None:
    from macroforecast.models import neural

    values = np.asarray([[1.0, np.nan], [np.inf, -np.inf]])
    mean = np.asarray([1.0, 2.0])
    scale = np.asarray([2.0, 4.0])

    scaled = neural._standardize_prediction_matrix(values, mean, scale)

    np.testing.assert_allclose(scaled, np.zeros_like(values))


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


def test_density_hnn_matches_aionx_density_contract_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    X, y = _xy(28)
    y = y + 0.15 * np.sin(np.arange(len(y)))

    fit = mf.models.density_hnn(
        X,
        y,
        neurons=4,
        max_epochs=1,
        n_estimators=2,
        prior_estimators=2,
        block_size=4,
        patience=1,
        device="cpu",
        quantile_levels=(0.1, 0.5, 0.9),
    )

    pred = fit.predict(X.iloc[-4:])
    variance = fit.predict_variance(X.iloc[-4:])
    volatility = fit.predict_volatility(X.iloc[-4:])
    quantiles = fit.predict_quantiles(X.iloc[-4:])

    assert len(pred) == 4
    assert variance.shape == (4,)
    assert volatility.shape == (4,)
    assert np.isfinite(variance).all()
    assert np.isfinite(volatility).all()
    assert (variance > 0).all()
    assert (volatility > 0).all()
    assert set(quantiles) == {0.1, 0.5, 0.9}
    assert all(values.shape == (4,) for values in quantiles.values())
    assert fit.metadata["prior_estimators"] == 2
    assert fit.metadata["rescale_volatility"] is True
    assert fit.diagnostics["density"]["backend_alignment"][
        "volatility_rescaling_algorithm"
    ] == "log residual-square calibration"
    assert fit.diagnostics["density"]["volatility_emphasis"] > 0
    assert fit.estimator.oob_prediction_ is not None
    assert {"conditional_mean", "conditional_volatility", "conditional_variance"} <= set(
        fit.estimator.oob_prediction_.columns
    )


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


def test_scaled_pca_supports_hounyo_li_pc2_squared_factor_head() -> None:
    X, y = _xy(42)
    X_train = X.iloc[:34]
    y_train = y.iloc[:34]
    X_test = X.iloc[34:38]

    fit = mf.models.scaled_pca(
        X_train,
        y_train,
        n_components=2,
        quadratic_factors=True,
    )
    _, expected_pred = _huang_scaled_pca_reference(
        X_train,
        y_train,
        X_test,
        n_components=2,
        quadratic_factors=True,
    )

    assert fit.metadata["quadratic_factors"] is True
    assert fit.estimator.factor_square_coefs_.shape == (2,)
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


def test_supervised_scaled_pca_supports_pc2_recursion() -> None:
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
        quadratic_factors=True,
    )
    expected = _matlab_style_spca_reference(
        X_train,
        y_train,
        X_test,
        n_components=2,
        n_selected=2,
        slope_scale=True,
        quadratic_factors=True,
    )

    assert fit.metadata["quadratic_factors"] is True
    assert fit.estimator.factor_square_coefs_.shape == (2,)
    np.testing.assert_allclose(fit.predict(X_test).to_numpy(), expected, atol=1e-10)


def test_tree_models_include_custom_macro_callables() -> None:
    X, y = _xy()

    qrf = mf.models.quantile_regression_forest(
        X,
        y,
        n_estimators=8,
        min_samples_leaf=2,
        random_state=0,
    )

    quantiles = qrf.estimator.predict_quantiles(X.iloc[:4], levels=(0.1, 0.9))
    assert set(quantiles) == {0.1, 0.9}
    assert quantiles[0.1].shape == (4,)


def test_quantile_regression_forest_uses_tree_equal_leaf_weights() -> None:
    class FakeForest:
        def apply(self, X):
            return np.asarray([[1, 2]] * len(X), dtype=int)

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    reg = mf.models.QuantileRegressionForestRegressor(quantile_levels=(0.75,))
    reg._forest = FakeForest()
    reg._leaf_targets = [
        {1: np.zeros(100, dtype=float)},
        {2: np.asarray([10.0], dtype=float)},
    ]

    quantiles = reg.predict_quantiles(pd.DataFrame({"x1": [0.0]}))

    assert quantiles[0.75][0] == pytest.approx(10.0)


def test_hybrid_quantile_levels_are_validated() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="quantile levels"):
        mf.models.quantile_regression_forest(X, y, quantile_levels=(0.0, 0.5))


def test_macro_random_forest_prediction_values_validate_backend_shape() -> None:
    output = {"pred_ensemble": pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})}

    values = mf.models.MacroRandomForestRegressor._prediction_values(output, 2)

    np.testing.assert_allclose(values, [2.0, 3.0])
    with pytest.raises(RuntimeError, match="did not return"):
        mf.models.MacroRandomForestRegressor._prediction_values({}, 2)


def test_macro_random_forest_position_validation() -> None:
    features = pd.Index(["x1", "x2"])

    with pytest.raises(ValueError, match="positions"):
        mf.models.MacroRandomForestRegressor._resolve_positions(None, (0,), features)
    with pytest.raises(ValueError, match="y_pos must be 0"):
        mf.models.macro_random_forest(
            pd.DataFrame({"x1": [1.0, 2.0]}),
            [1.0, 2.0],
            y_pos=1,
        )


@pytest.mark.parametrize(
    ("module", "extra", "fit_call"),
    [
        ("xgboost", "xgboost", lambda X, y: mf.models.xgboost(X, y)),
        ("lightgbm", "lightgbm", lambda X, y: mf.models.lightgbm(X, y)),
        ("lightgbm", "lightgbm", lambda X, y: mf.models.lgb_plus(X, y)),
        ("lightgbm", "lightgbm", lambda X, y: mf.models.lgba_plus(X, y)),
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


def test_lgb_plus_competition_records_reference_channels() -> None:
    pytest.importorskip("lightgbm")
    X, y = _xy(64)

    fit = mf.models.lgb_plus(
        X,
        y,
        n_ensemble=2,
        n_steps=6,
        min_data_in_leaf=4,
        num_leaves=5,
        linear_candidate_fraction=0.5,
        early_stop_patience=None,
        random_state=0,
    )
    components = fit.estimator.predict_components(X.iloc[:5])
    summary = fit.diagnostics["channel_summary"]

    np.testing.assert_allclose(
        components["prediction_total"],
        components["prediction_init"]
        + components["prediction_tree"]
        + components["prediction_linear"],
    )
    assert summary["total_tree"] + summary["total_linear"] == 12
    assert "channel_importance" in fit.diagnostics
    assert fit.diagnostics["channel_importance"].shape[0] == X.shape[1]
    assert fit.metadata["linear_candidate_fraction"] == 0.5
    assert "lgb_plus.R" in fit.diagnostics["training_history"]["source_reference"]


def test_lgba_plus_alternating_records_reference_channels() -> None:
    pytest.importorskip("lightgbm")
    X, y = _xy(64)

    fit = mf.models.lgba_plus(
        X,
        y,
        n_runs=2,
        n_cycles=3,
        trees_per_cycle=2,
        min_data_in_leaf=4,
        random_state=0,
    )
    components = fit.estimator.predict_components(X.iloc[:5])
    summary = fit.diagnostics["channel_summary"]

    np.testing.assert_allclose(
        components["prediction_total"],
        components["prediction_init"]
        + components["prediction_tree"]
        + components["prediction_linear"],
    )
    assert summary["total_tree_blocks"] == 6
    assert summary["total_linear_steps"] == 6
    assert fit.estimator.get_total_trees() == 12
    assert "channel_importance" in fit.diagnostics
    assert "lgb_plus_A.R" in fit.diagnostics["training_history"]["source_reference"]


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
    assert "RandomForestRegressor" in rf.metadata["implementation_note"]
    assert gb.metadata["loss"] == "absolute_error"
    assert "Booging" in gb.metadata["implementation_note"]


def test_tree_ensemble_public_signatures_match_specs() -> None:
    for name in (
        "lgb_plus",
        "lgba_plus",
        "quantile_regression_forest",
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


def test_quantile_regression_forest_records_feature_importance_diagnostics() -> None:
    X, y = _xy()

    fit = mf.models.quantile_regression_forest(
        X, y, n_estimators=8, min_samples_leaf=2, random_state=0
    )

    assert "feature_importance" in fit.diagnostics
    assert np.isfinite(
        fit.diagnostics["feature_importance"].to_numpy(dtype=float)
    ).all()


def test_macro_random_forest_rejects_duplicate_selector_styles() -> None:
    X, y = _xy()

    with pytest.raises(ValueError, match="x_columns or x_pos"):
        mf.models.macro_random_forest(X, y, x_columns=["x1"], x_pos=(1,))
    with pytest.raises(ValueError, match="S_columns or S_pos"):
        mf.models.macro_random_forest(X, y, S_columns=["x1"], S_pos=(1,))


def test_ar_and_var_fit() -> None:
    X, y = _xy()
    panel = pd.concat([y, X], axis=1)

    ar_fit = mf.models.ar(y, n_lag=2)
    var_fit = mf.models.var(panel, target="y", n_lag=1, type="both")

    assert len(ar_fit.predict(pd.DataFrame(index=y.index[-4:]))) == 4
    assert len(var_fit.predict(panel.iloc[-4:])) == 4
    assert var_fit.metadata["backend"] == "internal vars::VAR-aligned OLS"
    assert var_fit.metadata["type"] == "both"


def test_var_rejects_invalid_type_label() -> None:
    X, y = _xy()
    panel = pd.concat([y, X], axis=1)

    with pytest.raises(ValueError, match="type must be one of"):
        mf.models.var(panel, target="y", type="ctt")
