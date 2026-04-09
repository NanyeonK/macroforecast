"""Integration tests for pipeline/experiment.py — ForecastExperiment."""

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.components import CVScheme, LossFunction, Regularization, Window
from macrocast.pipeline.estimator import MacrocastEstimator
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.models import KRRModel, RFModel
from macrocast.pipeline.results import FailureRecord, ResultSet


@pytest.fixture()
def synthetic_panel():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2005-01", periods=120, freq="MS")
    X = rng.standard_normal((120, 10))
    y = X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(120) * 0.2
    panel = pd.DataFrame(X, index=dates, columns=[f"x{i}" for i in range(10)])
    target = pd.Series(y, index=dates, name="target")
    return panel, target


@pytest.fixture()
def krr_spec():
    return ModelSpec(
        model_cls=KRRModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_kwargs={"alpha_grid": [0.1, 1.0], "gamma_grid": [0.1], "cv_folds": 2},
    )


@pytest.fixture()
def rf_spec():
    return ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_kwargs={"n_estimators": 5, "min_samples_leaf_grid": [5], "cv_folds": 2},
    )


class BrokenModel(MacrocastEstimator):
    nonlinearity_type = RFModel.nonlinearity_type
    def fit(self, X, y):
        raise RuntimeError('boom')
    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture()
def broken_spec():
    return ModelSpec(
        model_cls=BrokenModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_id='broken_model',
    )


class TestForecastExperiment:
    def test_run_returns_result_set(self, synthetic_panel, krr_spec):
        panel, target = synthetic_panel
        exp = ForecastExperiment(panel=panel, target=target, horizons=[1], model_specs=[krr_spec], feature_spec=FeatureSpec(n_factors=2, n_lags=2, factor_type="X"), window=Window.EXPANDING, oos_start="2014-01-01", oos_end="2014-03-01", n_jobs=1)
        rs = exp.run()
        assert isinstance(rs, ResultSet)
        assert len(rs) > 0

    def test_run_captures_failure_record(self, synthetic_panel, broken_spec):
        panel, target = synthetic_panel
        exp = ForecastExperiment(panel=panel, target=target, horizons=[1], model_specs=[broken_spec], feature_spec=FeatureSpec(n_factors=2, n_lags=2, factor_type="X"), window=Window.EXPANDING, oos_start="2014-01-01", oos_end="2014-01-01", n_jobs=1)
        rs = exp.run()
        assert len(rs.failures) == 1
        assert isinstance(rs.failures[0], FailureRecord)
        assert rs.degraded is True

    def test_parquet_output_writes_failure_log_when_needed(self, synthetic_panel, broken_spec, tmp_path):
        panel, target = synthetic_panel
        exp = ForecastExperiment(panel=panel, target=target, horizons=[1], model_specs=[broken_spec], feature_spec=FeatureSpec(n_factors=2, n_lags=2), oos_start="2014-01-01", oos_end="2014-01-01", n_jobs=1, output_dir=tmp_path)
        exp.run()
        assert len(list(tmp_path.glob('*.failures.parquet'))) == 1
