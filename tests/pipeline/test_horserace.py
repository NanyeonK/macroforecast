"""Tests for pipeline/horserace.py — HorseRaceGrid."""

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.components import CVScheme, LossFunction, Nonlinearity, Regularization, Window
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.pipeline.horserace import HorseRaceGrid
from macrocast.pipeline.results import ResultSet


@pytest.fixture()
def synthetic_panel():
    """Small synthetic panel: T=120, N=5, monthly starting 2005-01 (ends 2014-12)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2005-01", periods=120, freq="MS")
    X = pd.DataFrame(rng.standard_normal((120, 5)), index=dates, columns=[f"x{i}" for i in range(5)])
    y = pd.Series(X.iloc[:, 0] + 0.3 * X.iloc[:, 1] + rng.standard_normal(120) * 0.1, index=dates)
    return X, y


@pytest.fixture()
def minimal_model_specs():
    """One KRR model spec with small grid for fast tests."""
    from macrocast.pipeline.models import KRRModel
    return [
        ModelSpec(
            model_cls=KRRModel,
            regularization=Regularization.NONE,
            cv_scheme=CVScheme.KFOLD(k=2),
            loss_function=LossFunction.L2,
            model_kwargs={"alpha_grid": [0.1], "gamma_grid": [0.1], "cv_folds": 2},
            model_id="krr_test",
        )
    ]


class TestHorseRaceGrid:
    def test_run_returns_result_set(self, synthetic_panel, minimal_model_specs):
        X, y = synthetic_panel
        feature_specs = [
            FeatureSpec(use_factors=False, n_lags=2, label="AR"),
            FeatureSpec(use_factors=True, n_factors=2, n_lags=2, label="F"),
        ]
        grid = HorseRaceGrid(
            panel=X,
            target=y,
            horizons=[1],
            model_specs=minimal_model_specs,
            feature_specs=feature_specs,
            oos_start="2014-01-01",
            oos_end="2014-03-01",
        )
        rs = grid.run()
        assert isinstance(rs, ResultSet)
        assert len(rs) > 0

    def test_feature_set_labels_populated(self, synthetic_panel, minimal_model_specs):
        X, y = synthetic_panel
        feature_specs = [
            FeatureSpec(use_factors=False, n_lags=2, label="AR"),
            FeatureSpec(use_factors=True, n_factors=2, n_lags=2, label="F"),
        ]
        grid = HorseRaceGrid(
            panel=X,
            target=y,
            horizons=[1],
            model_specs=minimal_model_specs,
            feature_specs=feature_specs,
            oos_start="2014-01-01",
            oos_end="2014-03-01",
        )
        rs = grid.run()
        df = rs.to_dataframe()
        feature_set_values = set(df["feature_set"].unique())
        assert "AR" in feature_set_values
        assert "F" in feature_set_values

    def test_merge_result_sets(self):
        """merge_result_sets tags records and combines them."""
        from macrocast.pipeline.results import ForecastRecord
        from macrocast.pipeline.components import Nonlinearity, Regularization, LossFunction, Window, CVScheme

        def make_record(feature_set: str) -> ForecastRecord:
            return ForecastRecord(
                experiment_id="test",
                model_id="m1",
                nonlinearity=Nonlinearity.KRR,
                regularization=Regularization.NONE,
                cv_scheme=CVScheme.KFOLD(k=2),
                loss_function=LossFunction.L2,
                window=Window.EXPANDING,
                horizon=1,
                train_end=pd.Timestamp("2010-01-01"),
                forecast_date=pd.Timestamp("2010-02-01"),
                y_hat=0.1,
                y_true=0.2,
                n_train=50,
                n_factors=None,
                n_lags=2,
                feature_set=feature_set,
            )

        rs1 = ResultSet()
        rs1.add(make_record(""))  # empty — should be tagged
        rs2 = ResultSet()
        rs2.add(make_record("F"))  # already set — should not be overwritten

        merged = HorseRaceGrid.merge_result_sets([rs1, rs2], ["AR", "F"])
        assert len(merged) == 2
        records = {r.feature_set for r in merged.records}
        assert "AR" in records
        assert "F" in records

    def test_two_specs_double_records(self, synthetic_panel, minimal_model_specs):
        """Two feature specs produce twice as many records as one."""
        X, y = synthetic_panel
        spec_ar = FeatureSpec(use_factors=False, n_lags=2, label="AR")
        spec_f = FeatureSpec(use_factors=True, n_factors=2, n_lags=2, label="F")

        grid_one = HorseRaceGrid(
            panel=X, target=y, horizons=[1],
            model_specs=minimal_model_specs, feature_specs=[spec_ar],
            oos_start="2014-01-01",
            oos_end="2014-03-01",
        )
        grid_two = HorseRaceGrid(
            panel=X, target=y, horizons=[1],
            model_specs=minimal_model_specs, feature_specs=[spec_ar, spec_f],
            oos_start="2014-01-01",
            oos_end="2014-03-01",
        )
        rs_one = grid_one.run()
        rs_two = grid_two.run()
        assert len(rs_two) == 2 * len(rs_one)
