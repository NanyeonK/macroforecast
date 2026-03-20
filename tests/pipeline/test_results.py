"""Tests for pipeline/results.py — ForecastRecord and ResultSet."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.results import ForecastRecord, ResultSet


def _make_record(y_hat: float = 1.0, y_true: float = 1.5, horizon: int = 1) -> ForecastRecord:
    return ForecastRecord(
        experiment_id="test-exp-001",
        model_id="krr__factors__KFold(k=5)__l2",
        nonlinearity=Nonlinearity.KRR,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(k=5),
        loss_function=LossFunction.L2,
        window=Window.EXPANDING,
        horizon=horizon,
        train_end=pd.Timestamp("2010-01-01"),
        forecast_date=pd.Timestamp("2010-02-01"),
        y_hat=y_hat,
        y_true=y_true,
        n_train=100,
        n_factors=8,
        n_lags=4,
    )


@pytest.fixture
def sample_record() -> ForecastRecord:
    return _make_record()


class TestForecastRecord:
    def test_feature_set_default(self, sample_record: ForecastRecord):
        assert sample_record.feature_set == ""
        assert "feature_set" in sample_record.to_dict()

    def test_error(self):
        r = _make_record(y_hat=1.0, y_true=2.0)
        assert r.error == pytest.approx(1.0)

    def test_squared_error(self):
        r = _make_record(y_hat=1.0, y_true=3.0)
        assert r.squared_error == pytest.approx(4.0)

    def test_to_dict_keys(self):
        r = _make_record()
        d = r.to_dict()
        assert "model_id" in d
        assert "nonlinearity" in d
        assert d["nonlinearity"] == "krr"


class TestResultSet:
    def test_add_and_len(self):
        rs = ResultSet()
        rs.add(_make_record())
        rs.add(_make_record())
        assert len(rs) == 2

    def test_to_dataframe_shape(self):
        rs = ResultSet()
        for _ in range(5):
            rs.add(_make_record())
        df = rs.to_dataframe()
        assert df.shape[0] == 5

    def test_to_dataframe_empty(self):
        rs = ResultSet()
        df = rs.to_dataframe()
        assert df.empty

    def test_msfe_by_model(self):
        rs = ResultSet()
        rs.add(_make_record(y_hat=1.0, y_true=2.0))  # se=1
        rs.add(_make_record(y_hat=1.0, y_true=3.0))  # se=4
        summary = rs.msfe_by_model()
        assert summary["msfe"].iloc[0] == pytest.approx(2.5)  # mean of 1,4

    def test_parquet_roundtrip(self):
        rs = ResultSet(experiment_id="round-trip-test")
        rs.add(_make_record(horizon=1))
        rs.add(_make_record(horizon=3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.parquet"
            rs.to_parquet(path)
            assert path.exists()
            df = pd.read_parquet(path)
            assert len(df) == 2
            assert set(df["horizon"]) == {1, 3}
