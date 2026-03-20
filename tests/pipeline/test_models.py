"""Tests for pipeline/models.py — all Python-side estimators.

Tests use small synthetic datasets to keep runtime fast.  Each test verifies:
1. fit() returns self (fluent API)
2. predict() returns correct shape
3. nonlinearity_type is declared
"""

import numpy as np
import pytest

from macrocast.pipeline.components import Nonlinearity
from macrocast.pipeline.models import (
    GBModel,
    KRRModel,
    LSTMModel,
    NNModel,
    RFModel,
    SVRLinearModel,
    SVRRBFModel,
    XGBoostModel,
)


@pytest.fixture()
def small_data():
    """Small (T=60, N=5) dataset for fast model tests."""
    rng = np.random.default_rng(42)
    T, N = 60, 5
    X = rng.standard_normal((T, N))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(T) * 0.1
    return X[:50], y[:50], X[50:], y[50:]


@pytest.fixture()
def small_seq_data():
    """Small sequence dataset (T=50, L=4, N=3) for LSTM tests."""
    rng = np.random.default_rng(42)
    T, L, N = 50, 4, 3
    X = rng.standard_normal((T, L, N))
    y = rng.standard_normal(T)
    return X[:40], y[:40], X[40:], y[40:]


# ---------------------------------------------------------------------------
# KRR
# ---------------------------------------------------------------------------


class TestKRRModel:
    def test_nonlinearity_type(self):
        assert KRRModel.nonlinearity_type == Nonlinearity.KRR

    def test_fit_returns_self(self, small_data):
        X_tr, y_tr, _, _ = small_data
        model = KRRModel(alpha_grid=[0.1, 1.0], gamma_grid=[0.1, 1.0], cv_folds=2)
        result = model.fit(X_tr, y_tr)
        assert result is model

    def test_predict_shape(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = KRRModel(alpha_grid=[0.1], gamma_grid=[0.1], cv_folds=2)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)


# ---------------------------------------------------------------------------
# SVR-RBF
# ---------------------------------------------------------------------------


class TestSVRRBFModel:
    def test_nonlinearity_type(self):
        assert SVRRBFModel.nonlinearity_type == Nonlinearity.SVR_RBF

    def test_fit_predict(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = SVRRBFModel(C_grid=[1.0], gamma_grid=[0.1], epsilon_grid=[0.1], cv_folds=2)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)


# ---------------------------------------------------------------------------
# SVR-Linear
# ---------------------------------------------------------------------------


class TestSVRLinearModel:
    def test_nonlinearity_type(self):
        assert SVRLinearModel.nonlinearity_type == Nonlinearity.SVR_LINEAR

    def test_fit_predict(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = SVRLinearModel(C_grid=[1.0], epsilon_grid=[0.1], cv_folds=2)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class TestRFModel:
    def test_nonlinearity_type(self):
        assert RFModel.nonlinearity_type == Nonlinearity.RANDOM_FOREST

    def test_fit_predict(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = RFModel(
            n_estimators=10,
            max_depth_grid=[3],
            min_samples_leaf_grid=[5],
            cv_folds=2,
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    def test_nonlinearity_type(self):
        assert XGBoostModel.nonlinearity_type == Nonlinearity.XGBOOST

    def test_fit_predict(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = XGBoostModel(
            n_estimators=20,
            max_depth_grid=[3],
            learning_rate_grid=[0.1],
            subsample_grid=[1.0],
            cv_folds=2,
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)


# ---------------------------------------------------------------------------
# Gradient Boosting (sklearn)
# ---------------------------------------------------------------------------


class TestGBModel:
    def test_nonlinearity_type(self):
        assert GBModel.nonlinearity_type == Nonlinearity.GRADIENT_BOOSTING

    def test_fit_returns_self(self, small_data):
        X_tr, y_tr, _, _ = small_data
        m = GBModel(
            n_estimators=50,
            max_depth_grid=[3],
            learning_rate_grid=[0.1],
            min_samples_leaf_grid=[1],
            subsample_grid=[1.0],
            cv_folds=2,
        )
        assert m.fit(X_tr, y_tr) is m

    def test_predict_shape(self, small_data):
        X_tr, y_tr, X_te, y_te = small_data
        model = GBModel(
            n_estimators=50,
            max_depth_grid=[3],
            learning_rate_grid=[0.1],
            min_samples_leaf_grid=[1],
            subsample_grid=[1.0],
            cv_folds=2,
        )
        model.fit(X_tr, y_tr)
        assert model.predict(X_te).shape == (len(y_te),)

    def test_custom_grids(self, small_data):
        X_tr, y_tr, _, _ = small_data
        model = GBModel(
            max_depth_grid=[3],
            learning_rate_grid=[0.1],
            min_samples_leaf_grid=[1],
            subsample_grid=[1.0],
            cv_folds=2,
        )
        model.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# NN
# ---------------------------------------------------------------------------


class TestNNModel:
    def test_nonlinearity_type(self):
        assert NNModel.nonlinearity_type == Nonlinearity.NEURAL_NET

    def test_fit_predict(self, small_data):
        X_tr, y_tr, X_te, _ = small_data
        model = NNModel(
            hidden_dims=[16],
            n_layers_options=[1],
            lr_options=[1e-3],
            dropout_options=[0.0],
            max_epochs=5,
            patience=3,
            batch_size=16,
            device="cpu",
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)

    def test_best_params_populated(self, small_data):
        X_tr, y_tr, _, _ = small_data
        model = NNModel(
            hidden_dims=[16],
            n_layers_options=[1],
            lr_options=[1e-3],
            dropout_options=[0.0],
            max_epochs=3,
            device="cpu",
        )
        model.fit(X_tr, y_tr)
        assert "hidden_dim" in model.best_params_


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


class TestLSTMModel:
    def test_nonlinearity_type(self):
        assert LSTMModel.nonlinearity_type == Nonlinearity.LSTM

    def test_fit_predict(self, small_seq_data):
        X_tr, y_tr, X_te, _ = small_seq_data
        model = LSTMModel(
            hidden_dims=[8],
            n_layers_options=[1],
            lr_options=[1e-3],
            dropout_options=[0.0],
            max_epochs=3,
            patience=2,
            batch_size=8,
            device="cpu",
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        assert y_hat.shape == (len(X_te),)
