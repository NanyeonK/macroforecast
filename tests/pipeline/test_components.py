"""Tests for pipeline/components.py."""

import pytest

from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
    _BICScheme,
    _KFoldCV,
    _POOSScheme,
)


class TestNonlinearity:
    def test_values_exist(self):
        assert Nonlinearity.LINEAR.value == "linear"
        assert Nonlinearity.KRR.value == "krr"
        assert Nonlinearity.SVR_RBF.value == "svr_rbf"
        assert Nonlinearity.SVR_LINEAR.value == "svr_linear"
        assert Nonlinearity.RANDOM_FOREST.value == "random_forest"
        assert Nonlinearity.XGBOOST.value == "xgboost"
        assert Nonlinearity.NEURAL_NET.value == "neural_net"
        assert Nonlinearity.LSTM.value == "lstm"
        assert Nonlinearity.GRADIENT_BOOSTING.value == "gradient_boosting"

    def test_total_count(self):
        assert len(Nonlinearity) == 9


class TestRegularization:
    def test_values_exist(self):
        assert Regularization.NONE.value == "none"
        assert Regularization.RIDGE.value == "ridge"
        assert Regularization.LASSO.value == "lasso"
        assert Regularization.ADAPTIVE_LASSO.value == "adaptive_lasso"
        assert Regularization.GROUP_LASSO.value == "group_lasso"
        assert Regularization.ELASTIC_NET.value == "elastic_net"
        assert Regularization.FACTORS.value == "factors"
        assert Regularization.TVP_RIDGE.value == "tvp_ridge"
        assert Regularization.BOOGING.value == "booging"

    def test_total_count(self):
        assert len(Regularization) == 9


class TestCVScheme:
    def test_bic_is_singleton(self):
        assert CVScheme.BIC is CVScheme.BIC
        assert isinstance(CVScheme.BIC, _BICScheme)

    def test_poos_is_singleton(self):
        assert CVScheme.POOS is CVScheme.POOS
        assert isinstance(CVScheme.POOS, _POOSScheme)

    def test_kfold_default(self):
        kf = CVScheme.KFOLD()
        assert isinstance(kf, _KFoldCV)
        assert kf.k == 5

    def test_kfold_custom(self):
        kf = CVScheme.KFOLD(k=10)
        assert kf.k == 10

    def test_kfold_frozen_equality(self):
        assert CVScheme.KFOLD(k=5) == CVScheme.KFOLD(k=5)
        assert CVScheme.KFOLD(k=3) != CVScheme.KFOLD(k=5)

    def test_kfold_hashable(self):
        # frozen dataclasses must be usable as dict keys
        d = {CVScheme.KFOLD(k=5): "five", CVScheme.BIC: "bic"}
        assert d[CVScheme.KFOLD(k=5)] == "five"


class TestLossFunction:
    def test_values(self):
        assert LossFunction.L2.value == "l2"
        assert LossFunction.EPSILON_INSENSITIVE.value == "epsilon_insensitive"


class TestWindow:
    def test_values(self):
        assert Window.EXPANDING.value == "expanding"
        assert Window.ROLLING.value == "rolling"
