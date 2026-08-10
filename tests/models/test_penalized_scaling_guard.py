"""Penalized regression must refuse (or standardize) on wildly-scaled features."""
import numpy as np
import pandas as pd
import pytest
from macroforecast.models import ridge, lasso, elastic_net

def _multi(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "trend": np.arange(n, dtype=float),
        "big": np.cumsum(rng.normal(0, 40, n)),
        "small": rng.normal(0, 0.02, n),
    }), pd.Series(rng.normal(size=n), name="y")

def _homog(n=100):
    rng = np.random.default_rng(1)
    return pd.DataFrame({f"c{i}": rng.normal(0, 1, n) for i in range(4)}), pd.Series(rng.normal(size=n), name="y")

@pytest.mark.parametrize("fn", [ridge, lasso, elastic_net])
def test_penalized_errors_on_unstandardized_multiscale(fn):
    X, y = _multi()
    with pytest.raises(ValueError, match="scales span"):
        fn(X, y)

@pytest.mark.parametrize("fn", [ridge, lasso, elastic_net])
def test_penalized_ok_when_standardized(fn):
    X, y = _multi()
    fn(X, y, standardize=True)  # must not raise

@pytest.mark.parametrize("fn", [ridge, lasso, elastic_net])
def test_penalized_ok_on_homogeneous_scales(fn):
    X, y = _homog()
    fn(X, y)  # comparable scales -> no guard

def test_ridge_gained_standardize_param():
    from macroforecast.models import get_model
    assert "standardize" in get_model("ridge").default_params
