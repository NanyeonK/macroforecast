"""Issue #194 -- gradient-based attribution methods.

When torch + captum are not installed (default lightweight install),
``_gradient_attribution_frame`` raises ``NotImplementedError`` with the
``[deep]`` install hint. The pin lives in the default lane; behind the
``deep`` marker we cover the captum-backed path.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from macrocast.core.runtime import _gradient_attribution_frame
from macrocast.core.types import ModelArtifact


HAS_DEEP = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("captum") is not None
)


@pytest.mark.parametrize("op", ["gradient_shap", "integrated_gradients", "saliency_map", "deep_lift"])
def test_gradient_methods_require_torch_when_model_is_sklearn(op):
    # Non-torch model -> NotImplementedError pointing at [deep].
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=20))
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="ols",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    with pytest.raises(NotImplementedError, match=r"(deep|torch)"):
        _gradient_attribution_frame(artifact, X, kind=op)


@pytest.mark.deep
@pytest.mark.skipif(not HAS_DEEP, reason="requires macrocast[deep] (torch + captum)")
def test_saliency_map_runs_under_torch_captum():
    from macrocast.core.runtime import _TorchSequenceModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(size=30))
    model = _TorchSequenceModel(kind="lstm", n_epochs=2, hidden_size=4, random_state=0).fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="lstm",
        fitted_object=model,
        framework="torch",
        feature_names=tuple(X.columns),
    )
    frame = _gradient_attribution_frame(artifact, X, kind="saliency_map")
    assert {"feature", "importance", "method"}.issubset(frame.columns)
    assert (frame["importance"] >= 0).all()
