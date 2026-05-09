"""Deep model seed safety for sweep-style execution.

The full sweep runner assigns distinct random seeds across cells. These tests
pin the torch-backed sequence model side of that contract: identical seeds
reproduce, while distinct seeds change the fitted sequence model.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.deep

from macroforecast.core.runtime import _TorchSequenceModel

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _panel(n_obs: int = 24) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(20260509)
    x1 = rng.normal(size=n_obs)
    x2 = rng.normal(size=n_obs)
    y = 0.6 * x1 - 0.25 * x2 + rng.normal(scale=0.05, size=n_obs)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y, name="y")


def _fit_predict(kind: str, seed: int) -> np.ndarray:
    X, y = _panel()
    model = _TorchSequenceModel(
        kind=kind,
        n_epochs=3,
        hidden_size=4,
        random_state=seed,
    )
    return model.fit(X, y).predict(X)


@pytest.mark.skipif(not HAS_TORCH, reason="requires macroforecast[deep] (torch)")
@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_deep_sequence_same_seed_reproduces(kind: str) -> None:
    first = _fit_predict(kind, seed=42)
    second = _fit_predict(kind, seed=42)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="requires macroforecast[deep] (torch)")
@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_deep_sequence_distinct_seed_changes_fit(kind: str) -> None:
    first = _fit_predict(kind, seed=42)
    second = _fit_predict(kind, seed=43)
    assert not np.allclose(first, second, rtol=1e-5, atol=1e-6)
