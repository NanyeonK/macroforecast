"""Sweep-safety: running two variants with different variant_ids must produce
different predictions through the executor path, and no state leaks across
runs (dispatch + seed policy threading).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.deep

from macrocast.execution.build import _run_deep_autoreg_executor
from macrocast.execution.models.deep import _base as deep_base
from macrocast.execution.models.deep._base import DeepModelConfig
from macrocast.execution.seed_policy import (
    ReproducibilityContext,
    reset_context,
    set_context,
)


@pytest.fixture(autouse=True)
def fast_config(monkeypatch):
    """Reduce max_epochs so the suite completes quickly — sweep-safety is
    about state-leakage, not training quality."""

    orig = deep_base.DeepModelConfig

    class FastConfig(DeepModelConfig):
        pass

    def _fast_ctor(*args, **kwargs):
        kwargs.setdefault("max_epochs", 3)
        return orig(*args, **kwargs)

    monkeypatch.setattr(deep_base, "DeepModelConfig", _fast_ctor)
    yield
    # monkeypatch restores automatically


def _synthetic_series(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal() * 0.4
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.Series(x, index=idx, name="y")


def _run_with_variant(family: str, variant_id: str, series: pd.Series) -> float:
    ctx = ReproducibilityContext(
        recipe_id="test-recipe",
        variant_id=variant_id,
        reproducibility_spec={"reproducibility_mode": "strict_reproducible"},
    )
    token = set_context(ctx)
    try:
        out = _run_deep_autoreg_executor(family, series, horizon=1)
    finally:
        reset_context(token)
    assert set(out.keys()) >= {"y_pred", "selected_lag", "selected_bic", "tuning_payload"}
    tuning_payload = out["tuning_payload"]
    assert tuning_payload["sequence_representation_contract"] == "sequence_representation_contract_v1"
    assert tuning_payload["sequence_generator_family"] == family
    assert tuning_payload["sequence_shape"][1:] == [12, 1]
    return float(out["y_pred"])


@pytest.mark.parametrize("family", ["lstm", "gru", "tcn"])
def test_sweep_variants_diverge_by_variant_id(family):
    series = _synthetic_series(n=60, seed=0)
    pred_a = _run_with_variant(family, variant_id="variant_a", series=series)
    pred_b = _run_with_variant(family, variant_id="variant_b", series=series)
    assert np.isfinite(pred_a)
    assert np.isfinite(pred_b)
    assert pred_a != pred_b, f"{family}: identical predictions across variants"


@pytest.mark.parametrize("family", ["lstm", "gru", "tcn"])
def test_sweep_same_variant_reproduces(family):
    series = _synthetic_series(n=60, seed=0)
    p1 = _run_with_variant(family, variant_id="same", series=series)
    p2 = _run_with_variant(family, variant_id="same", series=series)
    assert p1 == p2, f"{family}: same variant_id produced different predictions"
