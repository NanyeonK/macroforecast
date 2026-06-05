"""ADD: stl_decompose (seasonal-trend decomposition)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_stl_decompose_reconstructs_and_infers_period():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    t = np.arange(120)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 12)
    y = pd.Series(0.1 * t + seasonal + np.random.default_rng(0).normal(scale=0.2, size=120),
                  index=idx, name="x")
    out = mf.filters.stl_decompose(y)            # period inferred = 12 (monthly)
    assert out.params["period"] == 12
    comp = out.values
    assert list(comp.columns) == ["trend", "seasonal", "resid"]
    # additive STL: trend + seasonal + resid == observed
    recon = comp["trend"] + comp["seasonal"] + comp["resid"]
    np.testing.assert_allclose(recon.to_numpy(), y.to_numpy(), atol=1e-8)
    # recovered seasonal is ~ period-12 and captures the sine amplitude
    assert comp["seasonal"].std() > 1.0
    assert out.metadata["fit_policy"] == "full_input_two_sided"

def test_stl_requires_period_when_uninferrable():
    y = pd.Series(np.random.default_rng(0).normal(size=60))  # plain RangeIndex
    try:
        mf.filters.stl_decompose(y); assert False
    except ValueError:
        pass
