"""Correctness tests for phillips_ouliaris cointegration test (urca::ca.po)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

import macroforecast as mf


def test_detects_cointegration():
    rng = np.random.default_rng(0)
    n = 400
    x = np.cumsum(rng.standard_normal(n))
    y = 2.0 + 1.5 * x + rng.standard_normal(n) * 0.5
    res = mf.data_analysis.phillips_ouliaris(y, x)
    assert res["test"] == "phillips_ouliaris"
    assert res["cointegrated"] is True
    assert res["p_value"] < 0.05


def test_independent_not_cointegrated():
    rng = np.random.default_rng(1)
    n = 400
    y = np.cumsum(rng.standard_normal(n))
    z = np.cumsum(rng.standard_normal(n))
    res = mf.data_analysis.phillips_ouliaris(y, z)
    assert res["cointegrated"] is False


def test_matches_arch():
    from arch.unitroot.cointegration import phillips_ouliaris as _po
    rng = np.random.default_rng(5)
    n = 300
    x = np.cumsum(rng.standard_normal(n))
    y = 1.0 + 0.9 * x + rng.standard_normal(n) * 0.4
    res = mf.data_analysis.phillips_ouliaris(y, x, test_type="Zt")
    ref = _po(y, np.asarray(x).reshape(-1, 1), trend="c", test_type="Zt")
    np.testing.assert_allclose(res["statistic"], ref.stat, rtol=1e-8)
    np.testing.assert_allclose(res["p_value"], ref.pvalue, rtol=1e-6)


def test_single_panel_and_test_type_variants():
    rng = np.random.default_rng(3)
    n = 300
    x = np.cumsum(rng.standard_normal(n))
    y = 5.0 + 2.0 * x + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, "x": x})
    for tt in ("Zt", "Za"):
        res = mf.data_analysis.phillips_ouliaris(df, test_type=tt)
        assert res["test_type"] == tt
        assert res["cointegrated"] is True


def test_invalid_args():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(100))
    x = np.cumsum(rng.standard_normal(100))
    with pytest.raises(ValueError):
        mf.data_analysis.phillips_ouliaris(y, x, test_type="XX")
