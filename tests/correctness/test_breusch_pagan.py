"""Correctness tests for breusch_pagan_test (lmtest::bptest)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_studentized_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import het_breuschpagan
    rng = np.random.default_rng(9)
    n = 200
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 1 + 0.5 * x1 - 0.3 * x2 + rng.standard_normal(n) * (0.4 + 0.8 * np.abs(x1))
    X = pd.DataFrame({"x1": x1, "x2": x2})
    res = mf.data_analysis.breusch_pagan_test(X, y, studentize=True)
    lm, lm_p, _, _ = het_breuschpagan(
        y - sm.add_constant(X.values) @ np.linalg.lstsq(
            sm.add_constant(X.values), y, rcond=None)[0],
        sm.add_constant(X.values),
    )
    np.testing.assert_allclose(res["statistic"], lm, rtol=1e-8)
    np.testing.assert_allclose(res["p_value"], lm_p, rtol=1e-6)
    assert res["df"] == 2


def test_detects_heteroskedasticity():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.standard_normal(n)
    y_het = 1 + x + rng.standard_normal(n) * (0.2 + 1.5 * np.abs(x))
    y_hom = 1 + x + rng.standard_normal(n)
    X = pd.DataFrame({"x": x})
    assert mf.data_analysis.breusch_pagan_test(X, y_het)["p_value"] < 0.05
    assert mf.data_analysis.breusch_pagan_test(X, y_hom)["p_value"] > 0.05


def test_classic_variant_runs():
    rng = np.random.default_rng(4)
    n = 150
    x = rng.standard_normal(n)
    y = 1 + x + rng.standard_normal(n) * (0.3 + np.abs(x))
    res = mf.data_analysis.breusch_pagan_test(pd.DataFrame({"x": x}), y, studentize=False)
    assert res["version"] == "breusch_pagan_godfrey"
    assert res["statistic"] >= 0.0 and np.isfinite(res["p_value"])
