"""ADD: genuine Giacomini-White (2006) conditional predictive ability test."""
from __future__ import annotations
import numpy as np, pandas as pd
import pytest
import macroforecast as mf

def test_gw_no_rejection_for_equal_ability():
    rng = np.random.default_rng(0)
    la = pd.Series(np.abs(rng.normal(size=400)))
    lb = pd.Series(np.abs(rng.normal(size=400)))      # same distribution -> equal ability
    res = mf.tests.giacomini_white_test(la, lb, horizon=1)
    assert res.metadata["df"] == 2                     # constant + lagged dL
    assert res.p_value > 0.05 and res.decision is False

def test_gw_rejects_systematic_difference():
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(size=400))
    la = pd.Series(base + 0.8)                          # a is systematically worse
    lb = pd.Series(base)
    res = mf.tests.giacomini_white_test(la, lb, horizon=1)
    assert res.p_value < 0.01 and res.decision is True
    assert res.alternative == "conditional_equal_predictive_ability"


def _h4_fixture() -> tuple[pd.Series, pd.Series]:
    """Fixed-seed h=4 loss differential with a known finite MA(3) dependence
    structure -- the exact design WP-A1's MC size validation used to diagnose
    and fix the small-sample/HAC-kernel behavior of horizon>1 GW tests."""
    rng = np.random.default_rng(42)
    eps_a = rng.normal(size=63)
    eps_b = rng.normal(size=63)
    la = pd.Series(np.convolve(eps_a, np.ones(4), mode="valid") ** 2)
    lb = pd.Series(np.convolve(eps_b, np.ones(4), mode="valid") ** 2)
    return la, lb


def test_gw_h4_small_sample_default_statistic_and_pvalue_pinned():
    """WP-A1: pins the corrected (default small_sample=True) statistic and
    p-value for h>1 on a fixed fixture -- untapered ("acf"-style) HAC with
    PSD-shrink fallback, referenced against F(q, ESS-q) rather than chi2(q).
    See ``giacomini_white_test``'s docstring and
    ``tests/mc/test_giacomini_white_size.py`` for the full diagnosis/MC
    validation this correction is based on.
    """
    la, lb = _h4_fixture()
    res = mf.tests.giacomini_white_test(la, lb, horizon=4)
    assert res.statistic == pytest.approx(4.75009315253637, rel=1e-9)
    assert res.p_value == pytest.approx(0.17386696311948321, rel=1e-9)
    assert res.correction_policy == "acf_hac_small_sample_f"
    assert res.metadata["small_sample"] is True
    assert res.metadata["bandwidth_used"] == 3
    assert res.metadata["reference_distribution"] == "f"
    assert res.metadata["f_df2"] == pytest.approx(6.0)
    assert res.metadata["effective_sample_size"] == pytest.approx(8.0)


def test_gw_h4_small_sample_false_matches_pre_wpa1_bartlett_chi2():
    """``small_sample=False`` must reproduce the exact pre-WP-A1 (Bartlett-
    tapered HAC + chi2(q)) statistic/p-value, for users who need backward-
    compatible p-values."""
    la, lb = _h4_fixture()
    res = mf.tests.giacomini_white_test(la, lb, horizon=4, small_sample=False)
    assert res.statistic == pytest.approx(2.4446326291014664, rel=1e-9)
    assert res.p_value == pytest.approx(0.2945471123825396, rel=1e-9)
    assert res.correction_policy == "newey_west_hac"
    assert res.metadata["small_sample"] is False
    assert res.metadata["reference_distribution"] == "chi2"


def test_gw_h1_small_sample_default_matches_asymptotic_chi2():
    """horizon=1 has bandwidth=0 -- small_sample=True must reduce exactly to
    the original chi2(q) reference (no HAC lag to correct for; WP-V3/WP-A1 MC
    results already show h=1 well-calibrated, and this must not regress it)."""
    rng = np.random.default_rng(0)
    la = pd.Series(np.abs(rng.normal(size=400)))
    lb = pd.Series(np.abs(rng.normal(size=400)))
    res_default = mf.tests.giacomini_white_test(la, lb, horizon=1)
    res_explicit_asymptotic = mf.tests.giacomini_white_test(la, lb, horizon=1, small_sample=False)
    assert res_default.statistic == pytest.approx(res_explicit_asymptotic.statistic, rel=1e-12)
    assert res_default.p_value == pytest.approx(res_explicit_asymptotic.p_value, rel=1e-12)
    assert res_default.metadata["bandwidth_used"] == 0
    assert res_default.metadata["reference_distribution"] == "chi2"
