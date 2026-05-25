"""Tests for Cycle 29 L6 statistical test standalone callables.

Each test class covers one new function in ``mf.functions``.
Bit-exact assertions compare standalone callables against the runtime
primitive directly imported from ``macroforecast.core.runtime``.

RNG seed 42 throughout for reproducibility.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.core.runtime import (
    _diebold_mariano_test,
    _harvey_newbold_test,
    _long_run_variance,
)


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loss_arrays_100():
    """100-element loss arrays from seed 42."""
    rng = np.random.RandomState(42)
    y = rng.randn(100)
    loss_a = (y - rng.randn(100)) ** 2
    loss_b = (y - rng.randn(100)) ** 2
    return loss_a, loss_b


@pytest.fixture(scope="module")
def error_arrays_100():
    """100-element forecast error arrays from seed 42."""
    rng = np.random.RandomState(42)
    y = rng.randn(100)
    e_a = y - rng.randn(100)
    e_b = y - rng.randn(100)
    return e_a, e_b


@pytest.fixture(scope="module")
def nested_arrays_100():
    """100-element arrays for nested model tests from seed 42."""
    rng = np.random.RandomState(42)
    y = rng.randn(100)
    f_s = rng.randn(100)
    f_l = rng.randn(100)
    loss_s = (y - f_s) ** 2
    loss_l = (y - f_l) ** 2
    return loss_s, loss_l, f_s, f_l


# ---------------------------------------------------------------------------
# TestDmTest
# ---------------------------------------------------------------------------

class TestDmTest:
    def test_bit_exact_vs_runtime(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b, horizon=1)
        diff = pd.Series(loss_a - loss_b)
        stat_rt, pvalue_rt = _diebold_mariano_test(diff, horizon=1, hln=True, kernel="newey_west")
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14), \
            f"stat mismatch: {result.stat} vs {stat_rt}"
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14), \
            f"pvalue mismatch: {result.pvalue} vs {pvalue_rt}"

    def test_bit_exact_hln_false(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b, correction="none")
        diff = pd.Series(loss_a - loss_b)
        stat_rt, pvalue_rt = _diebold_mariano_test(diff, horizon=1, hln=False, kernel="newey_west")
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14)

    def test_returns_dmtestresult(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b)
        assert isinstance(result, mf.functions.DMTestResult)
        assert result.alternative == "two_sided"
        assert result.hln_correction is True
        assert result.correction_policy == "hln_nw"

    def test_frozen_dataclass(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b)
        with pytest.raises((AttributeError, TypeError)):
            result.stat = 99.0  # type: ignore

    def test_summary_content(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b)
        s = result.summary()
        assert "Diebold-Mariano" in s
        assert "stat=" in s.lower() or "Statistic" in s
        assert "p=" in s.lower() or "P-value" in s
        assert "decision" in s.lower() or "Decision" in s

    def test_n_obs_field(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b)
        assert result.n_obs == 100

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.dm_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.dm_test(np.array([]), np.array([]))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            mf.functions.dm_test(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))

    def test_horizon_zero_raises(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        with pytest.raises(ValueError, match="horizon"):
            mf.functions.dm_test(loss_a, loss_b, horizon=0)

    def test_small_n_returns_none_stat(self):
        """n < 3 returns None stat after NaN filtering."""
        loss_a = np.array([1.0, np.nan])
        loss_b = np.array([1.1, 1.2])
        result = mf.functions.dm_test(loss_a, loss_b)
        assert result.stat is None
        assert result.pvalue is None
        assert result.decision is False

    def test_andrews_kernel(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b, kernel="andrews")
        diff = pd.Series(loss_a - loss_b)
        stat_rt, pvalue_rt = _diebold_mariano_test(diff, horizon=1, hln=True, kernel="andrews")
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)

    def test_accepts_series(self):
        rng = np.random.RandomState(7)
        loss_a = pd.Series(rng.rand(50))
        loss_b = pd.Series(rng.rand(50))
        result = mf.functions.dm_test(loss_a, loss_b)
        assert isinstance(result, mf.functions.DMTestResult)

    def test_horizon_multiperiod(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.dm_test(loss_a, loss_b, horizon=4)
        diff = pd.Series(loss_a - loss_b)
        stat_rt, pvalue_rt = _diebold_mariano_test(diff, horizon=4, hln=True, kernel="newey_west")
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# TestGwTest
# ---------------------------------------------------------------------------

class TestGwTest:
    def test_bit_exact_vs_runtime(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.gw_test(loss_a, loss_b)
        diff = pd.Series(loss_a - loss_b)
        stat_rt, pvalue_rt = _diebold_mariano_test(diff, horizon=1, hln=True, kernel="newey_west")
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14)

    def test_identical_computation_to_dm(self, loss_arrays_100):
        """gw_test and dm_test are numerically identical (same runtime primitive)."""
        loss_a, loss_b = loss_arrays_100
        dm = mf.functions.dm_test(loss_a, loss_b)
        gw = mf.functions.gw_test(loss_a, loss_b)
        assert dm.stat == gw.stat
        assert dm.pvalue == gw.pvalue
        assert dm.decision == gw.decision

    def test_returns_gwtestresult(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        result = mf.functions.gw_test(loss_a, loss_b)
        assert isinstance(result, mf.functions.GWTestResult)
        assert result.alternative == "two_sided"

    def test_summary_header(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        s = mf.functions.gw_test(loss_a, loss_b).summary()
        assert "Giacomini-White" in s

    def test_horizon_zero_raises(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        with pytest.raises(ValueError, match="horizon"):
            mf.functions.gw_test(loss_a, loss_b, horizon=0)


# ---------------------------------------------------------------------------
# TestDmpTest
# ---------------------------------------------------------------------------

class TestDmpTest:
    def test_bit_exact_list_input(self):
        """Bit-exact vs manual computation using _long_run_variance."""
        from scipy import stats as ss
        rng = np.random.RandomState(42)
        diffs = [rng.randn(50) * 0.1 for _ in range(4)]
        result = mf.functions.dmp_test(diffs)

        # Manual computation
        stacked = np.concatenate(diffs).astype(float)
        stacked = stacked[np.isfinite(stacked)]
        n = len(stacked)
        mean_diff = float(stacked.mean())
        lr_var = _long_run_variance(stacked - mean_diff)
        se = float(math.sqrt(max(lr_var / n, 1e-12)))
        stat_expected = mean_diff / se if se > 0 else 0.0
        pvalue_expected = float(2 * (1 - ss.norm.cdf(abs(stat_expected))))

        assert np.isclose(result.stat, stat_expected, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_expected, rtol=1e-12, atol=1e-14)

    def test_pre_stacked_array(self):
        rng = np.random.RandomState(42)
        diffs_list = [rng.randn(50) * 0.1 for _ in range(4)]
        stacked = np.concatenate(diffs_list)
        result_list = mf.functions.dmp_test(diffs_list)
        result_arr = mf.functions.dmp_test(stacked)
        # Both should give the same stat since the stacked array is the same
        assert np.isclose(result_list.stat, result_arr.stat, rtol=1e-12, atol=1e-14)

    def test_returns_dmptestresult(self):
        rng = np.random.RandomState(42)
        result = mf.functions.dmp_test([rng.randn(30)])
        assert isinstance(result, mf.functions.DMPTestResult)
        assert result.alternative == "two_sided"
        assert result.correction_policy == "nw"
        assert result.horizon is None

    def test_summary_content(self):
        rng = np.random.RandomState(42)
        result = mf.functions.dmp_test([rng.randn(30)])
        s = result.summary()
        assert "Diebold-Mariano-Pesaran" in s
        assert "joint" in s.lower() or "stacked" in s.lower() or "Joint" in s or "Stacked" in s

    def test_small_n_returns_none_stat(self):
        result = mf.functions.dmp_test([np.array([1.0, np.nan])])
        assert result.stat is None
        assert result.pvalue is None
        assert result.decision is False

    def test_empty_list_small_n(self):
        result = mf.functions.dmp_test([np.array([])])
        assert result.stat is None

    def test_decision_field(self):
        rng = np.random.RandomState(42)
        # Craft differentials that are clearly non-zero
        diffs = [np.ones(100) * 5.0]
        result = mf.functions.dmp_test(diffs)
        assert result.decision is True

    def test_n_obs_stacked(self):
        rng = np.random.RandomState(42)
        diffs = [rng.randn(50) for _ in range(3)]
        result = mf.functions.dmp_test(diffs)
        assert result.n_obs_stacked == 150


# ---------------------------------------------------------------------------
# TestHnTest
# ---------------------------------------------------------------------------

class TestHnTest:
    def test_bit_exact_vs_runtime(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        result = mf.functions.hn_test(e_a, e_b, horizon=1)
        stat_rt, pvalue_rt = _harvey_newbold_test(e_a, e_b, horizon=1, small_sample=True)
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14), \
            f"stat mismatch: {result.stat} vs {stat_rt}"
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14), \
            f"pvalue mismatch: {result.pvalue} vs {pvalue_rt}"

    def test_bit_exact_small_sample_false(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        result = mf.functions.hn_test(e_a, e_b, small_sample=False)
        stat_rt, pvalue_rt = _harvey_newbold_test(e_a, e_b, horizon=1, small_sample=False)
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)

    def test_bit_exact_horizon_4(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        result = mf.functions.hn_test(e_a, e_b, horizon=4)
        stat_rt, pvalue_rt = _harvey_newbold_test(e_a, e_b, horizon=4, small_sample=True)
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)

    def test_returns_hntestresult(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        result = mf.functions.hn_test(e_a, e_b)
        assert isinstance(result, mf.functions.HNTestResult)
        assert result.alternative == "one_sided"
        assert result.correction_policy == "hln_nw"
        assert result.encompassing == "a_over_b"

    def test_summary_content(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        s = mf.functions.hn_test(e_a, e_b).summary()
        assert "Harvey-Newbold" in s
        assert "one_sided" in s or "one-sided" in s.lower() or "one_sided" in s

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.hn_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.hn_test(np.array([]), np.array([]))

    def test_horizon_zero_raises(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        with pytest.raises(ValueError, match="horizon"):
            mf.functions.hn_test(e_a, e_b, horizon=0)

    def test_small_n_returns_none_stat(self):
        """n < 5 after finite filter -> None stat."""
        e_a = np.array([1.0, np.nan, np.nan])
        e_b = np.array([1.1, 1.2, 1.3])
        result = mf.functions.hn_test(e_a, e_b)
        assert result.stat is None
        assert result.pvalue is None
        assert result.decision is False

    def test_accepts_series(self, error_arrays_100):
        e_a, e_b = error_arrays_100
        result = mf.functions.hn_test(pd.Series(e_a), pd.Series(e_b))
        assert isinstance(result, mf.functions.HNTestResult)


# ---------------------------------------------------------------------------
# TestCwTest
# ---------------------------------------------------------------------------

class TestCwTest:
    def test_bit_exact_vs_runtime(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        result = mf.functions.cw_test(loss_s, loss_l, f_s, f_l)
        # Reconstruct manually using _diebold_mariano_test
        improvement = loss_s - loss_l
        adjustment = (f_s - f_l) ** 2
        f_value = pd.Series(improvement + adjustment)
        stat_rt, p_two_rt = _diebold_mariano_test(f_value, horizon=1, hln=False, kernel="newey_west")
        if p_two_rt is not None and stat_rt is not None and stat_rt > 0:
            pvalue_rt = p_two_rt / 2.0
        else:
            pvalue_rt = p_two_rt
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14)

    def test_returns_cwtestresult(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        result = mf.functions.cw_test(loss_s, loss_l, f_s, f_l)
        assert isinstance(result, mf.functions.CWTestResult)
        assert result.alternative == "one_sided"
        assert result.correction_policy == "nw"
        assert result.cw_adjustment is True

    def test_summary_content(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        s = mf.functions.cw_test(loss_s, loss_l, f_s, f_l).summary()
        assert "Clark-West" in s
        assert "one_sided" in s or "one-sided" in s.lower() or "one_sided" in s

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.cw_test(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.cw_test(
                np.array([]), np.array([]), np.array([]), np.array([])
            )

    def test_horizon_zero_raises(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        with pytest.raises(ValueError, match="horizon"):
            mf.functions.cw_test(loss_s, loss_l, f_s, f_l, horizon=0)

    def test_small_n_returns_none_stat(self):
        loss_s = np.array([1.0, np.nan])
        loss_l = np.array([1.1, 1.2])
        f_s = np.array([0.9, 1.0])
        f_l = np.array([1.0, 1.1])
        result = mf.functions.cw_test(loss_s, loss_l, f_s, f_l)
        assert result.stat is None

    def test_n_obs_field(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        result = mf.functions.cw_test(loss_s, loss_l, f_s, f_l)
        assert result.n_obs == 100


# ---------------------------------------------------------------------------
# TestEncNewTest
# ---------------------------------------------------------------------------

class TestEncNewTest:
    def test_bit_exact_vs_runtime(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        result = mf.functions.enc_new_test(loss_s, loss_l)
        f_value = pd.Series(loss_s - loss_l)
        stat_rt, p_two_rt = _diebold_mariano_test(f_value, horizon=1, hln=False, kernel="newey_west")
        if p_two_rt is not None and stat_rt is not None and stat_rt > 0:
            pvalue_rt = p_two_rt / 2.0
        else:
            pvalue_rt = p_two_rt
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14)

    def test_returns_encnewtestresult(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        result = mf.functions.enc_new_test(loss_s, loss_l)
        assert isinstance(result, mf.functions.EncNewTestResult)
        assert result.alternative == "one_sided"
        assert result.correction_policy == "nw"

    def test_summary_header(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        s = mf.functions.enc_new_test(loss_s, loss_l).summary()
        assert "Enc-New" in s

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.enc_new_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_small_n_returns_none_stat(self):
        result = mf.functions.enc_new_test(np.array([1.0, np.nan]), np.array([1.1, 1.2]))
        assert result.stat is None


# ---------------------------------------------------------------------------
# TestEncTTest
# ---------------------------------------------------------------------------

class TestEncTTest:
    def test_bit_exact_vs_runtime(self, nested_arrays_100):
        loss_s, loss_l, f_s, f_l = nested_arrays_100
        result = mf.functions.enc_t_test(loss_s, loss_l)
        f_value = pd.Series(loss_s - loss_l)
        stat_rt, p_two_rt = _diebold_mariano_test(f_value, horizon=1, hln=False, kernel="newey_west")
        if p_two_rt is not None and stat_rt is not None and stat_rt > 0:
            pvalue_rt = p_two_rt / 2.0
        else:
            pvalue_rt = p_two_rt
        assert np.isclose(result.stat, stat_rt, rtol=1e-12, atol=1e-14)
        assert np.isclose(result.pvalue, pvalue_rt, rtol=1e-12, atol=1e-14)

    def test_identical_computation_to_enc_new(self, nested_arrays_100):
        """enc_t and enc_new share identical computation."""
        loss_s, loss_l, _, _ = nested_arrays_100
        r_new = mf.functions.enc_new_test(loss_s, loss_l)
        r_t = mf.functions.enc_t_test(loss_s, loss_l)
        assert r_new.stat == r_t.stat
        assert r_new.pvalue == r_t.pvalue
        assert r_new.decision == r_t.decision

    def test_returns_enctestresult(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        result = mf.functions.enc_t_test(loss_s, loss_l)
        assert isinstance(result, mf.functions.EncTTestResult)
        assert result.alternative == "one_sided"

    def test_summary_header(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        s = mf.functions.enc_t_test(loss_s, loss_l).summary()
        assert "Enc-T" in s

    def test_horizon_zero_raises(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        with pytest.raises(ValueError, match="horizon"):
            mf.functions.enc_t_test(loss_s, loss_l, horizon=0)


# ---------------------------------------------------------------------------
# Cross-op integration checks
# ---------------------------------------------------------------------------

class TestCrossOpInvariants:
    def test_dm_gw_numerically_identical(self, loss_arrays_100):
        loss_a, loss_b = loss_arrays_100
        dm = mf.functions.dm_test(loss_a, loss_b)
        gw = mf.functions.gw_test(loss_a, loss_b)
        assert dm.stat == gw.stat
        assert dm.pvalue == gw.pvalue

    def test_enc_new_enc_t_numerically_identical(self, nested_arrays_100):
        loss_s, loss_l, _, _ = nested_arrays_100
        enc_new = mf.functions.enc_new_test(loss_s, loss_l)
        enc_t = mf.functions.enc_t_test(loss_s, loss_l)
        assert enc_new.stat == enc_t.stat
        assert enc_new.pvalue == enc_t.pvalue

    def test_all_7_callables_importable(self):
        assert callable(mf.functions.dm_test)
        assert callable(mf.functions.gw_test)
        assert callable(mf.functions.dmp_test)
        assert callable(mf.functions.hn_test)
        assert callable(mf.functions.cw_test)
        assert callable(mf.functions.enc_new_test)
        assert callable(mf.functions.enc_t_test)

    def test_all_7_result_classes_importable(self):
        assert mf.functions.DMTestResult
        assert mf.functions.GWTestResult
        assert mf.functions.DMPTestResult
        assert mf.functions.HNTestResult
        assert mf.functions.CWTestResult
        assert mf.functions.EncNewTestResult
        assert mf.functions.EncTTestResult

    def test_mf_functions_all_contains_14_names(self):
        all_names = mf.functions.__all__
        test_names = [n for n in all_names if "test" in n.lower() or "Test" in n]
        assert len(test_names) == 14, f"Expected 14 test names in __all__, got {len(test_names)}: {test_names}"
