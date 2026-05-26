"""TDD tests for PR9: bvar_minnesota_fit hyperparameter expansion.

Tests four behaviours:
1. Signature accepts new params without error (and result stores them).
2. lambda_cross=0 suppresses cross-lag coefficients vs lambda_cross=1.
3. Backward compatibility: existing call signature produces identical output.
4. b_AR anchors own-lag-1 coefficient toward its value under a tight prior.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.functions import bvar_minnesota_fit


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + 0.5 * rng.normal(size=50)
    return X, y


# ---------------------------------------------------------------------------
# Test 1 — Signature accepts new params; result stores them
# ---------------------------------------------------------------------------

class TestSignatureExpansion:
    """bvar_minnesota_fit accepts lambda_cross, lambda_decay, b_AR."""

    def test_extended_params_no_error(self, _xy):
        X, y = _xy
        result = bvar_minnesota_fit(X, y, lambda_cross=0.3, lambda_decay=2.0, b_AR=0.9)
        assert result.lambda_cross == pytest.approx(0.3)
        assert result.lambda_decay == pytest.approx(2.0)
        assert result.b_AR == pytest.approx(0.9)

    def test_result_stores_lambda1_still_accessible(self, _xy):
        X, y = _xy
        result = bvar_minnesota_fit(X, y, lambda1=0.15, lambda_cross=0.3)
        assert result.lambda1 == pytest.approx(0.15)
        assert result.lambda_cross == pytest.approx(0.3)

    def test_summary_includes_new_fields(self, _xy):
        X, y = _xy
        result = bvar_minnesota_fit(X, y, lambda_cross=0.3, lambda_decay=2.0, b_AR=0.9)
        s = result.summary()
        assert "lambda_cross" in s
        assert "lambda_decay" in s
        assert "b_AR" in s

    def test_validation_lambda_cross_negative_raises(self, _xy):
        X, y = _xy
        with pytest.raises(ValueError, match="lambda_cross"):
            bvar_minnesota_fit(X, y, lambda_cross=-0.1)

    def test_validation_lambda_decay_nonpositive_raises(self, _xy):
        X, y = _xy
        with pytest.raises(ValueError, match="lambda_decay"):
            bvar_minnesota_fit(X, y, lambda_decay=0.0)

    def test_validation_b_AR_out_of_range_raises(self, _xy):
        X, y = _xy
        with pytest.raises(ValueError, match="b_AR"):
            bvar_minnesota_fit(X, y, b_AR=2.5)


# ---------------------------------------------------------------------------
# Test 2 — lambda_cross=0 suppresses cross-lag coefficients
# ---------------------------------------------------------------------------

class TestLambdaCrossEffect:
    """lambda_cross=0 should push cross-lag coefficients toward zero."""

    def test_lambda_cross_zero_suppresses_cross_lags(self, _xy):
        X, y = _xy
        result_no_cross = bvar_minnesota_fit(X, y, n_lag=2, lambda_cross=0.0)
        result_with_cross = bvar_minnesota_fit(X, y, n_lag=2, lambda_cross=1.0)

        B_no = result_no_cross._model._results._B
        B_with = result_with_cross._model._results._B

        K = B_no.shape[0]
        # lag-1 columns: indices 1..K (column 0 is intercept)
        cross_lag_mismatch = 0
        for eq in range(K):
            for col in range(1, 1 + K):
                if col - 1 != eq:  # cross-lag coefficient
                    no_cross_abs = abs(B_no[eq, col])
                    with_cross_abs = abs(B_with[eq, col])
                    if no_cross_abs >= with_cross_abs + 1e-6:
                        cross_lag_mismatch += 1

        assert cross_lag_mismatch == 0, (
            f"lambda_cross=0 did not suppress all cross-lag coefficients; "
            f"{cross_lag_mismatch} violations found."
        )


# ---------------------------------------------------------------------------
# Test 3 — Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing call pattern produces identical results after signature expansion."""

    def test_default_call_unchanged(self, _xy):
        X, y = _xy
        r1 = bvar_minnesota_fit(X, y)
        r2 = bvar_minnesota_fit(X, y, lambda_cross=0.5, lambda_decay=1.0, b_AR=1.0)

        np.testing.assert_allclose(
            r1.predict(X), r2.predict(X), rtol=1e-10,
            err_msg="Default call should be identical to explicit defaults."
        )
        assert r1.lambda1 == r2.lambda1
        assert r1.n_lag == r2.n_lag
        assert r1.n_obs == r2.n_obs

    def test_predict_shape_unchanged(self, _xy):
        X, y = _xy
        r = bvar_minnesota_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_two_arg_call_still_works(self, _xy):
        X, y = _xy
        r = bvar_minnesota_fit(X, y)
        assert r.n_lag == 1
        assert r.lambda1 == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Test 4 — b_AR anchors own-lag-1 toward its value under tight prior
# ---------------------------------------------------------------------------

class TestBARAnchor:
    """Under a tight prior, own-lag-1 coefficient should be close to b_AR."""

    def test_b_AR_1_anchors_own_lag1_near_one(self, _xy):
        X, y = _xy
        result_rw = bvar_minnesota_fit(X, y, lambda1=0.001, b_AR=1.0)
        B_rw = result_rw._model._results._B
        # Own-lag-1 for equation 0 (column 1 in _B, which is intercept-inclusive)
        own_lag1 = B_rw[0, 1]
        assert abs(own_lag1 - 1.0) < 0.1, (
            f"b_AR=1.0 (tight prior) did not anchor own-lag-1 near 1.0: got {own_lag1:.4f}"
        )

    def test_b_AR_0_anchors_own_lag1_near_zero(self, _xy):
        X, y = _xy
        result_wn = bvar_minnesota_fit(X, y, lambda1=0.001, b_AR=0.0)
        B_wn = result_wn._model._results._B
        own_lag1 = B_wn[0, 1]
        assert abs(own_lag1 - 0.0) < 0.1, (
            f"b_AR=0.0 (tight prior) did not anchor own-lag-1 near 0.0: got {own_lag1:.4f}"
        )

    def test_b_AR_rw_vs_wn_differ(self, _xy):
        X, y = _xy
        result_rw = bvar_minnesota_fit(X, y, lambda1=0.001, b_AR=1.0)
        result_wn = bvar_minnesota_fit(X, y, lambda1=0.001, b_AR=0.0)

        B_rw = result_rw._model._results._B
        B_wn = result_wn._model._results._B

        # The two posterior means must be meaningfully different
        max_diff = np.max(np.abs(B_rw - B_wn))
        assert max_diff > 0.05, (
            f"b_AR=1 and b_AR=0 produced nearly identical coefficients "
            f"(max |diff|={max_diff:.6f}); prior is not propagating."
        )
