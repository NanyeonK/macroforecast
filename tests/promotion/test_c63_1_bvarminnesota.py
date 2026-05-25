"""Independent validation suite — C63.1 BVARMinnesota prior enforcement.

Tests B1-B8 from test-spec.md Section 4. Written by tester, isolated from
builder's implementation details. Verifies behavioral contracts from
test-spec.md Section 3.

Test IDs map to test-spec.md:
    B1 — Default construction: BVARMinnesota() succeeds
    B2 — Explicit prior='minnesota' succeeds
    B3 — prior='normal_inverse_wishart' raises ValueError with prescribed message
    B4 — Arbitrary invalid prior raises ValueError with prescribed message
    B5 — Other kwargs (lambda1, p) pass through to _BayesianVAR
    B6 — fit/predict pipeline works after construction
    B7 — isinstance backward compat: public -> private True, private -> public False
    B8 — Flat re-export from macroforecast.layers.l4_models works
    I3 — Property invariant: prior enforcement is unconditional
    I4 — Property invariant: BVARMinnesota is always a _BayesianVAR
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.layers.l4_models.bayesian import BVARMinnesota
from macroforecast.core.runtime import _BayesianVAR


# ---------------------------------------------------------------------------
# Shared test fixture (from test-spec.md Section 4)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xy_fixture() -> dict:
    """Small random X, y for fit/predict smoke tests."""
    np.random.seed(0)
    X = pd.DataFrame(
        {
            "x1_lag1": np.random.randn(50),
            "x2_lag1": np.random.randn(50),
        }
    )
    y = pd.Series(np.random.randn(50), name="y")
    return {"X": X, "y": y}


# ---------------------------------------------------------------------------
# B1 — Default construction: no prior kwarg
# ---------------------------------------------------------------------------

def test_B1_default_construction() -> None:
    """B1: BVARMinnesota() constructs without exception."""
    m = BVARMinnesota()
    assert isinstance(m, BVARMinnesota)
    assert isinstance(m, _BayesianVAR)


# ---------------------------------------------------------------------------
# B2 — Explicit prior='minnesota' succeeds
# ---------------------------------------------------------------------------

def test_B2_explicit_minnesota_prior() -> None:
    """B2: BVARMinnesota(prior='minnesota') constructs without exception."""
    m = BVARMinnesota(prior="minnesota")
    assert isinstance(m, BVARMinnesota)
    assert isinstance(m, _BayesianVAR)


# ---------------------------------------------------------------------------
# B3 — prior='normal_inverse_wishart' raises ValueError with full message
# ---------------------------------------------------------------------------

def test_B3_normal_inverse_wishart_raises() -> None:
    """B3: prior='normal_inverse_wishart' raises ValueError with prescribed message."""
    with pytest.raises(ValueError) as exc_info:
        BVARMinnesota(prior="normal_inverse_wishart")

    msg = str(exc_info.value)
    assert "BVARMinnesota requires prior='minnesota'" in msg, (
        f"B3 FAIL: message does not contain required phrase. Got: {msg!r}"
    )
    assert "normal_inverse_wishart" in msg, (
        f"B3 FAIL: message does not contain offending prior value. Got: {msg!r}"
    )
    assert "Use BVAR for arbitrary priors" in msg, (
        f"B3 FAIL: message does not contain usage hint. Got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# B4 — Arbitrary invalid prior raises ValueError with prescribed message
# ---------------------------------------------------------------------------

def test_B4_arbitrary_invalid_prior_raises() -> None:
    """B4: BVARMinnesota(prior='some_other_prior') raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        BVARMinnesota(prior="some_other_prior")

    msg = str(exc_info.value)
    assert "BVARMinnesota requires prior='minnesota'" in msg, (
        f"B4 FAIL: message does not contain required phrase. Got: {msg!r}"
    )
    assert "some_other_prior" in msg, (
        f"B4 FAIL: offending value 'some_other_prior' not in message. Got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# B5 — Other kwargs (lambda1, p) pass through to _BayesianVAR
# ---------------------------------------------------------------------------

def test_B5_other_kwargs_pass_through() -> None:
    """B5: lambda1 and p are stored on the instance (pass-through to _BayesianVAR)."""
    m = BVARMinnesota(lambda1=0.5, p=3)

    # lambda1 may be stored as-is or scaled (NIW path adjusts it by *1.25)
    # For Minnesota prior no scaling occurs: lambda1 == 0.5
    assert hasattr(m, "lambda1"), "B5 FAIL: instance has no attribute 'lambda1'"
    assert m.lambda1 == pytest.approx(0.5), (
        f"B5 FAIL: m.lambda1={m.lambda1}, expected 0.5"
    )
    assert hasattr(m, "p"), "B5 FAIL: instance has no attribute 'p'"
    assert m.p == 3, f"B5 FAIL: m.p={m.p}, expected 3"


# ---------------------------------------------------------------------------
# B6 — fit/predict pipeline works end-to-end
# ---------------------------------------------------------------------------

def test_B6_fit_predict_pipeline(xy_fixture: dict) -> None:
    """B6: fit returns self, predict returns array of shape (50,) with finite values."""
    X = xy_fixture["X"]
    y = xy_fixture["y"]

    m = BVARMinnesota()
    fit_result = m.fit(X, y)

    # fit should return self
    assert fit_result is m, "B6 FAIL: fit() did not return self"

    preds = m.predict(X)

    # Shape: (50,) or (50, 1)
    assert preds.shape[0] == 50, (
        f"B6 FAIL: predict output has shape {preds.shape}, expected 50 rows"
    )
    assert np.all(np.isfinite(preds)), (
        "B6 FAIL: predictions contain NaN or Inf values"
    )


# ---------------------------------------------------------------------------
# B7 — isinstance backward compat: one direction True, other False
# ---------------------------------------------------------------------------

def test_B7_isinstance_backward_compat() -> None:
    """B7: isinstance(BVARMinnesota(), _BayesianVAR) True; reverse False."""
    m_pub = BVARMinnesota()

    # Public instance IS-A private base
    assert isinstance(m_pub, _BayesianVAR), (
        "B7 FAIL: isinstance(BVARMinnesota(), _BayesianVAR) is False"
    )

    # Private base instance is NOT a BVARMinnesota
    m_priv = _BayesianVAR()
    assert not isinstance(m_priv, BVARMinnesota), (
        "B7 FAIL: isinstance(_BayesianVAR(), BVARMinnesota) is True (unexpected)"
    )


# ---------------------------------------------------------------------------
# B8 — Flat re-export from macroforecast.layers.l4_models works
# ---------------------------------------------------------------------------

def test_B8_flat_reexport() -> None:
    """B8: from macroforecast.layers.l4_models import BVARMinnesota works."""
    from macroforecast.layers.l4_models import BVARMinnesota as BVARMinnesota_flat  # noqa: F401

    m = BVARMinnesota_flat()
    assert isinstance(m, _BayesianVAR), (
        "B8 FAIL: flat re-exported BVARMinnesota is not _BayesianVAR"
    )


# ---------------------------------------------------------------------------
# I3 — Property invariant: prior enforcement is unconditional
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_prior", [
    "normal_inverse_wishart",
    "flat",
    "jeffreys",
    "horseshoe",
    "",
    "MINNESOTA",  # case-sensitive check
    "Minnesota",
])
def test_I3_prior_enforcement_unconditional(bad_prior: str) -> None:
    """I3: BVARMinnesota(prior=x) raises ValueError for all x != 'minnesota'."""
    with pytest.raises(ValueError):
        BVARMinnesota(prior=bad_prior)


# ---------------------------------------------------------------------------
# I4 — Property invariant: BVARMinnesota is always a _BayesianVAR
# ---------------------------------------------------------------------------

def test_I4_bvarminnesota_is_always_bayesianvar() -> None:
    """I4: isinstance(BVARMinnesota(), _BayesianVAR) is True for valid construction."""
    # Test with various valid kwargs
    for kwargs in [
        {},
        {"lambda1": 0.1},
        {"lambda1": 0.3, "p": 2},
        {"n_draws": 100},
        {"prior": "minnesota"},
    ]:
        m = BVARMinnesota(**kwargs)
        assert isinstance(m, _BayesianVAR), (
            f"I4 FAIL: isinstance check failed for BVARMinnesota({kwargs})"
        )


# ---------------------------------------------------------------------------
# Additional: internal prior translation check
# ---------------------------------------------------------------------------

def test_internal_prior_translation() -> None:
    """B7 supplement: after __init__, self.prior == 'bvar_minnesota' (translation)."""
    m = BVARMinnesota()
    # _BayesianVAR stores the internal key 'bvar_minnesota' (not 'minnesota')
    assert m.prior == "bvar_minnesota", (
        f"Prior translation FAIL: m.prior={m.prior!r}, expected 'bvar_minnesota'"
    )
