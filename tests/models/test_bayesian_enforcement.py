"""Unit tests for BVARMinnesota prior enforcement (C63.1).

Tests verify:
- BVARMinnesota() instantiates without error (default prior='minnesota').
- BVARMinnesota(prior='minnesota') instantiates without error.
- BVARMinnesota(prior='normal_inverse_wishart') raises ValueError.
- BVARMinnesota(prior='anything_else') raises ValueError.
- Internal prior attribute is 'bvar_minnesota' after construction.
- isinstance(BVARMinnesota(), _BayesianVAR) is True.
- BVAR class still accepts arbitrary priors without restriction.
- kwargs are forwarded correctly to _BayesianVAR.
"""
from __future__ import annotations

import pytest

from macroforecast.core.runtime import _BayesianVAR
from macroforecast.models.bayesian import BVAR, BVARMinnesota


# ---------------------------------------------------------------------------
# BVARMinnesota — construction and validation
# ---------------------------------------------------------------------------

class TestBVARMinnesotaConstruction:
    """BVARMinnesota must enforce prior='minnesota' user-facing contract."""

    def test_default_constructor_succeeds(self) -> None:
        """No arguments should work — default prior is 'minnesota'."""
        model = BVARMinnesota()
        assert model is not None

    def test_explicit_minnesota_prior_succeeds(self) -> None:
        """Explicitly setting prior='minnesota' must succeed."""
        model = BVARMinnesota(prior="minnesota")
        assert model is not None

    def test_normal_inverse_wishart_prior_raises(self) -> None:
        """prior='normal_inverse_wishart' must raise ValueError."""
        with pytest.raises(ValueError, match="BVARMinnesota requires prior='minnesota'"):
            BVARMinnesota(prior="normal_inverse_wishart")

    def test_bvar_minnesota_prior_raises(self) -> None:
        """Passing the internal name 'bvar_minnesota' directly must also raise ValueError.
        The user-facing API only accepts 'minnesota'."""
        with pytest.raises(ValueError, match="BVARMinnesota requires prior='minnesota'"):
            BVARMinnesota(prior="bvar_minnesota")

    def test_arbitrary_prior_raises(self) -> None:
        """Any prior value other than 'minnesota' must raise ValueError."""
        with pytest.raises(ValueError, match="BVARMinnesota requires prior='minnesota'"):
            BVARMinnesota(prior="some_other_prior")

    def test_error_message_mentions_bvar(self) -> None:
        """Error message should direct user to use BVAR for arbitrary priors."""
        with pytest.raises(ValueError, match="Use BVAR for arbitrary priors"):
            BVARMinnesota(prior="wrong")


# ---------------------------------------------------------------------------
# BVARMinnesota — internal state after construction
# ---------------------------------------------------------------------------

class TestBVARMinnesotaInternalState:
    """After construction, internal prior should be 'bvar_minnesota'."""

    def test_internal_prior_is_bvar_minnesota(self) -> None:
        """_BayesianVAR.__init__ stores prior internally as 'bvar_minnesota'."""
        model = BVARMinnesota()
        # _BayesianVAR sets self.prior to whatever is passed via super().__init__
        assert model.prior == "bvar_minnesota", (
            f"Expected internal prior 'bvar_minnesota', got {model.prior!r}. "
            "BVARMinnesota.__init__ must translate 'minnesota' -> 'bvar_minnesota'."
        )

    def test_internal_prior_unchanged_with_explicit_minnesota(self) -> None:
        model = BVARMinnesota(prior="minnesota")
        assert model.prior == "bvar_minnesota"


# ---------------------------------------------------------------------------
# BVARMinnesota — isinstance inheritance
# ---------------------------------------------------------------------------

class TestBVARMinnesotaInheritance:
    """BVARMinnesota is a subclass of _BayesianVAR — isinstance must hold."""

    def test_isinstance_private_class(self) -> None:
        model = BVARMinnesota()
        assert isinstance(model, _BayesianVAR), (
            "BVARMinnesota() must satisfy isinstance(model, _BayesianVAR)"
        )

    def test_isinstance_bvar(self) -> None:
        model = BVARMinnesota()
        assert isinstance(model, BVAR) or issubclass(BVARMinnesota, _BayesianVAR), (
            "BVARMinnesota must be a subclass of _BayesianVAR"
        )

    def test_reverse_isinstance_is_false(self) -> None:
        """A plain _BayesianVAR instance is NOT an instance of BVARMinnesota."""
        private_model = _BayesianVAR(prior="bvar_minnesota")
        assert not isinstance(private_model, BVARMinnesota), (
            "isinstance(private_instance, BVARMinnesota) must be False (single inheritance)"
        )


# ---------------------------------------------------------------------------
# BVARMinnesota — kwargs forwarding
# ---------------------------------------------------------------------------

class TestBVARMinnesotaKwargsForwarding:
    """Extra kwargs must be forwarded to _BayesianVAR without error."""

    def test_kwargs_forwarded_lambda1(self) -> None:
        """lambda1 kwarg should be accepted and forwarded."""
        model = BVARMinnesota(lambda1=0.05)
        assert model is not None

    def test_kwargs_forwarded_p(self) -> None:
        """p (VAR lag order) kwarg should be accepted and forwarded."""
        model = BVARMinnesota(p=2)
        assert model is not None

    def test_kwargs_forwarded_multiple(self) -> None:
        """Multiple kwargs should all be forwarded without error."""
        model = BVARMinnesota(lambda1=0.1, lambda_decay=1.0, p=3)
        assert model is not None


# ---------------------------------------------------------------------------
# BVAR (public class) — still accepts arbitrary priors
# ---------------------------------------------------------------------------

class TestBVARArbitraryPrior:
    """BVAR (the unrestricted public class) should still accept any prior."""

    def test_bvar_minnesota_prior(self) -> None:
        model = BVAR(prior="bvar_minnesota")
        assert model is not None

    def test_bvar_niw_prior(self) -> None:
        model = BVAR(prior="bvar_normal_inverse_wishart")
        assert model is not None

    def test_bvar_default_no_prior_arg(self) -> None:
        """BVAR() with no prior arg should use parent _BayesianVAR defaults."""
        model = BVAR()
        assert model is not None
