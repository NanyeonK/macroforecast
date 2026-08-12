"""How ``anatomy_provider(params=...)`` is routed to each model (F-067).

``_resolve_model_params`` chose between "one shared parameter set" and "one
parameter set per model" by asking whether *every* top-level value was a
Mapping. That question is not about the caller's intent, so three distinct
requests were silently rewritten instead of honoured or refused:

* ``params={"init": {"a": 1}}`` -- a single *shared* parameter that happens to
  take a dict value -- looked like per-model routing. No key matched a model
  alias, so every model was fitted with ``{}`` and the parameter vanished.
* ``params={"m1": {...}, "typo": {...}}`` dropped ``typo`` without a word, so a
  misspelled alias silently disabled the parameters the caller asked for.
* ``params={"m1": {"a": 1}, "alpha": 0.5}`` was classified as *shared* because
  ``0.5`` is not a Mapping, so **every** model -- including ``m2`` -- was fitted
  with a keyword argument literally named ``m1``.

The contract pinned here is: per-model routing is detected by *intersection with
the known model aliases*, and once detected every key and value is validated
fail-closed. A key set disjoint from the aliases is shared flat parameters and
reaches every model unchanged. Ambiguity between an alias and a hyperparameter
of the same name is unavoidable; it resolves alias-first, and is reported only
when that collision is detectable -- a non-Mapping colliding value, or non-alias
keys mixed in. An all-alias, all-Mapping request stays indistinguishable from
shared intent and is routed per model without warning.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.interpretation.anatomy import _resolve_model_params

requires_anatomy = pytest.mark.skipif(
    importlib.util.find_spec("anatomy") is None,
    reason="requires optional anatomy backend",
)


class _Zero:
    """Minimal fitted object; the anatomy backend only needs ``predict``."""

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def _specs(*aliases: str) -> dict[str, Any]:
    """Real ``ModelSpec`` objects keyed by the aliases under test."""

    return {
        alias: mf.models.custom_model(f"f067_{alias}", lambda X, y, **kw: _Zero())
        for alias in aliases
    }


def _recording_specs(aliases: tuple[str, ...], calls: dict[str, Any]) -> dict[str, Any]:
    """Specs that record the keyword arguments their fit actually received."""

    def make(alias: str) -> Any:
        def fit(X: Any, y: Any, **kw: Any) -> _Zero:
            calls[alias] = kw
            return _Zero()

        return mf.models.custom_model(f"f067_rec_{alias}", fit)

    return {alias: make(alias) for alias in aliases}


# --------------------------------------------------------------------------
# Shared parameters: disjoint from the aliases, so they reach every model.
# --------------------------------------------------------------------------


def test_nested_shared_param_reaches_every_model_unchanged() -> None:
    """A shared parameter whose value is a Mapping is not per-model routing.

    No key matches a model alias, so there is nothing to route by; the request
    is one parameter named ``init`` shared by both models.
    """

    nested = {"a": 1, "b": [2, 3]}
    resolved = _resolve_model_params(_specs("m1", "m2"), {"init": nested})

    assert resolved == {"m1": {"init": nested}, "m2": {"init": nested}}
    # Passed through, not rebuilt: the callee must see the caller's own object.
    assert resolved["m1"]["init"] is nested
    assert resolved["m2"]["init"] is nested


def test_several_nested_shared_params_all_survive() -> None:
    """Every disjoint key survives, not just the first one."""

    params = {"init": {"a": 1}, "grid": {"b": 2}, "scale": 3.0}
    resolved = _resolve_model_params(_specs("m1", "m2"), params)

    assert resolved["m1"] == params
    assert resolved["m2"] == params


def test_shared_params_are_broadcast_to_a_single_model() -> None:
    """The one-model case takes the same branch as the many-model case."""

    resolved = _resolve_model_params(_specs("only"), {"init": {"a": 1}})

    assert resolved == {"only": {"init": {"a": 1}}}


# --------------------------------------------------------------------------
# Per-model routing: detected by intersection with the known aliases.
# --------------------------------------------------------------------------


def test_alias_keys_route_per_model_and_omitted_aliases_get_empty_params() -> None:
    """One alias key is enough to select routing; the rest default to ``{}``."""

    resolved = _resolve_model_params(
        _specs("m1", "m2", "m3"), {"m1": {"alpha": 1.0}, "m3": {"beta": 2.0}}
    )

    assert resolved == {"m1": {"alpha": 1.0}, "m2": {}, "m3": {"beta": 2.0}}


def test_routed_params_are_copied_not_shared_with_the_caller() -> None:
    """Each model's parameters are a copy, so a later fit cannot edit them."""

    inner = {"alpha": 1.0}
    resolved = _resolve_model_params(_specs("m1", "m2"), {"m1": inner})

    assert resolved["m1"] == inner
    assert resolved["m1"] is not inner


# --------------------------------------------------------------------------
# Fail-closed validation, once routing has been detected.
# --------------------------------------------------------------------------


def test_key_that_is_not_an_alias_is_refused_not_dropped() -> None:
    """A misspelled alias must be reported, never silently discarded.

    Dropping it produced a run whose models were fitted with default
    parameters while the caller believed their parameters had been applied.
    """

    with pytest.raises(ValueError, match="not model aliases"):
        _resolve_model_params(
            _specs("m1", "m2"), {"m1": {"alpha": 1.0}, "typo": {"beta": 2.0}}
        )


def test_mixed_alias_and_flat_key_is_refused_not_broadcast() -> None:
    """The case that fitted every model with a keyword named after a model.

    ``{"m1": {...}, "alpha": 0.5}`` is not a coherent request under either
    reading, so it is refused rather than resolved by guessing.
    """

    with pytest.raises(ValueError, match="not model aliases"):
        _resolve_model_params(_specs("m1", "m2"), {"m1": {"a": 1}, "alpha": 0.5})


def test_non_mapping_value_for_an_alias_is_refused() -> None:
    """Routing was detected, so every value must be a parameter mapping."""

    with pytest.raises(ValueError, match="must be a mapping"):
        _resolve_model_params(_specs("m1", "m2"), {"m1": {"a": 1}, "m2": 0.5})


def test_alias_that_collides_with_a_hyperparameter_name_resolves_alias_first() -> None:
    """The unavoidable ambiguity is reported, with the alias-first rule named.

    A model aliased ``alpha`` makes ``params={"alpha": 0.5}`` ambiguous: it can
    mean "parameters for model ``alpha``" or "shared hyperparameter ``alpha``".
    Aliases win, so the value must be a mapping, and the error has to say so or
    the caller cannot tell why a valid-looking shared parameter was rejected.
    """

    with pytest.raises(ValueError, match="alias"):
        _resolve_model_params(_specs("alpha", "m2"), {"alpha": 0.5})


def test_alias_named_shared_param_with_mapping_value_is_routed_silently() -> None:
    """The irreducible residual cost of alias-first: this collision is silent.

    A model aliased ``init`` makes ``params={"init": {"a": 1}}`` ambiguous, but
    unlike the case above nothing distinguishes the two readings: every key is
    an alias and every value is a ``Mapping``, so routing is well formed and
    there is no mismatch to report. Alias-first therefore routes it per model
    with no warning, and a caller who meant a shared nested parameter has to
    rename the alias or the parameter.
    """

    resolved = _resolve_model_params(_specs("init", "m2"), {"init": {"a": 1}})

    assert resolved == {"init": {"a": 1}, "m2": {}}


def test_non_string_parameter_name_is_refused() -> None:
    """Parameter names become ``**kwargs``, so non-strings fail here, clearly."""

    with pytest.raises(ValueError, match="strings"):
        _resolve_model_params(_specs("m1", "m2"), {"m1": {3: 1.0}})


# --------------------------------------------------------------------------
# Nothing is mutated, and no two models share one parameter dict.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"init": {"a": 1}},
        {"m1": {"alpha": 1.0}},
        {"m1": {"alpha": 1.0}, "m2": {"beta": 2.0}},
        {"scale": 2.0},
        {},
        None,
    ],
    ids=["nested_shared", "routed_one", "routed_both", "flat_shared", "empty", "none"],
)
def test_caller_input_is_never_mutated(params: dict[str, Any] | None) -> None:
    """The caller's mapping is read, never written to."""

    before = copy.deepcopy(params)
    _resolve_model_params(_specs("m1", "m2"), params)

    assert params == before


@pytest.mark.parametrize(
    "params",
    [None, {}, {"scale": 2.0}, {"init": {"a": 1}}, {"m1": {"alpha": 1.0}}],
    ids=["none", "empty", "flat_shared", "nested_shared", "routed"],
)
def test_per_model_outputs_are_not_the_same_object(
    params: dict[str, Any] | None,
) -> None:
    """Editing one model's parameters must not edit another model's."""

    resolved = _resolve_model_params(_specs("m1", "m2"), params)

    assert resolved["m1"] is not resolved["m2"]
    resolved["m1"]["injected"] = True
    assert "injected" not in resolved["m2"]


def test_no_params_gives_every_alias_empty_params() -> None:
    """The documented default: every model is fitted with its own defaults."""

    assert _resolve_model_params(_specs("m1", "m2"), None) == {"m1": {}, "m2": {}}
    assert _resolve_model_params(_specs("m1", "m2"), {}) == {"m1": {}, "m2": {}}


# --------------------------------------------------------------------------
# The public boundary: what actually reaches each model's fit function.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xy_window() -> tuple[pd.DataFrame, pd.Series, Any]:
    """Smallest X/y/window that yields a usable anatomy provider."""

    n_obs = 26
    index = pd.period_range("2000-01", periods=n_obs, freq="M").to_timestamp("M")
    rng = np.random.default_rng(20260812)
    X = pd.DataFrame(
        {"x1": rng.normal(size=n_obs), "x2": rng.normal(size=n_obs)}, index=index
    )
    y = pd.Series(rng.normal(size=n_obs), index=index, name="target")
    window = mf.window.from_cutoffs(
        test_start=X.index[20],
        test_end=X.index[22],
        estimation_min_size=12,
        val_method="last_block",
        val_size=3,
        horizon=1,
        step=1,
    )
    return X, y, window


@requires_anatomy
def test_nested_shared_param_reaches_the_fit_of_every_model(xy_window) -> None:
    """End to end: the shared parameter arrives at both fits, unchanged."""

    import anatomy as anatomy_mod

    from macroforecast.interpretation.anatomy import anatomy_provider

    X, y, window = xy_window
    calls: dict[str, Any] = {}
    nested = {"a": 1, "b": [2, 3]}
    provider = anatomy_provider(
        X,
        y,
        _recording_specs(("m1", "m2"), calls),
        window=window,
        params={"init": nested},
    )

    key = anatomy_mod.AnatomyModelProvider.PeriodKey
    for alias in ("m1", "m2"):
        provider.provider_fn(key(0, alias))
        assert calls[alias] == {"init": nested}
        assert calls[alias]["init"] is nested


@requires_anatomy
def test_per_model_params_reach_only_their_own_model_fit(xy_window) -> None:
    """End to end: routing sends ``alpha`` to ``m1`` alone, not to ``m2``."""

    import anatomy as anatomy_mod

    from macroforecast.interpretation.anatomy import anatomy_provider

    X, y, window = xy_window
    calls: dict[str, Any] = {}
    provider = anatomy_provider(
        X,
        y,
        _recording_specs(("m1", "m2"), calls),
        window=window,
        params={"m1": {"alpha": 1.0}},
    )

    key = anatomy_mod.AnatomyModelProvider.PeriodKey
    provider.provider_fn(key(0, "m1"))
    provider.provider_fn(key(0, "m2"))

    assert calls["m1"] == {"alpha": 1.0}
    assert calls["m2"] == {}


@requires_anatomy
def test_provider_refuses_an_unroutable_params_mapping(xy_window) -> None:
    """The public entry point fails before any model is fitted."""

    from macroforecast.interpretation.anatomy import anatomy_provider

    X, y, window = xy_window
    with pytest.raises(ValueError, match="not model aliases"):
        anatomy_provider(
            X,
            y,
            _specs("m1", "m2"),
            window=window,
            params={"m1": {"a": 1}, "bad": 1.0},
        )
