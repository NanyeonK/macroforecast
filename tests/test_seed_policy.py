from __future__ import annotations

import pytest

from macrocast.execution.seed_policy import (
    ReproducibilityContext,
    VALID_MODES,
    current_seed,
    get_context,
    reset_context,
    resolve_seed,
    set_context,
)


@pytest.fixture(autouse=True)
def _clear_reproducibility_context():
    token = set_context(ReproducibilityContext(recipe_id="", variant_id=None, reproducibility_spec={}))
    try:
        yield
    finally:
        reset_context(token)


def test_valid_modes_enumeration():
    assert VALID_MODES == {
        "strict_reproducible",
        "seeded_reproducible",
        "best_effort",
        "exploratory",
    }


def test_seeded_reproducible_returns_base_seed():
    seed = resolve_seed(
        recipe_id="r",
        reproducibility_spec={"reproducibility_mode": "seeded_reproducible", "seed": 123},
    )
    assert seed == 123


def test_seeded_reproducible_defaults_to_42():
    seed = resolve_seed(
        recipe_id="r",
        reproducibility_spec={"reproducibility_mode": "seeded_reproducible"},
    )
    assert seed == 42


def test_best_effort_behaves_like_seeded():
    seed = resolve_seed(
        recipe_id="r",
        reproducibility_spec={"reproducibility_mode": "best_effort", "seed": 7},
    )
    assert seed == 7


def test_strict_reproducible_is_deterministic():
    spec = {"reproducibility_mode": "strict_reproducible"}
    a = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="rf")
    b = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="rf")
    assert a == b


def test_strict_reproducible_varies_by_variant():
    spec = {"reproducibility_mode": "strict_reproducible"}
    a = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="rf")
    b = resolve_seed(recipe_id="r1", variant_id="v2", reproducibility_spec=spec, model_family="rf")
    assert a != b


def test_strict_reproducible_varies_by_model_family():
    spec = {"reproducibility_mode": "strict_reproducible"}
    a = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="rf")
    b = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="lgbm")
    assert a != b


def test_strict_reproducible_range():
    spec = {"reproducibility_mode": "strict_reproducible"}
    seed = resolve_seed(recipe_id="r1", variant_id="v1", reproducibility_spec=spec, model_family="rf")
    assert isinstance(seed, int)
    assert 0 <= seed <= 0x7FFFFFFF


def test_exploratory_varies_across_calls():
    spec = {"reproducibility_mode": "exploratory"}
    seeds = {resolve_seed(recipe_id="r", reproducibility_spec=spec) for _ in range(10)}
    assert len(seeds) > 1


def test_unknown_mode_raises_valueerror():
    with pytest.raises(ValueError, match="unknown reproducibility_mode"):
        resolve_seed(recipe_id="r", reproducibility_spec={"reproducibility_mode": "bogus"})


def test_current_seed_reads_installed_context():
    ctx = ReproducibilityContext(
        recipe_id="R",
        variant_id="V",
        reproducibility_spec={"reproducibility_mode": "strict_reproducible"},
    )
    token = set_context(ctx)
    try:
        assert get_context() == ctx
        seed_rf = current_seed(model_family="rf")
        seed_lgbm = current_seed(model_family="lgbm")
        assert isinstance(seed_rf, int)
        assert seed_rf != seed_lgbm
    finally:
        reset_context(token)


def test_current_seed_falls_back_without_context():
    empty_ctx = ReproducibilityContext(
        recipe_id="",
        variant_id=None,
        reproducibility_spec={"reproducibility_mode": "seeded_reproducible", "seed": 42},
    )
    token = set_context(empty_ctx)
    try:
        assert current_seed(model_family="rf") == 42
    finally:
        reset_context(token)
