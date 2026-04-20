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


# --- apply_reproducibility_mode — 0.5 global state tightening ---


def test_apply_strict_sets_numpy_global_state():
    import numpy as np
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    apply_reproducibility_mode(mode="strict_reproducible", seed=42, configure_torch=False)
    a = np.random.rand(4)
    apply_reproducibility_mode(mode="strict_reproducible", seed=42, configure_torch=False)
    b = np.random.rand(4)
    assert (a == b).all(), "strict_reproducible must pin numpy global RNG"


def test_apply_seeded_pins_python_random():
    import random
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    apply_reproducibility_mode(mode="seeded_reproducible", seed=7, configure_torch=False)
    a = [random.random() for _ in range(4)]
    apply_reproducibility_mode(mode="seeded_reproducible", seed=7, configure_torch=False)
    b = [random.random() for _ in range(4)]
    assert a == b, "seeded_reproducible must pin Python random.random state"


def test_apply_exploratory_is_noop_on_global_state():
    import random
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    random.seed(99)
    baseline_next = random.random()
    random.seed(99)
    # apply_reproducibility_mode must NOT reset the RNG state
    apply_reproducibility_mode(mode="exploratory", seed=42, configure_torch=False)
    actual_next = random.random()
    assert actual_next == baseline_next, "exploratory mode must not touch Python random state"


def test_apply_strict_warns_without_python_hash_seed():
    import os
    import warnings
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    original = os.environ.pop("PYTHONHASHSEED", None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_reproducibility_mode(mode="strict_reproducible", seed=42, configure_torch=False)
        assert any(
            issubclass(w.category, RuntimeWarning) and "PYTHONHASHSEED" in str(w.message)
            for w in caught
        ), f"expected PYTHONHASHSEED RuntimeWarning; got {[str(w.message) for w in caught]}"
    finally:
        if original is not None:
            os.environ["PYTHONHASHSEED"] = original


def test_apply_strict_sets_cublas_workspace_config():
    import os
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    original = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        summary = apply_reproducibility_mode(mode="strict_reproducible", seed=42, configure_torch=False)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert summary["cublas_workspace_config"] == ":4096:8"
    finally:
        if original is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = original
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_apply_torch_flags_when_torch_available():
    torch = None
    try:
        import torch
    except ImportError:
        pass
    if torch is None:
        import pytest as _pytest
        _pytest.skip("torch not installed")

    from macrocast.execution.seed_policy import apply_reproducibility_mode

    apply_reproducibility_mode(mode="strict_reproducible", seed=42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    t1 = torch.rand(4)
    apply_reproducibility_mode(mode="strict_reproducible", seed=42)
    t2 = torch.rand(4)
    assert torch.equal(t1, t2), "torch tensors must be bit-identical under strict with fixed seed"


def test_apply_unknown_mode_raises():
    import pytest as _pytest
    from macrocast.execution.seed_policy import apply_reproducibility_mode

    with _pytest.raises(ValueError, match="unknown reproducibility_mode"):
        apply_reproducibility_mode(mode="not_a_mode", seed=42, configure_torch=False)
