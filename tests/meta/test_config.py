from __future__ import annotations

import pytest

from macroforecast import meta
from macroforecast.core.execution import _resolve_seed


def teardown_function() -> None:
    meta.reset_config()


def test_default_random_seed_is_meta_owned():
    assert meta.DEFAULT_RANDOM_SEED == 42
    assert meta.get_config()["random_seed"] == meta.DEFAULT_RANDOM_SEED


def test_configure_updates_global_defaults():
    active = meta.configure(random_seed=7, n_jobs="auto", on_error="continue", verbose=2)

    assert active == {
        "random_seed": 7,
        "n_jobs": "auto",
        "on_error": "continue",
        "verbose": 2,
    }
    assert meta.get_config() == active
    assert meta.get_option("random_seed") == 7


def test_reset_config_restores_defaults():
    meta.configure(random_seed=None, n_jobs=3, on_error="continue", verbose=1)

    assert meta.reset_config() == {
        "random_seed": 42,
        "n_jobs": 1,
        "on_error": "raise",
        "verbose": 0,
    }


def test_use_config_restores_previous_config():
    meta.configure(random_seed=11, n_jobs=2)

    with meta.use_config(random_seed=5, on_error="continue") as active:
        assert active["random_seed"] == 5
        assert active["n_jobs"] == 2
        assert active["on_error"] == "continue"

    assert meta.get_config()["random_seed"] == 11
    assert meta.get_config()["n_jobs"] == 2
    assert meta.get_config()["on_error"] == "raise"


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="random_seed"):
        meta.configure(random_seed=-1)
    with pytest.raises(ValueError, match="n_jobs"):
        meta.configure(n_jobs=0)
    with pytest.raises(ValueError, match="on_error"):
        meta.configure(on_error="skip")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verbose"):
        meta.configure(verbose=-1)


def test_seed_resolution_uses_global_config_without_legacy_block():
    meta.configure(random_seed=123)

    assert _resolve_seed({}) == 123
