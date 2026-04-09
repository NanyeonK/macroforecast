from macrocast.meta import (
    load_axes_registry,
    load_benchmark_registry,
    load_execution_policy_registry,
    load_extension_registry,
    load_naming_policy,
    load_preset_registry,
)


def test_meta_loaders_return_dicts() -> None:
    for loader in [
        load_axes_registry,
        load_benchmark_registry,
        load_execution_policy_registry,
        load_extension_registry,
        load_naming_policy,
        load_preset_registry,
    ]:
        assert isinstance(loader(), dict)
