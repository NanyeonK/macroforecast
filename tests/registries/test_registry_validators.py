from macrocast.registries import load_registry_bundle, load_registry_layer, validate_registry_bundle, validate_registry_layer


def test_registries_validate_registry_layer() -> None:
    validate_registry_layer('meta', load_registry_layer('meta'))


def test_registries_validate_registry_bundle() -> None:
    validate_registry_bundle(load_registry_bundle())
