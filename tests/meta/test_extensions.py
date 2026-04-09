from macrocast.meta import load_extension_registry, validate_extension_registry


def test_extension_registry_valid() -> None:
    validate_extension_registry(load_extension_registry())


def test_extension_registry_requires_family_fields() -> None:
    reg = load_extension_registry()
    reg['required_fields'].pop('model')
    try:
        validate_extension_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
