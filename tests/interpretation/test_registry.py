from macrocast.interpretation.registry import load_interpretation_registry, validate_interpretation_registry


def test_interpretation_registry_valid() -> None:
    validate_interpretation_registry(load_interpretation_registry())


def test_interpretation_registry_has_core_entries() -> None:
    reg = load_interpretation_registry()
    ids = {item['id'] for item in reg['interpretation_methods']}
    assert {'tree_native_importance', 'permutation_importance_table'}.issubset(ids)


def test_interpretation_registry_requires_allowed_model_families() -> None:
    reg = load_interpretation_registry()
    del reg['interpretation_methods'][0]['compatibility_rules']['allowed_model_families']
    try:
        validate_interpretation_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
