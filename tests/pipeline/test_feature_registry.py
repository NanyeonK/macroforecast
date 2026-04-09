from macrocast.pipeline import get_feature_defaults, load_feature_registry, validate_feature_registry


def test_feature_registry_valid() -> None:
    validate_feature_registry(load_feature_registry())


def test_feature_registry_has_core_entries() -> None:
    reg = load_feature_registry()
    ids = {item['id'] for item in reg['features']}
    assert {'factors_x', 'marx_panel', 'maf_factors'}.issubset(ids)


def test_feature_registry_exposes_defaults() -> None:
    reg = load_feature_registry()
    defaults = get_feature_defaults(reg, 'factors_x')
    assert defaults['feature_recipe'] == 'factors_x'
    assert defaults['factor_extraction_recipe'] == 'pca_static'
