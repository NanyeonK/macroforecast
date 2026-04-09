from macrocast.pipeline import (
    load_feature_registry,
    load_model_registry,
    validate_feature_model_compatibility,
    validate_feature_registry,
    validate_model_registry,
)


def test_feature_and_model_registries_load_together() -> None:
    f = validate_feature_registry(load_feature_registry())
    m = validate_model_registry(load_model_registry())
    assert f['features']
    assert m['models']


def test_feature_model_compatibility_has_pairs() -> None:
    f = validate_feature_registry(load_feature_registry())
    m = validate_model_registry(load_model_registry())
    pairs = validate_feature_model_compatibility(f, m)
    assert ('factors_x', 'rf') in pairs
    assert ('maf_factors', 'krr') in pairs
