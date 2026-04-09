from macrocast.pipeline import get_model_defaults, load_model_registry, validate_model_registry


def test_model_registry_valid() -> None:
    validate_model_registry(load_model_registry())


def test_model_registry_has_core_entries() -> None:
    reg = load_model_registry()
    ids = {item['id'] for item in reg['models']}
    assert {'ar', 'elastic_net', 'rf', 'krr'}.issubset(ids)


def test_model_registry_exposes_defaults() -> None:
    reg = load_model_registry()
    defaults = get_model_defaults(reg, 'rf')
    assert defaults['model_family'] == 'tree_ensemble'
    assert defaults['hyperparameter_space'] == 'rf_default'
