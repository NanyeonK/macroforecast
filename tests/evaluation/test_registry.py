from macrocast.evaluation import load_evaluation_registry, load_test_registry, validate_evaluation_registry, validate_test_registry


def test_evaluation_registry_valid() -> None:
    validate_evaluation_registry(load_evaluation_registry())


def test_test_registry_valid() -> None:
    validate_test_registry(load_test_registry())


def test_evaluation_registry_has_core_entries() -> None:
    reg = load_evaluation_registry()
    ids = {item['id'] for item in reg['metric_suites']}
    assert {'point_default', 'directional_default'}.issubset(ids)


def test_test_registry_has_core_entries() -> None:
    reg = load_test_registry()
    ids = {item['id'] for item in reg['statistical_tests']}
    suite_ids = {item['id'] for item in reg['test_suites']}
    assert {'dm_default', 'cw_default', 'mcs_default'}.issubset(ids)
    assert {'forecast_comparison_default', 'model_set_default'}.issubset(suite_ids)
