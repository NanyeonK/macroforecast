from macrocast.meta import load_benchmark_registry, validate_benchmark_registry


def test_benchmark_registry_valid() -> None:
    validate_benchmark_registry(load_benchmark_registry())


def test_benchmark_registry_has_family_and_variants() -> None:
    data = load_benchmark_registry()
    assert data['benchmark_families']
    assert data['benchmark_variants']
    assert any(item['id'] == 'ar' for item in data['benchmark_families'])
    assert any(item['id'] == 'ar_bic_expanding' for item in data['benchmark_variants'])
