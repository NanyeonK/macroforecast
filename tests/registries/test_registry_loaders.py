from macrocast.registries import load_registry_bundle, load_registry_file, load_registry_layer


def test_registries_load_registry_file() -> None:
    data = load_registry_file('meta/global_defaults.yaml')
    assert data['registry']['id'] == 'global_defaults'


def test_registries_load_registry_layer() -> None:
    bundle = load_registry_layer('data')
    assert 'datasets' in bundle


def test_registries_load_registry_bundle() -> None:
    bundle = load_registry_bundle()
    assert set(bundle) == {'meta', 'data', 'training', 'evaluation', 'output'}
