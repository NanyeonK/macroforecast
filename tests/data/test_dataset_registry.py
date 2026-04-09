from macrocast.data import load_dataset_registry, validate_dataset_registry


def test_dataset_registry_valid() -> None:
    reg = load_dataset_registry()
    validate_dataset_registry(reg)


def test_dataset_registry_has_core_fred_sets() -> None:
    reg = load_dataset_registry()
    ids = {item['id'] for item in reg['datasets']}
    assert {'fred_md', 'fred_qd', 'fred_sd'}.issubset(ids)


def test_dataset_registry_defaults_include_task_id() -> None:
    reg = load_dataset_registry()
    for item in reg['datasets']:
        assert 'task_id' in item['defaults']


def test_dataset_registry_rejects_duplicate_ids() -> None:
    reg = load_dataset_registry()
    reg['datasets'].append(dict(reg['datasets'][0]))
    try:
        validate_dataset_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
