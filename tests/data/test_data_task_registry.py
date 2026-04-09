from macrocast.data import (
    get_data_task_defaults,
    load_data_task_registry,
    validate_data_task_registry,
)


def test_data_task_registry_valid() -> None:
    reg = load_data_task_registry()
    validate_data_task_registry(reg)


def test_data_task_registry_returns_dataset_specific_defaults() -> None:
    reg = load_data_task_registry()
    defaults = get_data_task_defaults(reg, 'fred_md_default', dataset_id='fred_md')
    assert defaults['minimum_train_size'] == 120
    assert defaults['horizon_grid_default'] == [1, 3, 6, 12]


def test_data_task_registry_rejects_missing_required_field() -> None:
    reg = load_data_task_registry()
    del reg['data_tasks'][0]['defaults']['outer_window']
    try:
        validate_data_task_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
