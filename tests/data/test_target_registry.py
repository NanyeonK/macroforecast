from macrocast.data import (
    get_target_defaults,
    load_target_registry,
    validate_target_registry,
)


def test_target_registry_valid() -> None:
    reg = load_target_registry()
    validate_target_registry(reg)


def test_target_registry_merges_default_and_specific() -> None:
    reg = load_target_registry()
    defaults = get_target_defaults(reg, 'INDPRO', dataset_id='fred_md')
    assert defaults['target'] == 'INDPRO'
    assert defaults['evaluation_scale'] == 'transformed_scale'
    assert defaults['own_lag_policy'] == 'include'


def test_target_registry_requires_evaluation_scale() -> None:
    reg = load_target_registry()
    del reg['targets'][0]['defaults']['evaluation_scale']
    try:
        validate_target_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
