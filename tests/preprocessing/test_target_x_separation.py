from macrocast.preprocessing import load_preprocessing_registry, validate_preprocessing_registry


def test_preprocessing_policy_requires_target_x_separation() -> None:
    reg = load_preprocessing_registry()
    assert reg['preprocessing']['policies']['target_x_must_be_separate'] is True
    validate_preprocessing_registry(reg)


def test_invalid_family_is_rejected() -> None:
    reg = load_preprocessing_registry()
    reg['preprocessing']['x_recipes'][0]['family'] = 'target'
    try:
        validate_preprocessing_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_learned_recipe_requires_train_window_only_fit_scope() -> None:
    reg = load_preprocessing_registry()
    reg['preprocessing']['x_recipes'][0]['fit_scope'] = 'none'
    try:
        validate_preprocessing_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_target_and_x_recipe_keys_must_differ() -> None:
    reg = load_preprocessing_registry()
    reg['preprocessing']['policies']['x_recipe_key'] = reg['preprocessing']['policies']['target_recipe_key']
    try:
        validate_preprocessing_registry(reg)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
