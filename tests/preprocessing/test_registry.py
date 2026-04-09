from macrocast.preprocessing import get_target_recipe, get_x_recipe, load_preprocessing_registry, validate_preprocessing_registry


def test_preprocessing_registry_valid() -> None:
    reg = load_preprocessing_registry()
    validate_preprocessing_registry(reg)


def test_preprocessing_registry_contains_target_and_x_recipes() -> None:
    reg = load_preprocessing_registry()
    root = reg['preprocessing']
    assert root['target_recipes']
    assert root['x_recipes']


def test_preprocessing_registry_recipe_lookup_is_separate() -> None:
    reg = load_preprocessing_registry()
    assert get_target_recipe(reg, 'basic_none')['family'] == 'target'
    assert get_x_recipe(reg, 'basic_none')['family'] == 'x'
