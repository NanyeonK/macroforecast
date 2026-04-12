from macrocast.recipes import load_recipe, load_recipe_schema, list_recipe_files, validate_recipe, validate_recipe_schema


def test_recipe_schema_loads_and_validates() -> None:
    schema = load_recipe_schema()
    validate_recipe_schema(schema)


def test_recipe_files_listed() -> None:
    files = list_recipe_files()
    assert 'baselines/minimal_fred_md.yaml' in files


def test_baseline_recipe_validates_against_schema() -> None:
    schema = validate_recipe_schema(load_recipe_schema())
    recipe = load_recipe('baselines/minimal_fred_md.yaml')
    validate_recipe(recipe, schema)
    assert recipe['recipe_id'] == 'minimal_fred_md'
    assert recipe['taxonomy_path']['benchmark'] == 'ar'
