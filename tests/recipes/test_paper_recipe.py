from macrocast.recipes import load_recipe, load_recipe_schema, validate_recipe, validate_recipe_schema


def test_clss_paper_recipe_validates() -> None:
    schema = validate_recipe_schema(load_recipe_schema())
    recipe = validate_recipe(load_recipe('papers/clss2021.yaml'), schema)
    assert recipe['recipe_id'] == 'clss2021'
    assert recipe['kind'] == 'paper'
    assert recipe['taxonomy_path']['benchmark'] == 'ar'
    assert recipe['benchmark_options']['lag_selection_rule'] == 'bic'
