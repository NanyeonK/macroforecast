from macrocast.specs.compiler import compile_experiment_spec_from_recipe


def test_compile_experiment_spec_from_recipe_baseline() -> None:
    compiled = compile_experiment_spec_from_recipe('baselines/minimal_fred_md.yaml', preset_id='researcher_explicit')
    assert compiled.meta_config['dataset'] == 'fred_md'
    assert compiled.meta_config['target'] == 'INDPRO'
    assert compiled.meta_config['horizon'] == [1]
    assert compiled.meta_config['benchmark_family'] == 'ar'
    assert compiled.meta_config['benchmark_id'] == 'ar_bic_expanding'
    assert compiled.meta_config['recipe_id'] == 'minimal_fred_md'
    assert compiled.meta_config['taxonomy_path']['model'] == 'random_forest'
    assert compiled.meta_config['model_family'] == 'tree_ensemble'
