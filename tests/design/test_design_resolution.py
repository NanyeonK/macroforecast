from macrocast.specs import CompiledExperimentSpec, compile_experiment_spec_from_dict


MINIMAL_CONFIG = {
    'experiment': {'id': 'design-test', 'output_dir': '/tmp/macrocast_test', 'horizons': [1, 3], 'window': 'expanding', 'n_jobs': 1},
    'data': {'dataset': 'fred_md', 'target': 'INDPRO'},
    'features': {'factor_type': 'X', 'n_factors': 4, 'n_lags': 2},
    'models': [{'name': 'rf'}],
}


def test_design_resolution_returns_compiled_spec() -> None:
    compiled = compile_experiment_spec_from_dict(MINIMAL_CONFIG, preset_id='researcher_explicit')
    assert isinstance(compiled, CompiledExperimentSpec)
    assert compiled.experiment_config.data.dataset == 'fred_md'
    assert compiled.meta_config['benchmark_id'] == 'ar_bic_expanding'
    assert compiled.meta_config['model_family'] == 'tree_ensemble'
    assert compiled.meta_config['minimum_train_size'] == 120


def test_design_resolution_defaults_dataset_target_and_task() -> None:
    compiled = compile_experiment_spec_from_dict(MINIMAL_CONFIG)
    assert compiled.meta_config['dataset'] == 'fred_md'
    assert compiled.meta_config['target'] == 'INDPRO'
    assert compiled.meta_config['frequency'] == 'monthly'
    assert compiled.meta_config['evaluation_scale'] == 'transformed_scale'
    assert compiled.meta_config['sample_period']['start'] == '1960-01-01'
    assert compiled.meta_config['validation_design'] == 'last_block'
