from macrocast.specs import (
    CompiledExperimentSpec,
    compile_experiment_spec_from_dict,
    validate_compiled_experiment_spec,
)


MINIMAL_CONFIG = {
    'experiment': {'id': 'spec-test', 'output_dir': '/tmp/macrocast_test', 'horizons': [1], 'window': 'expanding', 'n_jobs': 1},
    'data': {'dataset': 'fred_md', 'target': 'INDPRO'},
    'features': {'factor_type': 'X', 'n_factors': 2, 'n_lags': 1},
    'models': [{'name': 'rf'}],
}


def test_compile_experiment_spec_from_dict_returns_compiled_spec() -> None:
    compiled = compile_experiment_spec_from_dict(MINIMAL_CONFIG, preset_id='researcher_explicit')
    assert isinstance(compiled, CompiledExperimentSpec)
    assert compiled.meta_config['dataset'] == 'fred_md'
    assert compiled.meta_config['target_preprocess_recipe'] == 'basic_none'
    assert compiled.meta_config['x_preprocess_recipe'] == 'basic_none'


def test_compiled_spec_contract_dict_contains_core_fields() -> None:
    compiled = compile_experiment_spec_from_dict(MINIMAL_CONFIG, preset_id='researcher_explicit')
    contract = compiled.to_contract_dict()
    for key in [
        'dataset', 'target', 'horizon', 'benchmark_id', 'benchmark_family',
        'evaluation_scale', 'target_preprocess_recipe', 'x_preprocess_recipe',
        'sample_period', 'oos_period', 'minimum_train_size', 'validation_design',
        'outer_window', 'model_family', 'tuning_method', 'hyperparameter_space',
    ]:
        assert key in contract


def test_validate_compiled_experiment_spec_rejects_missing_contract_field() -> None:
    compiled = compile_experiment_spec_from_dict(MINIMAL_CONFIG, preset_id='researcher_explicit')
    del compiled.meta_config['benchmark_id']
    try:
        validate_compiled_experiment_spec(compiled)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
