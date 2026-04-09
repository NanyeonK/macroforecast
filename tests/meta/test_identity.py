from macrocast.meta import make_cell_id, make_config_hash, make_experiment_id, make_run_id


def test_identity_builders_are_deterministic() -> None:
    exp = make_experiment_id(
        target_id='INDPRO', dataset_id='fred_md', sample_id='1960_2017', preprocess_id='basic_none', split_id='expanding', benchmark_id='ar_bic_expanding'
    )
    run = make_run_id(
        experiment_id=exp, feature_set_id='factors_x', model_set_id='rf_en', tuning_policy_id='default', code_version='v1'
    )
    cell = make_cell_id(run_id=run, horizon=6, model_id='rf', feature_recipe_id='factors_x')
    assert exp == make_experiment_id(
        target_id='INDPRO', dataset_id='fred_md', sample_id='1960_2017', preprocess_id='basic_none', split_id='expanding', benchmark_id='ar_bic_expanding'
    )
    assert run.startswith(exp)
    assert '__h06__' in cell


def test_config_hash_changes_with_payload() -> None:
    h1 = make_config_hash({'a': 1, 'b': 2})
    h2 = make_config_hash({'a': 1, 'b': 3})
    assert h1 != h2
