from macrocast.meta import load_axes_registry, load_preset_registry, resolve_meta_config
from macrocast.meta.exceptions import IllegalOverrideError


def test_meta_resolution_applies_preset_and_overrides() -> None:
    resolved = resolve_meta_config(
        preset_registry=load_preset_registry(),
        axes_registry=load_axes_registry(),
        preset_id='researcher_explicit',
        global_defaults={'benchmark_family': 'historical_mean'},
        dataset_defaults={'dataset': 'fred_md', 'frequency': 'monthly'},
        target_defaults={'target': 'INDPRO', 'evaluation_scale': 'transformed_scale'},
        model_defaults={'model_family': 'tree_ensemble', 'tuning_method': 'grid_search'},
        experiment_overrides={'dataset': 'fred_md', 'target': 'INDPRO'},
    )
    assert resolved['benchmark_family'] == 'ar'
    assert resolved['benchmark_id'] == 'ar_bic_expanding'
    assert resolved['dataset'] == 'fred_md'
    assert resolved['target'] == 'INDPRO'
    assert resolved['model_family'] == 'tree_ensemble'
    assert resolved['evaluation_scale'] == 'explicit_required'


def test_meta_resolution_rejects_invariant_override() -> None:
    try:
        resolve_meta_config(
            preset_registry=load_preset_registry(),
            axes_registry=load_axes_registry(),
            preset_id='researcher_explicit',
            experiment_overrides={'no_lookahead_rule': False},
        )
    except IllegalOverrideError:
        return
    raise AssertionError('expected IllegalOverrideError')


def test_meta_resolution_rejects_run_override_of_experiment_fixed_axis() -> None:
    try:
        resolve_meta_config(
            preset_registry=load_preset_registry(),
            axes_registry=load_axes_registry(),
            preset_id='researcher_explicit',
            experiment_overrides={'dataset': 'fred_md'},
            run_overrides={'dataset': 'fred_qd'},
        )
    except IllegalOverrideError:
        return
    raise AssertionError('expected IllegalOverrideError')
