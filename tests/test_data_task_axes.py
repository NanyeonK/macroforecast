from __future__ import annotations

import pytest

from macrocast.registry import get_axis_registry


_NEW_AXES = {
    'release_lag_rule': {
        'layer': '1_data_task',
        'expected': {
            'ignore_release_lag', 'fixed_lag_all_series', 'series_specific_lag',
            'calendar_exact_lag', 'lag_conservative', 'lag_aggressive',
        },
    },
    'missing_availability': {
        'layer': '1_data_task',
        'expected': {
            'complete_case_only', 'available_case', 'target_date_drop_if_missing',
            'x_impute_only', 'real_time_missing_as_missing',
            'state_space_fill', 'factor_fill', 'em_fill',
        },
    },
    'variable_universe': {
        'layer': '1_data_task',
        'expected': {
            'all_variables', 'preselected_core', 'category_subset', 'paper_replication_subset',
            'target_specific_subset', 'expert_curated_subset', 'stability_filtered_subset',
            'correlation_screened_subset', 'feature_selection_dynamic_subset',
        },
    },
    'separation_rule': {
        'layer': '2_preprocessing',
        'expected': {
            'strict_separation', 'shared_transform_then_split',
            'X_only_transform', 'target_only_transform', 'joint_preprocessor',
        },
    },
}


def test_phase3_new_axes_registered():
    reg = get_axis_registry()
    for axis_name in _NEW_AXES:
        assert axis_name in reg, f'missing axis {axis_name!r}'


@pytest.mark.parametrize('axis_name,info', list(_NEW_AXES.items()))
def test_phase3_axis_layer_and_values(axis_name, info):
    reg = get_axis_registry()
    entry = reg[axis_name]
    assert entry.layer == info['layer']
    assert set(entry.allowed_values) == info['expected']


@pytest.mark.parametrize('axis_name', list(_NEW_AXES.keys()))
def test_phase3_axis_has_at_least_one_operational(axis_name):
    entry = get_axis_registry()[axis_name]
    statuses = list(entry.current_status.values())
    assert 'operational' in statuses, f'{axis_name} has no operational value: {statuses}'


def test_existing_reused_axes_still_present():
    reg = get_axis_registry()
    for axis in ('min_train_size', 'structural_break_segmentation', 'evaluation_scale'):
        assert axis in reg, f'expected reused axis {axis} missing'


def test_min_train_size_values_match_plan():
    expected = {
        'fixed_n_obs', 'fixed_years',
        'model_specific_min_train', 'target_specific_min_train', 'horizon_specific_min_train',
    }
    assert set(get_axis_registry()['min_train_size'].allowed_values) == expected


def test_structural_break_segmentation_values_match_plan():
    expected = {
        'none', 'pre_post_crisis', 'pre_post_covid',
        'user_break_dates', 'break_test_detected', 'rolling_break_adaptive',
    }
    assert set(get_axis_registry()['structural_break_segmentation'].allowed_values) == expected


def test_evaluation_scale_renamed_and_extended():
    entry = get_axis_registry()['evaluation_scale']
    values = set(entry.allowed_values)
    assert values == {'original_scale', 'transformed_scale', 'both'}
    assert 'raw_level' not in values
    for v in values:
        assert entry.current_status[v] == 'operational'
