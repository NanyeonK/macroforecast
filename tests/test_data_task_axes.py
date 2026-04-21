from __future__ import annotations

import pytest

from macrocast.registry import get_axis_registry


_NEW_AXES = {
    'release_lag_rule': {
        'layer': '1_data_task',
        'expected': {
            'ignore_release_lag', 'fixed_lag_all_series', 'series_specific_lag',
        },
    },
    'missing_availability': {
        'layer': '1_data_task',
        'expected': {
            'complete_case_only', 'available_case', 'x_impute_only',
        },
    },
    'variable_universe': {
        'layer': '1_data_task',
        'expected': {
            'all_variables', 'preselected_core', 'category_subset',
            'target_specific_subset', 'handpicked_set',
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
    # v1.0 1.5 cleanup trimmed 2 future values (break_test_detected, rolling_break_adaptive).
    expected = {"none", "pre_post_crisis", "pre_post_covid"}
    assert set(get_axis_registry()['structural_break_segmentation'].allowed_values) == expected


