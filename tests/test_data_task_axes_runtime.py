from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from macrocast.execution.build import (
    _PHASE3_DEFAULTS,
    _data_task_axis,
    _phase3_axis_consumption,
    _apply_release_lag,
    _apply_missing_availability,
    _apply_variable_universe,
)


@dataclass(frozen=True)
class _StubRecipe:
    data_task_spec: dict = field(default_factory=dict)


def _recipe_with(**axes) -> _StubRecipe:
    return _StubRecipe(data_task_spec=dict(axes))


@pytest.mark.parametrize('axis_name,default', list(_PHASE3_DEFAULTS.items()))
def test_data_task_axis_defaults(axis_name, default):
    r = _recipe_with()
    assert _data_task_axis(r, axis_name) == default


@pytest.mark.parametrize('axis_name', list(_PHASE3_DEFAULTS.keys()))
def test_data_task_axis_round_trip(axis_name):
    r = _recipe_with(**{axis_name: 'sentinel_value_xyz'})
    assert _data_task_axis(r, axis_name) == 'sentinel_value_xyz'


def test_phase3_axis_consumption_helper_lists_all_axes():
    consumption = _phase3_axis_consumption()
    assert set(consumption.keys()) == set(_PHASE3_DEFAULTS.keys())


def test_phase3_axis_consumption_grep_anchors_present_in_build_py():
    import macrocast.execution.build as build_mod
    src = open(build_mod.__file__).read()
    for axis in _PHASE3_DEFAULTS:
        anchor = f'_data_task_axis(recipe, "{axis}")'
        assert anchor in src, f'missing anchor {anchor!r} for axis={axis}'


class _StubRaw:
    def __init__(self, data):
        self.data = data
        self.dataset_metadata = None
        self.artifact = None


def _make_raw():
    import pandas as pd
    df = pd.DataFrame({'date': pd.date_range('2000-01-01', periods=5, freq='MS'), 'INDPRO': [1.0, 2, 3, 4, 5]})
    return _StubRaw(df)


def test_release_lag_ignore_is_noop():
    r = _make_raw()
    out = _apply_release_lag(r, 'ignore_release_lag')
    assert out is r


def test_release_lag_fixed_lag_shifts_columns():
    import pandas as pd
    r = _make_raw()
    out = _apply_release_lag(r, 'fixed_lag_all_series')
    assert pd.isna(out.data['INDPRO'].iloc[0])
    assert out.data['INDPRO'].iloc[1] == 1.0


@pytest.mark.parametrize('rule', ['complete_case_only', 'available_case', 'x_impute_only'])
def test_missing_availability_passes_through(rule):
    r = _make_raw()
    out = _apply_missing_availability(r, rule)
    assert out is r


def test_variable_universe_all_is_noop():
    r = _make_raw()
    out = _apply_variable_universe(r, 'all_variables')
    assert out is r


def test_variable_universe_preselected_core_filters_when_core_present():
    import pandas as pd
    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=3, freq='MS'),
        'INDPRO': [1.0, 2, 3],
        'PAYEMS': [10, 20, 30],
        'IRRELEVANT_X': [99, 98, 97],
    })
    raw = _StubRaw(df)
    out = _apply_variable_universe(raw, 'preselected_core')
    cols = set(out.data.columns)
    assert 'IRRELEVANT_X' not in cols
    assert 'INDPRO' in cols
    assert 'PAYEMS' in cols


def test_min_train_size_round_trip():
    r = _recipe_with(min_train_size='fixed_years')
    assert _data_task_axis(r, 'min_train_size') == 'fixed_years'


def test_structural_break_segmentation_round_trip():
    r = _recipe_with(structural_break_segmentation='pre_post_covid')
    assert _data_task_axis(r, 'structural_break_segmentation') == 'pre_post_covid'




def test_separation_rule_round_trip():
    r = _recipe_with(separation_rule='shared_transform_then_split')
    assert _data_task_axis(r, 'separation_rule') == 'shared_transform_then_split'
