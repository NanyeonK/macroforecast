from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from macrocast.execution.build import (
    _PHASE3_DEFAULTS,
    _TRAINING_AXIS_DEFAULTS,
    _data_task_axis,
    _layer2_runtime_spec,
    _training_axis,
    _phase3_axis_consumption,
    _apply_release_lag,
    _apply_missing_availability,
    _apply_raw_missing_policy,
    _apply_raw_outlier_policy,
    _apply_tcode_preprocessing,
    _apply_variable_universe,
)


@dataclass(frozen=True)
class _StubRecipe:
    data_task_spec: dict = field(default_factory=dict)
    training_spec: dict = field(default_factory=dict)


def _recipe_with(**axes) -> _StubRecipe:
    return _StubRecipe(data_task_spec=dict(axes))


def _recipe_with_training(**axes) -> _StubRecipe:
    return _StubRecipe(training_spec=dict(axes))


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


@pytest.mark.parametrize('axis_name,default', list(_TRAINING_AXIS_DEFAULTS.items()))
def test_training_axis_defaults(axis_name, default):
    r = _recipe_with()
    assert _training_axis(r, axis_name) == default


@pytest.mark.parametrize('axis_name', list(_TRAINING_AXIS_DEFAULTS.keys()))
def test_training_axis_round_trip(axis_name):
    r = _recipe_with_training(**{axis_name: 'sentinel_value_xyz'})
    assert _training_axis(r, axis_name) == 'sentinel_value_xyz'


def test_training_axis_falls_back_to_legacy_data_task_spec():
    r = _recipe_with(min_train_size='fixed_years')
    assert _training_axis(r, 'min_train_size') == 'fixed_years'


def test_phase3_axis_consumption_grep_anchors_present_in_build_py():
    import macrocast.execution.build as build_mod
    src = open(build_mod.__file__).read()
    for axis in _PHASE3_DEFAULTS:
        anchor = f'_data_task_axis(recipe, "{axis}")'
        assert anchor in src, f'missing anchor {anchor!r} for axis={axis}'


class _StubRaw:
    def __init__(self, data, transform_codes=None):
        self.data = data
        self.dataset_metadata = None
        self.artifact = None
        self.transform_codes = dict(transform_codes or {})


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


def test_missing_availability_complete_case_is_noop():
    # complete_case_only is the default no-op; available_case and x_impute_only
    # now do real work (1.5 impl) — see tests/test_stage1_5_impl.py for their coverage.
    r = _make_raw()
    out = _apply_missing_availability(r, 'complete_case_only')
    assert out is r


def test_raw_missing_policy_x_impute_raw_fills_predictors_only():
    import pandas as pd
    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=3, freq='MS'),
        'INDPRO': [1.0, None, 3.0],
        'RPI': [1.0, None, 3.0],
    })
    raw = _StubRaw(df)

    out = _apply_raw_missing_policy(
        raw,
        'x_impute_raw',
        target='INDPRO',
        spec={'raw_x_imputation': 'mean'},
    )

    assert pd.isna(out.data['INDPRO'].iloc[1])
    assert out.data['RPI'].iloc[1] == 2.0
    assert out.data.attrs['macrocast_reports']['raw_missing']['before_official_transform'] is True


def test_raw_missing_policy_zero_fill_leading_x_before_tcode():
    import pandas as pd
    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=4, freq='MS'),
        'INDPRO': [1.0, 2.0, 3.0, 4.0],
        'RPI': [None, None, 3.0, 4.0],
    }).set_index('date')
    raw = _StubRaw(df)

    out = _apply_raw_missing_policy(raw, 'zero_fill_leading_x_before_tcode', target='INDPRO', spec={})

    assert out.data['RPI'].iloc[0] == 0.0
    assert out.data['RPI'].iloc[1] == 0.0
    assert out.data.attrs['macrocast_reports']['raw_missing']['leading_zero_filled']['RPI'] == [
        '2000-01-01',
        '2000-02-01',
    ]


def test_raw_outlier_policy_iqr_clip_raw_clips_before_tcode():
    import pandas as pd
    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=5, freq='MS'),
        'INDPRO': [1.0, 2.0, 3.0, 4.0, 100.0],
    })
    raw = _StubRaw(df)

    out = _apply_raw_outlier_policy(raw, 'iqr_clip_raw')

    assert out.data['INDPRO'].iloc[-1] < 100.0
    assert out.data.attrs['macrocast_reports']['raw_outliers']['before_official_transform'] is True


def test_tcode_preprocessing_prefers_data_task_axes_over_contract_bridge():
    import pandas as pd

    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=4, freq='MS'),
        'INDPRO': [1.0, 2.0, 4.0, 7.0],
        'RPI': [10.0, 20.0, 40.0, 80.0],
    }).set_index('date')
    raw = _StubRaw(df, transform_codes={'INDPRO': 2, 'RPI': 2})
    recipe = _recipe_with(
        official_transform_policy='dataset_tcode',
        official_transform_scope='apply_tcode_to_target',
        official_transform_source={
            'policy_source': 'layer1_axis',
            'scope_source': 'layer1_axis',
            'legacy_bridge_axes': [],
        },
    )
    contract = SimpleNamespace(tcode_policy='raw_only', tcode_application_scope='apply_tcode_to_X')

    out = _apply_tcode_preprocessing(raw, recipe, contract, target='INDPRO')

    assert pd.isna(out.data['INDPRO'].iloc[0])
    assert out.data['INDPRO'].iloc[1:].tolist() == [1.0, 2.0, 3.0]
    assert out.data['RPI'].tolist() == [10.0, 20.0, 40.0, 80.0]
    report = out.data.attrs['macrocast_reports']['tcode']
    assert report['scope'] == 'apply_tcode_to_target'
    assert report['source']['runtime_policy_source'] == 'data_task_spec'
    assert report['source']['runtime_scope_source'] == 'data_task_spec'
    assert report['source']['legacy_contract_fallback'] is False


def test_tcode_preprocessing_marks_legacy_contract_fallback_source():
    import pandas as pd

    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=3, freq='MS'),
        'INDPRO': [1.0, 2.0, 4.0],
        'RPI': [10.0, 20.0, 40.0],
    }).set_index('date')
    raw = _StubRaw(df, transform_codes={'INDPRO': 2, 'RPI': 2})
    contract = SimpleNamespace(tcode_policy='tcode_only', tcode_application_scope='apply_tcode_to_X')

    out = _apply_tcode_preprocessing(raw, _recipe_with(), contract, target='INDPRO')

    assert out.data['INDPRO'].tolist() == [1.0, 2.0, 4.0]
    assert pd.isna(out.data['RPI'].iloc[0])
    assert out.data['RPI'].iloc[1:].tolist() == [10.0, 20.0]
    report = out.data.attrs['macrocast_reports']['tcode']
    assert report['policy'] == 'dataset_tcode'
    assert report['scope'] == 'apply_tcode_to_X'
    assert report['source']['runtime_policy_source'] == 'legacy_preprocess_contract'
    assert report['source']['runtime_scope_source'] == 'legacy_preprocess_contract'
    assert report['source']['legacy_contract_fallback'] is True


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
    r = _recipe_with_training(min_train_size='fixed_years')
    assert _training_axis(r, 'min_train_size') == 'fixed_years'


def test_structural_break_segmentation_round_trip():
    r = _recipe_with(structural_break_segmentation='pre_post_covid')
    assert _layer2_runtime_spec(r)['structural_break_segmentation'] == 'pre_post_covid'




def test_separation_rule_round_trip():
    r = _recipe_with(separation_rule='shared_transform_then_split')
    assert _data_task_axis(r, 'separation_rule') == 'shared_transform_then_split'
