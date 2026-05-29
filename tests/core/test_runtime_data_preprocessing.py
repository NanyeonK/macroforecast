from __future__ import annotations

import math

import pandas as pd
import pytest

from macroforecast.core import execute_data_preprocessing
from macroforecast.core.runtime import materialize_l1
from macroforecast.core.yaml import parse_recipe_yaml


def test_execute_data_preprocessing_materializes_inline_custom_panel():
    yaml_text = """
    data:
      fixed_axes:
        panel_composition: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01]
          y: [1.0, 2.0, 3.0]
          x1: [10.0, null, 30.0]
          x2: [5.0, 6.0, 7.0]
    preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: none
        imputation_policy: mean
        imputation_temporal_rule: block_recompute
        frame_edge_policy: keep_unbalanced
    """
    result = execute_data_preprocessing(yaml_text)

    l1_artifact = result.sink("l1_data_definition_v1")
    preprocessed_artifact = result.sink("preprocessed_panel_v1")

    assert l1_artifact.raw_panel.shape == (3, 3)
    assert l1_artifact.raw_panel.column_names == ("y", "x1", "x2")
    assert preprocessed_artifact.panel.shape == (3, 3)
    assert preprocessed_artifact.n_imputed_cells == 1
    assert preprocessed_artifact.panel.data.loc[pd.Timestamp("2020-02-01"), "x1"] == 20.0


def test_execute_data_preprocessing_applies_sample_window_and_tcode():
    yaml_text = """
    data:
      fixed_axes:
        panel_composition: custom_panel_only
        frequency: monthly
        sample_start_rule: fixed_date
      leaf_config:
        target: y
        sample_start_date: "2020-02-01"
        custom_tcode_map: {x1: 2}
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01]
          y: [1.0, 2.0, 3.0]
          x1: [10.0, 15.0, 21.0]
    preprocessing:
      fixed_axes:
        transform_policy: apply_official_tcode
        outlier_policy: none
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
    """
    result = execute_data_preprocessing(yaml_text)
    preprocessed_artifact = result.sink("preprocessed_panel_v1")

    assert list(preprocessed_artifact.panel.data.index) == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
    assert pd.isna(preprocessed_artifact.panel.data.iloc[0]["x1"])
    assert preprocessed_artifact.panel.data.iloc[1]["x1"] == 6.0
    assert preprocessed_artifact.transform_map_applied == {"x1": 2}


def test_materialize_l1_supports_custom_source_path(tmp_path):
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "y": [1.0, 2.0],
            "x": [3.0, 4.0],
        }
    ).to_csv(path, index=False)
    root = parse_recipe_yaml(
        f"""
        data:
          fixed_axes:
            panel_composition: custom_panel_only
            frequency: monthly
          leaf_config:
            target: y
            custom_source_path: {path}
        """
    )

    artifact, _, _ = materialize_l1(root)
    assert artifact.raw_panel.shape == (2, 2)
    assert artifact.raw_panel.data.index[0] == pd.Timestamp("2020-01-01")


def test_materialize_l1_supports_official_fred_md_local_source(tmp_path):
    root = parse_recipe_yaml(
        f"""
        data:
          fixed_axes:
            panel_composition: official_only
            dataset: fred_md
            frequency: monthly
          leaf_config:
            target: INDPRO
            local_raw_source: tests/fixtures/fred_md_ar_sample.csv
            cache_root: {tmp_path}
        """
    )

    artifact, _, resolved = materialize_l1(root)

    assert resolved["panel_composition"] == "official_only"
    assert artifact.raw_panel.shape[1] == 4
    assert artifact.raw_panel.column_names == ("INDPRO", "RPI", "UNRATE", "CPIAUCSL")
    assert artifact.raw_panel.data.index[0] == pd.Timestamp("2000-01-01")
    assert artifact.raw_panel.metadata.values["source"] == "official"
    assert artifact.raw_panel.metadata.values["dataset"] == "fred_md"
    assert artifact.raw_panel.metadata.values["transform_codes"]["INDPRO"] == 5


def test_execute_data_preprocessing_applies_official_fred_md_transform_codes(tmp_path):
    yaml_text = f"""
    data:
      fixed_axes:
        panel_composition: official_only
        dataset: fred_md
        frequency: monthly
      leaf_config:
        target: INDPRO
        local_raw_source: tests/fixtures/fred_md_ar_sample.csv
        cache_root: {tmp_path}
    preprocessing:
      fixed_axes:
        transform_policy: apply_official_tcode
        outlier_policy: none
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
    """

    result = execute_data_preprocessing(yaml_text)
    preprocessed_artifact = result.sink("preprocessed_panel_v1")

    assert preprocessed_artifact.transform_map_applied["INDPRO"] == 5
    assert pd.isna(preprocessed_artifact.panel.data.loc[pd.Timestamp("2000-01-01"), "INDPRO"])
    assert preprocessed_artifact.panel.data.loc[pd.Timestamp("2000-02-01"), "INDPRO"] == pytest.approx(
        math.log(101.0) - math.log(100.0)
    )


def test_execute_data_preprocessing_winsorize_replace_with_cap_value_counts_capped_cells():
    yaml_text = """
    data:
      fixed_axes:
        panel_composition: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01]
          y: [1.0, 1.0, 1.0, 1.0]
          x1: [1.0, 2.0, 3.0, 100.0]
    preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: winsorize
        outlier_action: replace_with_cap_value
        imputation_policy: none_propagate
        imputation_temporal_rule: block_recompute
        frame_edge_policy: keep_unbalanced
      leaf_config:
        winsorize_quantiles: [0.25, 0.75]
    """

    result = execute_data_preprocessing(yaml_text)
    preprocessed_artifact = result.sink("preprocessed_panel_v1")

    assert preprocessed_artifact.n_outliers_flagged == 2
    assert preprocessed_artifact.panel.data["x1"].max() < 100.0
    assert preprocessed_artifact.panel.data["x1"].min() > 1.0


def test_materialize_l1_loads_fred_sd_from_local_fixture(tmp_path):
    from pathlib import Path

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    root = parse_recipe_yaml(
        f"""
        data:
          fixed_axes:
            panel_composition: official_only
            dataset: fred_sd
            frequency: monthly
          leaf_config:
            target: UR_CA
            cache_root: {tmp_path}
            local_raw_source: {fixtures / 'fred_sd_sample.csv'}
        """
    )

    artifact, regime, resolved = materialize_l1(root)
    assert artifact.dataset == "fred_sd"
    assert "UR_CA" in artifact.raw_panel.column_names
    assert artifact.raw_panel.shape[0] > 0
