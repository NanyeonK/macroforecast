from __future__ import annotations

import math

import pandas as pd
import pytest

from macrocast.core import execute_l1_l2
from macrocast.core.types import DiagnosticArtifact
from macrocast.core.runtime import materialize_l1
from macrocast.core.yaml import parse_recipe_yaml


def test_execute_l1_l2_materializes_inline_custom_panel():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01]
          y: [1.0, 2.0, 3.0]
          x1: [10.0, null, 30.0]
          x2: [5.0, 6.0, 7.0]
    2_preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: none
        imputation_policy: mean
        frame_edge_policy: keep_unbalanced
    """
    result = execute_l1_l2(yaml_text)

    l1_artifact = result.sink("l1_data_definition_v1")
    l2_artifact = result.sink("l2_clean_panel_v1")

    assert l1_artifact.raw_panel.shape == (3, 3)
    assert l1_artifact.raw_panel.column_names == ("y", "x1", "x2")
    assert l2_artifact.panel.shape == (3, 3)
    assert l2_artifact.n_imputed_cells == 1
    assert l2_artifact.panel.data.loc[pd.Timestamp("2020-02-01"), "x1"] == 20.0


def test_execute_l1_l2_applies_sample_window_and_tcode():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
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
    2_preprocessing:
      fixed_axes:
        transform_policy: apply_official_tcode
        outlier_policy: none
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
    """
    result = execute_l1_l2(yaml_text)
    l2_artifact = result.sink("l2_clean_panel_v1")

    assert list(l2_artifact.panel.data.index) == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
    assert pd.isna(l2_artifact.panel.data.iloc[0]["x1"])
    assert l2_artifact.panel.data.iloc[1]["x1"] == 6.0
    assert l2_artifact.transform_map_applied == {"x1": 2}


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
        1_data:
          fixed_axes:
            custom_source_policy: custom_panel_only
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
        1_data:
          fixed_axes:
            custom_source_policy: official_only
            dataset: fred_md
            frequency: monthly
          leaf_config:
            target: INDPRO
            local_raw_source: tests/fixtures/fred_md_ar_sample.csv
            cache_root: {tmp_path}
        """
    )

    artifact, _, resolved = materialize_l1(root)

    assert resolved["custom_source_policy"] == "official_only"
    assert artifact.raw_panel.shape[1] == 4
    assert artifact.raw_panel.column_names == ("INDPRO", "RPI", "UNRATE", "CPIAUCSL")
    assert artifact.raw_panel.data.index[0] == pd.Timestamp("2000-01-01")
    assert artifact.raw_panel.metadata.values["source"] == "official"
    assert artifact.raw_panel.metadata.values["dataset"] == "fred_md"
    assert artifact.raw_panel.metadata.values["transform_codes"]["INDPRO"] == 5


def test_execute_l1_l2_applies_official_fred_md_transform_codes(tmp_path):
    yaml_text = f"""
    1_data:
      fixed_axes:
        custom_source_policy: official_only
        dataset: fred_md
        frequency: monthly
      leaf_config:
        target: INDPRO
        local_raw_source: tests/fixtures/fred_md_ar_sample.csv
        cache_root: {tmp_path}
    2_preprocessing:
      fixed_axes:
        transform_policy: apply_official_tcode
        outlier_policy: none
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
    """

    result = execute_l1_l2(yaml_text)
    l2_artifact = result.sink("l2_clean_panel_v1")

    assert l2_artifact.transform_map_applied["INDPRO"] == 5
    assert pd.isna(l2_artifact.panel.data.loc[pd.Timestamp("2000-01-01"), "INDPRO"])
    assert l2_artifact.panel.data.loc[pd.Timestamp("2000-02-01"), "INDPRO"] == pytest.approx(
        math.log(101.0) - math.log(100.0)
    )


def test_execute_l1_l2_winsorize_replace_with_cap_value_counts_capped_cells():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01]
          y: [1.0, 1.0, 1.0, 1.0]
          x1: [1.0, 2.0, 3.0, 100.0]
    2_preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: winsorize
        outlier_action: replace_with_cap_value
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
      leaf_config:
        winsorize_quantiles: [0.25, 0.75]
    """

    result = execute_l1_l2(yaml_text)
    l2_artifact = result.sink("l2_clean_panel_v1")

    assert l2_artifact.n_outliers_flagged == 2
    assert l2_artifact.panel.data["x1"].max() < 100.0
    assert l2_artifact.panel.data["x1"].min() > 1.0


def test_execute_l1_l2_materializes_l1_5_l2_5_diagnostics():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01]
          y: [1.0, 2.0, 3.0, 4.0]
          x1: [10.0, null, 30.0, 40.0]
          x2: [5.0, 6.0, 7.0, 8.0]
    1_5_data_summary:
      enabled: true
      fixed_axes:
        summary_metrics: [mean, sd, n_obs, n_missing]
        correlation_view: full_matrix
    2_preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: none
        imputation_policy: mean
        frame_edge_policy: keep_unbalanced
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        correlation_shift: delta_matrix
    """

    result = execute_l1_l2(yaml_text)
    l1_diag = result.sink("l1_5_diagnostic_v1")
    l2_diag = result.sink("l2_5_diagnostic_v1")

    assert isinstance(l1_diag, DiagnosticArtifact)
    assert l1_diag.enabled is True
    assert l1_diag.metadata["sample_coverage"]["n_missing"]["x1"] == 1
    assert l1_diag.metadata["univariate_summary"]["x1"]["n_obs"] == 3
    assert "correlation" in l1_diag.metadata
    assert isinstance(l2_diag, DiagnosticArtifact)
    assert l2_diag.enabled is True
    assert l2_diag.metadata["comparison"]["raw_missing_total"] == 1
    assert l2_diag.metadata["comparison"]["clean_missing_total"] == 0
    assert l2_diag.metadata["cleaning_effect_summary"]["n_imputed_cells"] == 1
    assert "correlation_shift" in l2_diag.metadata
    assert result.resolved_axes["l1_5"]["diagnostic_format"] == "pdf"


def test_execute_l1_l2_materializes_disabled_diagnostic_artifact():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
      leaf_config:
        target: y
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01]
          y: [1.0, 2.0]
    1_5_data_summary:
      enabled: false
    2_preprocessing:
      fixed_axes:
        transform_policy: no_transform
        outlier_policy: none
        imputation_policy: none_propagate
        frame_edge_policy: keep_unbalanced
    """

    result = execute_l1_l2(yaml_text)
    diagnostic = result.sink("l1_5_diagnostic_v1")

    assert diagnostic.enabled is False
    assert diagnostic.metadata["runtime"] == "core_diagnostic_disabled"


def test_materialize_l1_rejects_unsupported_official_core_dataset(tmp_path):
    root = parse_recipe_yaml(
        f"""
        1_data:
          fixed_axes:
            custom_source_policy: official_only
            dataset: fred_sd
            frequency: monthly
          leaf_config:
            target: CPIAUCSL
            cache_root: {tmp_path}
        """
    )

    with pytest.raises(NotImplementedError, match="not supported by core L1 runtime yet"):
        materialize_l1(root)
