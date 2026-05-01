from __future__ import annotations

import pandas as pd
import pytest

from macrocast.core import execute_l1_l2
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


def test_execute_l1_l2_rejects_official_runtime_for_now():
    yaml_text = """
    1_data:
      fixed_axes: {}
      leaf_config:
        target: CPIAUCSL
    """
    with pytest.raises(NotImplementedError, match="official FRED runtime loading is deferred"):
        execute_l1_l2(yaml_text)
