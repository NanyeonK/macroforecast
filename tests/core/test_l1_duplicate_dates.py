"""F-P1-4 -- duplicate dates in L1 custom panels raise RuntimeError.

The fix (F-P1-4) replaces silent coalesce/undefined behaviour with an explicit
RuntimeError that lists the offending timestamps.
"""
from __future__ import annotations

import pytest

from macroforecast.core import execute_data_preprocessing


_BASE_YAML = """
data:
  fixed_axes:
    panel_composition: custom_panel_only
    frequency: monthly
  leaf_config:
    target: y
    custom_panel_inline: {}
preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
"""


def _recipe_with_inline(rows: dict) -> str:
    import yaml
    recipe = {
        "data": {
            "fixed_axes": {
                "panel_composition": "custom_panel_only",
                "frequency": "monthly",
            },
            "leaf_config": {
                "target": "y",
                "custom_panel_inline": rows,
            },
        },
        "preprocessing": {
            "fixed_axes": {
                "transform_policy": "no_transform",
                "outlier_policy": "none",
                "imputation_policy": "none_propagate",
                "frame_edge_policy": "keep_unbalanced",
            }
        },
    }
    return yaml.dump(recipe)


def test_duplicate_dates_raise_runtime_error():
    """Panel with two identical dates should raise RuntimeError listing the duplicate."""
    rows = {
        "date": ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"],
        "y": [1.0, 2.0, 2.5, 3.0],
        "x1": [10.0, 20.0, 21.0, 30.0],
    }
    yaml_text = _recipe_with_inline(rows)
    with pytest.raises(RuntimeError, match="duplicate dates"):
        execute_data_preprocessing(yaml_text)


def test_no_duplicate_dates_passes():
    """Panel with unique dates should proceed without error."""
    rows = {
        "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "y": [1.0, 2.0, 3.0],
        "x1": [10.0, 20.0, 30.0],
    }
    yaml_text = _recipe_with_inline(rows)
    result = execute_data_preprocessing(yaml_text)
    l1_artifact = result.sink("l1_data_definition_v1")
    assert l1_artifact.raw_panel.shape == (3, 2)  # y + x1


def test_duplicate_dates_error_lists_offending_date():
    """RuntimeError message must include the actual duplicated date."""
    rows = {
        "date": ["2020-01-01", "2020-03-01", "2020-03-01"],
        "y": [1.0, 3.0, 3.5],
    }
    yaml_text = _recipe_with_inline(rows)
    with pytest.raises(RuntimeError, match="2020-03"):
        execute_data_preprocessing(yaml_text)
