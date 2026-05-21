"""test(c50): ALFRED vintage adapter -- 8 scenarios from test-spec.md.

Tests the behavioral contract for real_time_alfred vintage policy promotion
from 'future' to 'operational' (Cycle 50). All scenarios use local snapshot
mode only (no network calls).
"""
from __future__ import annotations

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario 1.1 -- Validator accepts real_time_alfred with local snapshot dir
# ---------------------------------------------------------------------------

def test_alfred_validator_accepts_with_snapshot_dir():
    """Contract: real_time_alfred is accepted when alfred_snapshot_dir is provided."""
    yaml_text = """
    1_data:
      fixed_axes:
        vintage_policy: real_time_alfred
      leaf_config:
        target: CPIAUCSL
        alfred_snapshot_dir: /tmp/mock_alfred_dir
    """
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors, (
        f"Expected no hard errors when alfred_snapshot_dir is provided, "
        f"got: {[i.message for i in report.hard_errors]}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.2 -- Validator rejects real_time_alfred without alfred_snapshot_dir
# ---------------------------------------------------------------------------

def test_alfred_validator_rejects_without_snapshot_dir():
    """Contract: real_time_alfred without alfred_snapshot_dir emits a message
    referencing alfred_snapshot_dir (hard error or warning)."""
    yaml_text = """
    1_data:
      fixed_axes:
        vintage_policy: real_time_alfred
      leaf_config:
        target: CPIAUCSL
    """
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    report = validate_layer(parse_layer_yaml(yaml_text))
    all_messages = (
        [i.message for i in report.hard_errors]
        + [i.message for i in getattr(report, "warnings", [])]
    )
    assert any("alfred_snapshot_dir" in m for m in all_messages), (
        f"Expected 'alfred_snapshot_dir' in validation output, got: {all_messages}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.3 -- Validator still accepts current_vintage (regression guard)
# ---------------------------------------------------------------------------

def test_alfred_current_vintage_still_accepted():
    """Regression: current_vintage must still be accepted without errors."""
    yaml_text = """
    1_data:
      fixed_axes:
        vintage_policy: current_vintage
      leaf_config:
        target: CPIAUCSL
    """
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors, (
        f"current_vintage must be accepted, got: {[i.message for i in report.hard_errors]}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.4 -- real_time_alfred option is status="operational" in AxisSpec
# ---------------------------------------------------------------------------

def test_alfred_option_status_is_operational():
    """Contract: real_time_alfred option carries status='operational' in the l1_a AxisSpec.

    Note: vintage_policy is in the l1_a (Source selection) sub-layer, not l1_g.
    Options use .value attribute (not .name) per the codebase convention.
    """
    from macroforecast.core.layers.l1 import L1_LAYER_SPEC

    # vintage_policy is in the l1_a (Source selection) sub-layer.
    vintage_axis = L1_LAYER_SPEC.axes["l1_a"]["vintage_policy"]
    opt = next(
        (o for o in vintage_axis.options if o.value == "real_time_alfred"), None
    )
    assert opt is not None, (
        "real_time_alfred option not found in l1_a vintage_policy axis; "
        f"available options: {[o.value for o in vintage_axis.options]}"
    )
    assert opt.status == "operational", (
        f"Expected status='operational', got status={opt.status!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.5 -- load_alfred_vintage_snapshot: synthetic value recovery
# ---------------------------------------------------------------------------

def test_load_alfred_vintage_snapshot_synthetic(tmp_path):
    """Verify wide-format value recovery from a synthetic parquet snapshot.

    Tolerances: atol=1e-6 (exact float per test-spec.md).
    """
    from macroforecast.raw.alfred_adapter import load_alfred_vintage_snapshot

    snapshot_df = pd.DataFrame({
        "series_id": ["CPIAUCSL", "CPIAUCSL"],
        "observation_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "vintage_date": ["2020-01", "2020-02"],
        "value": [257.971, 258.678],
    })
    snapshot_df.to_parquet(tmp_path / "alfred_vintages.parquet", index=False)

    result = load_alfred_vintage_snapshot(
        snapshot_path=tmp_path,
        series_ids=["CPIAUCSL"],
        vintage_date="2020-02",
    )

    # Shape and column checks.
    assert "CPIAUCSL" in result.columns, (
        f"Expected CPIAUCSL column, got {list(result.columns)}"
    )
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"

    # Cross-property: output has DatetimeIndex.
    assert isinstance(result.index, pd.DatetimeIndex), (
        "load_alfred_vintage_snapshot must return DataFrame with DatetimeIndex"
    )

    # Value recovery (atol=1e-6 per test-spec.md).
    assert abs(result["CPIAUCSL"].iloc[0] - 257.971) < 1e-6, (
        f"Expected 257.971, got {result['CPIAUCSL'].iloc[0]}"
    )
    assert abs(result["CPIAUCSL"].iloc[1] - 258.678) < 1e-6, (
        f"Expected 258.678, got {result['CPIAUCSL'].iloc[1]}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.6 -- apply_alfred_vintage_to_panel: static mode replaces values
# ---------------------------------------------------------------------------

def test_apply_alfred_vintage_static_mode(tmp_path):
    """Contract: static mode overwrites matching panel values with vintage values.

    Tolerance: atol=1e-6 per test-spec.md.
    """
    from macroforecast.raw.alfred_adapter import apply_alfred_vintage_to_panel

    snapshot_df = pd.DataFrame({
        "series_id": ["CPIAUCSL"],
        "observation_date": pd.to_datetime(["2020-01-01"]),
        "vintage_date": ["2020-01"],
        "value": [257.971],
    })
    snapshot_df.to_parquet(tmp_path / "alfred_vintages.parquet", index=False)

    # Panel has a deliberately wrong "current" value (999.0).
    panel = pd.DataFrame(
        {"CPIAUCSL": [999.0]},
        index=pd.DatetimeIndex(["2020-01-01"]),
    )
    resolved = {"vintage_policy": "real_time_alfred"}
    leaf_config = {
        "alfred_snapshot_dir": str(tmp_path),
        "alfred_vintage_date": "2020-01",
        "alfred_mode": "local",
    }
    result = apply_alfred_vintage_to_panel(panel, resolved, leaf_config)

    assert abs(result["CPIAUCSL"].iloc[0] - 257.971) < 1e-6, (
        f"Expected 257.971 from vintage snapshot, got {result['CPIAUCSL'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# Scenario 1.7 -- apply_alfred_vintage_to_panel: current_vintage passthrough
# ---------------------------------------------------------------------------

def test_apply_alfred_vintage_passthrough_on_current_vintage():
    """Contract: current_vintage policy returns the panel unchanged (identity)."""
    from macroforecast.raw.alfred_adapter import apply_alfred_vintage_to_panel

    panel = pd.DataFrame(
        {"CPIAUCSL": [257.0]},
        index=pd.DatetimeIndex(["2020-01-01"]),
    )
    resolved = {"vintage_policy": "current_vintage"}
    leaf_config: dict = {}
    result = apply_alfred_vintage_to_panel(panel, resolved, leaf_config)

    assert result is panel or result.equals(panel), (
        "Panel must be unchanged (identity or equality) for current_vintage"
    )


# ---------------------------------------------------------------------------
# Scenario 1.8 -- load_alfred_vintage_snapshot: missing file raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_load_alfred_missing_file_raises(tmp_path):
    """Contract: FileNotFoundError when snapshot directory exists but has no
    alfred_vintages.parquet or alfred_vintages.csv file."""
    from macroforecast.raw.alfred_adapter import load_alfred_vintage_snapshot

    # tmp_path exists but is empty (no parquet or csv file).
    with pytest.raises(FileNotFoundError):
        load_alfred_vintage_snapshot(
            tmp_path, series_ids=["CPIAUCSL"], vintage_date="2020-01"
        )
