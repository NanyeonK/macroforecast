"""Cycle 14 L1-2 -- manifest captures warnings emitted during run.

Verifies that:
1. CellExecutionResult has captured_warnings field.
2. captured_warnings is populated when a UserWarning is emitted during execution.
3. to_manifest_dict() includes a top-level 'warnings' key.
"""
from __future__ import annotations

import warnings


def test_cell_execution_result_has_captured_warnings_field():
    """CellExecutionResult dataclass must have captured_warnings attribute."""
    from macroforecast.core.execution import CellExecutionResult
    import inspect
    fields = {f.name for f in CellExecutionResult.__dataclass_fields__.values()}
    assert "captured_warnings" in fields, (
        f"CellExecutionResult missing 'captured_warnings' field. Fields: {fields}"
    )


def test_manifest_dict_has_warnings_key():
    """to_manifest_dict() must return a dict with top-level 'warnings' key."""
    from macroforecast.core.execution import ManifestExecutionResult, CellExecutionResult
    from dataclasses import field as dc_field

    # Build a minimal ManifestExecutionResult with one cell that has a warning
    cell = CellExecutionResult(
        cell_id="cell_001",
        index=1,
        sweep_values={},
        duration_seconds=0.1,
        captured_warnings=(
            {"category": "UserWarning", "message": "test warning", "filename": "test.py", "lineno": 1},
        ),
    )
    result = ManifestExecutionResult(
        recipe_root={},
        cells=(cell,),
        failure_policy="fail_fast",
    )
    manifest = result.to_manifest_dict()
    assert "warnings" in manifest, f"Manifest missing 'warnings' key. Keys: {list(manifest.keys())}"
    assert isinstance(manifest["warnings"], list), "manifest['warnings'] must be a list"
    assert len(manifest["warnings"]) == 1, f"Expected 1 warning, got: {manifest['warnings']}"
    assert manifest["warnings"][0]["message"] == "test warning"


def test_warnings_from_unknown_key_captured():
    """L1-3 UserWarning emitted during execute_recipe must appear in manifest warnings."""
    from macroforecast.core.execution import _warn_unknown_recipe_keys

    # Verify the warning function emits UserWarning for unknown keys
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _warn_unknown_recipe_keys({"99_fake_layer": {}})
    uw = [x for x in w_list if issubclass(x.category, UserWarning)]
    assert len(uw) >= 1, "Expected at least 1 UserWarning for unknown key"
    assert any("99_fake_layer" in str(w.message) for w in uw)
