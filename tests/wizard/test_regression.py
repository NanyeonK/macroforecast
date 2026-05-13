"""Regression test — Scenario RG-01: old run_wizard emits DeprecationWarning."""
from __future__ import annotations

import warnings
import tempfile
import os
import pytest


def test_rg01_run_wizard_emits_deprecation_warning(tmp_path):
    """RG-01: run_wizard() emits DeprecationWarning containing 'deprecated' or 'wizard'."""
    from macroforecast.scaffold.wizard import run_wizard

    output_path = tmp_path / "recipe.yaml"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call with a no-op input_fn to avoid interactive prompts
        # Pass interactive_layers=() to skip all layer walks
        run_wizard(
            input_fn=lambda _: "",
            output_path=str(output_path),
            interactive_layers=(),
        )

    assert len(w) >= 1, (
        "Expected at least one warning from run_wizard(), got none"
    )

    deprecation_warnings = [
        warning for warning in w
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) >= 1, (
        f"Expected DeprecationWarning, got: {[(warning.category, str(warning.message)) for warning in w]}"
    )

    message = str(deprecation_warnings[0].message).lower()
    assert "deprecated" in message or "wizard" in message, (
        f"DeprecationWarning message does not contain 'deprecated' or 'wizard': {message!r}"
    )
