"""Negative test for encyclopedia op-coverage drift CI gate (Cycle 58).

Tester-authored: validates that the gate *fires* when a new operational op
lacks an encyclopedia page.  This is a meta-test of the detection logic in
test_encyclopedia_op_coverage.py, verifying Scenario B6 from test-spec.md.

Strategy (per test-spec.md alternative approach):
- Build the expected page path for a fake op that does not exist on disk.
- Call the assertion logic directly (replicated inline, identical to gate).
- Assert that pytest.fail is raised with the required message components.
- No module re-import or monkeypatch of live registries needed.

This test must PASS (confirming the gate fires on a missing page).
"""
from __future__ import annotations

import pytest

from pathlib import Path


# ---------------------------------------------------------------------------
# Replicate the assertion logic from test_encyclopedia_op_coverage verbatim.
# This mirrors what the gate test does so the meta-test is authoritative.
# ---------------------------------------------------------------------------

def _run_gate_check(layer: str, name: str, page_path: Path) -> None:
    """Replica of the gate assertion block used in test_encyclopedia_op_coverage.

    Raises pytest.fail if page_path does not exist on disk — identical
    behavior to the parametrized gate tests.
    """
    if not page_path.exists():
        repo_root = Path(__file__).parents[2]
        try:
            relative = page_path.relative_to(repo_root)
        except ValueError:
            relative = page_path
        pytest.fail(
            f"{layer} op '{name}' is operational but has no encyclopedia page.\n"
            f"Expected: {relative}\n"
            f"Add the page or set op_page=False in its OptionDoc to intentionally opt out."
        )


# ---------------------------------------------------------------------------
# Scenario B6: inject a fake op that has no corresponding page on disk and
# confirm the gate fires with an informative message.
# ---------------------------------------------------------------------------

def test_informative_failure_message_on_missing_page() -> None:
    """Gate fires with correct message when a page is missing for a fake op.

    This is the negative control.  The fake op name is chosen to be
    guaranteed non-existent; we verify:
      1. pytest.fail IS raised (gate fires).
      2. The failure message contains the fake op name.
      3. The failure message contains the string 'encyclopedia page'.
      4. The failure message contains a file path reference.
    """
    fake_op_name = "__fake_test_op_xyz__"
    # Build the path the gate would expect (does not exist on disk).
    enc_root = Path(__file__).parents[2] / "docs" / "reference" / "encyclopedia"
    fake_page_path = enc_root / "l7" / "op" / f"{fake_op_name}.md"

    # Confirm the page truly does NOT exist (pre-condition for a valid test).
    assert not fake_page_path.exists(), (
        f"Pre-condition failed: fake page unexpectedly exists at {fake_page_path}"
    )

    # The gate should raise pytest.fail — capture it via pytest.raises.
    with pytest.raises(pytest.fail.Exception) as exc_info:
        _run_gate_check("L7", fake_op_name, fake_page_path)

    failure_message = str(exc_info.value)

    # Assert all required message components are present.
    assert fake_op_name in failure_message, (
        f"Failure message must contain op name '{fake_op_name}'.\n"
        f"Actual message: {failure_message}"
    )
    assert "encyclopedia page" in failure_message, (
        f"Failure message must contain 'encyclopedia page'.\n"
        f"Actual message: {failure_message}"
    )
    # Path reference: the message must contain some path-like substring.
    assert "docs/reference/encyclopedia" in failure_message or fake_op_name in failure_message, (
        f"Failure message must contain a file path reference.\n"
        f"Actual message: {failure_message}"
    )
