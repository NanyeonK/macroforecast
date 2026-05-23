"""Independent validation suite — C63.1 CHANGELOG wording correction.

Tests C1-C3 from test-spec.md Section 5. Verifies that:
    C1 — CHANGELOG.md contains the corrected isinstance wording.
    C2 — The old misleading phrase 'both directions' is absent.
    C3 — The reverse-direction clarification ('is False') is present.

test-spec.md Section 5 specifies manual-reviewer inspection; these tests
provide automated string-match evidence for audit completeness.

The corrected line (19) in the file reads (raw markdown, backticks included):
    `isinstance(public_instance, _PrivateClass)` continues to hold
We match the prose-level content allowing for markdown backtick wrapping.
"""
from __future__ import annotations

import pathlib

# Locate CHANGELOG.md: this file is at tests/promotion/, two levels up = worktree root
_WORKTREE_ROOT = pathlib.Path(__file__).parents[2]
_CHANGELOG = _WORKTREE_ROOT / "CHANGELOG.md"


def _read_changelog_text() -> str:
    """Read full CHANGELOG.md text."""
    return _CHANGELOG.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C1 — Corrected wording is present in the file
# ---------------------------------------------------------------------------

def test_C1_corrected_wording_present() -> None:
    """C1: CHANGELOG.md contains the corrected isinstance phrasing.

    The raw text uses markdown backticks around the call expression.
    We test that the core prose phrase is present (backtick-tolerant match).
    """
    text = _read_changelog_text()

    # The corrected line contains (with markdown backticks in raw form):
    #   `isinstance(public_instance, _PrivateClass)` continues to hold
    # We match without backticks to be tolerant of markdown escaping.
    assert "isinstance(public_instance, _PrivateClass)" in text, (
        "C1 FAIL: 'isinstance(public_instance, _PrivateClass)' not found in CHANGELOG.md."
    )
    assert "continues to hold" in text, (
        "C1 FAIL: 'continues to hold' not found in CHANGELOG.md."
    )


# ---------------------------------------------------------------------------
# C2 — Old misleading phrase is absent
# ---------------------------------------------------------------------------

def test_C2_old_phrase_absent() -> None:
    """C2: CHANGELOG.md does NOT contain the old phrase 'both directions'."""
    text = _read_changelog_text()

    old_phrase = "both directions"
    assert old_phrase not in text, (
        f"C2 FAIL: old misleading phrase 'both directions' still present in CHANGELOG.md."
    )


# ---------------------------------------------------------------------------
# C3 — Reverse direction clarification is present
# ---------------------------------------------------------------------------

def test_C3_reverse_direction_clarification_present() -> None:
    """C3: CHANGELOG.md mentions that the reverse isinstance check is False."""
    text = _read_changelog_text()

    # The corrected wording should contain clarification that private->public is False
    # e.g. "is False, as expected with single inheritance"
    assert "is False" in text, (
        "C3 FAIL: CHANGELOG.md does not clarify that the reverse isinstance is False."
    )
