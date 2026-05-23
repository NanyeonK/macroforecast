"""Independent validation suite — C63.1 CHANGELOG wording correction.

Tests C1-C3 from test-spec.md Section 5. Verifies that:
    C1 — CHANGELOG.md contains the corrected isinstance wording.
    C2 — The old misleading phrase 'both directions' is absent from the
         active C63 [Unreleased] section (not file-global; the Fixed —
         Cycle 63.1 section legitimately quotes the old phrase as a
         historical description of the bug that was corrected).
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


def _read_c63_active_section() -> str:
    """Extract only the Cycle 63 [Unreleased] active section from CHANGELOG.md.

    The C63 active section begins at '### Added — Cycle 63' and ends at the
    '---' separator that precedes '### Added — Cycle 63.1'.  The Fixed —
    Cycle 63.1 block that follows intentionally quotes the old phrasing as a
    historical record of the bug; the C2 test must not fire on that quote.
    """
    text = _read_changelog_text()
    # Delimit the C63 active block: everything from its header up to (but not
    # including) the first '---' horizontal rule that closes it.
    start_marker = "### Added — Cycle 63"
    end_marker = "---"
    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise AssertionError(
            "C2/setup FAIL: '### Added — Cycle 63' section not found in CHANGELOG.md."
        )
    # Search for the closing '---' *after* the section start.
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        # If no '---' exists after the section, take the rest of the file.
        return text[start_idx:]
    return text[start_idx:end_idx]


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
# C2 — Old misleading phrase is absent from the active C63 section
# ---------------------------------------------------------------------------

def test_C2_old_phrase_absent() -> None:
    """C2: The active C63 [Unreleased] section does NOT use 'both directions'.

    Scope is narrowed to the C63 active section only (from '### Added —
    Cycle 63' up to the closing '---' separator).  The subsequent
    'Fixed — Cycle 63.1' block legitimately quotes the old phrase as a
    historical description of the corrected bug; a file-global search would
    produce a false positive against that historical record.
    """
    c63_section = _read_c63_active_section()

    old_phrase = "both directions"
    assert old_phrase not in c63_section, (
        "C2 FAIL: old misleading phrase 'both directions' is still present "
        "as an active claim in the Cycle 63 [Unreleased] section of CHANGELOG.md."
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
