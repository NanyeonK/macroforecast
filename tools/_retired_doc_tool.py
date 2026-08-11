"""Shared refusal for the retired documentation tools.

These scripts predate the current package layout and the canonical generator. They are
kept as paths so an old command line still resolves to something that explains itself,
and they refuse before touching the filesystem or the network -- the point of keeping
them is that an old invocation fails loudly instead of half-working.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


def retire(
    script: str,
    *,
    reason: str,
    replacement: Sequence[str],
    argv: Sequence[str] | None = None,
) -> int:
    """Explain the retirement on stderr and return a non-zero status.

    Arbitrary legacy arguments are accepted and echoed rather than parsed, because the
    argument grammar belonged to the removed implementation and re-implementing it would
    only make the refusal look like a working tool.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    lines = [
        f"{script} has been retired and does nothing.",
        "",
        reason,
        "",
        "Supported commands:",
    ]
    lines.extend(f"    {command}" for command in replacement)
    if arguments:
        lines += ["", f"Ignored legacy arguments: {' '.join(arguments)}"]
    lines += [
        "",
        "No documentation or data file was written and no network fetch was made.",
    ]
    print("\n".join(lines), file=sys.stderr)
    return 2
