#!/usr/bin/env python3
"""Validate the README "Quick Use" example against the live package API.

History: this used to extract a ``A minimal recipe:`` YAML block and run it via
``mf.run(<yaml>)``. That recipe/runner belonged to an earlier "layered-ops"
architecture (``0_meta`` / ``layer_ref`` schema) that no longer exists. The
current ``mf.run`` fits exactly one model and does not accept a recipe file, so
the old check could never pass.

The README now carries a "Quick Use" python example that reads an external
``panel.csv``, so it cannot be executed end to end in CI without sample data.
This check instead guards against the realistic drift: it verifies the block
(a) parses and (b) only references ``mf.*`` attributes that actually exist on
the installed package, so the README cannot advertise a removed or renamed API.
"""
import ast
import os
import re
import sys

import macroforecast as mf

README_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
)

with open(README_PATH) as f:
    readme = f.read()

match = re.search(r"## Quick Use\n+```python\n(.*?)\n```", readme, re.DOTALL)
if not match:
    print(
        "ERROR: Could not find the '## Quick Use' python block in README.md",
        file=sys.stderr,
    )
    sys.exit(1)

code = match.group(1)

try:
    tree = ast.parse(code)
except SyntaxError as exc:
    print(f"README Quick Use: syntax error — {exc}", file=sys.stderr)
    sys.exit(1)


def _attr_chain(node: ast.Attribute):
    """Return the dotted parts of an attribute chain rooted at a bare Name."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


bad = set()
for node in ast.walk(tree):
    if not isinstance(node, ast.Attribute):
        continue
    chain = _attr_chain(node)
    if not chain or chain[0] not in ("mf", "macroforecast"):
        continue
    obj = mf
    resolved = chain[0]
    for part in chain[1:]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
            resolved += "." + part
        else:
            bad.add(f"{'.'.join(chain)}  (missing at {resolved}.{part})")
            break

if bad:
    print("README Quick Use references APIs that do not exist:", file=sys.stderr)
    for b in sorted(bad):
        print(f"  - {b}", file=sys.stderr)
    sys.exit(1)

print("README Quick Use: OK (parses; all mf.* references resolve against the package)")
sys.exit(0)
