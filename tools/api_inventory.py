"""Inventory the top-level API and propose a stability tier for each symbol.

The classification is mechanical and stated so it can be argued with:

  STABLE     -- named in the README or getting-started guide, or a type a user
                must construct to call the package at all.
  ADVANCED   -- reachable through a submodule (mf.models.x, mf.tests.x, ...) and
                documented in the reference. Public, but the canonical path is
                the submodule, not the top level.
  EXPERIMENTAL -- a subsystem with no correctness audit yet, or one whose
                contract is still moving.

Nothing is removed here. The output is the list a human decides from.
"""
from __future__ import annotations

import collections
import inspect
import json
import re
from pathlib import Path

import macroforecast as mf

ROOT = Path(mf.__file__).parent
REPO = ROOT.parent

# --- what the docs actually tell a user to call ---------------------------- #
doc_text = ""
for rel in ("README.md", "docs/guide/getting_started.md", "docs/index.md"):
    p = REPO / rel
    if p.exists():
        doc_text += p.read_text(encoding="utf-8")
mentioned = set(re.findall(r"mf\.([A-Za-z_][A-Za-z0-9_]*)", doc_text))

# --- which submodule each top-level name is re-exported from --------------- #
SUBMODULES = [
    "data", "window", "preprocessing", "feature_engineering", "models",
    "model_selection", "model_ensemble", "forecasting", "pipeline",
    "evaluation", "output", "interpretation", "tests", "metrics", "reporting",
    "filters", "meta",
]
owner: dict[str, str] = {}
for name in SUBMODULES:
    module = getattr(mf, name, None)
    if module is None:
        continue
    for attr in dir(module):
        if not attr.startswith("_"):
            owner.setdefault(attr, name)

# subsystems with no correctness audit as of this inventory
EXPERIMENTAL_OWNERS = {"interpretation", "reporting", "meta"}

top = sorted(n for n in dir(mf) if not n.startswith("_"))
rows = []
for name in top:
    obj = getattr(mf, name, None)
    if inspect.ismodule(obj):
        tier = "submodule"
    elif name in mentioned:
        tier = "STABLE"
    elif owner.get(name) in EXPERIMENTAL_OWNERS:
        tier = "EXPERIMENTAL"
    elif name in owner:
        tier = "ADVANCED"
    else:
        tier = "ADVANCED"
    rows.append({
        "name": name,
        "tier": tier,
        "canonical": f"mf.{owner[name]}.{name}" if name in owner else f"mf.{name}",
        "kind": type(obj).__name__ if not inspect.isfunction(obj) else "function",
        "in_docs": name in mentioned,
    })

counts = collections.Counter(r["tier"] for r in rows)
print(f"top-level public symbols: {len(rows)}")
for tier, n in counts.most_common():
    print(f"  {tier:14s} {n}")
print()
print("STABLE (named in README / getting started):")
for r in rows:
    if r["tier"] == "STABLE":
        print(f"  mf.{r['name']}")
print()
print("EXPERIMENTAL (no correctness audit yet):")
exp = [r for r in rows if r["tier"] == "EXPERIMENTAL"]
for r in exp[:15]:
    print(f"  mf.{r['name']:34s} -> {r['canonical']}")
if len(exp) > 15:
    print(f"  ... and {len(exp) - 15} more")
print()
print("ADVANCED with a submodule home (top-level alias is redundant):")
adv = [r for r in rows if r["tier"] == "ADVANCED" and r["name"] in owner]
print(f"  {len(adv)} symbols; e.g.")
for r in adv[:8]:
    print(f"  mf.{r['name']:34s} -> {r['canonical']}")

out = REPO / "docs" / "api_inventory.json"
out.write_text(json.dumps({"symbols": rows, "counts": dict(counts)}, indent=1) + "\n")
print(f"\nwrote {out.relative_to(REPO)}")
