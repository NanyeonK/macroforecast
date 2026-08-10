"""Inventory the top-level API from the sources that declare it.

Nothing is removed here, and nothing is judged here. The output is the evidence
a human decides from.

Why this tool exists in this shape
----------------------------------
An earlier revision resolved a symbol's owner by scanning ``dir()`` of a
hard-coded submodule list and taking the first module that happened to expose
the name. That is not an authority -- it is an accident of import order, and it
produced two measurable errors:

* ``axis_contribution`` lives in ``macroforecast.analysis``, which was missing
  from the hard-coded list, so it was reported with no submodule home at all.
* ``Any``, ``annotations`` and ``import_module`` are globals of the *root*
  ``macroforecast/__init__.py`` (``from typing import Any`` and friends). The
  scan attributed them to ``mf.tests``, ``mf.data`` and ``mf.evaluation``
  because those submodules also happen to expose them.

So ownership now comes from sources that actually declare it:

1. ``macroforecast._LAZY_EXPORTS`` -- the root's own name -> home-submodule map.
   This is the authority for a canonical path.
2. ``macroforecast._LAZY_MODULES`` -- the namespace exports.
3. each submodule's ``__all__`` -- corroboration that the home module really
   claims the name as public API.
4. ``object.__module__`` -- corroboration only. It is *not* reliable for type
   aliases or plain constants, and the inventory records where it disagrees
   rather than pretending the disagreement is an error.

Three surfaces, kept separate
-----------------------------
``dir(mf)`` conflates things that a stability decision must not conflate, so the
inventory splits them:

* ``supported`` -- the ``mf.__all__`` rows. This is the surface a stability
  decision is actually about.
* ``special`` -- deliberately public, deliberately outside ``__all__``.
  ``__version__`` is the only member: it is stable and documented, but an
  underscore-name filter over ``dir()`` silently drops it.
* ``non_api_globals`` -- public in ``dir(mf)``, absent from ``__all__``, and not
  API: stdlib names bound at root module scope.

Facts, not policy
-----------------
This inventory reports only what the package declares: which submodule owns a
name, whether the name is a lazy export or a namespace, what kind of object it
is, whether the entry docs reference it, and where the corroborating signals
disagree.

It deliberately assigns **no stability tier**. A tier says what the package
should promise to keep working, which is a policy judgement no generator can
make -- and an earlier revision proved the point: it hard-coded ``meta`` as an
experimental subsystem, so seven ``meta`` aliases were labelled EXPERIMENTAL
while ``docs/api_tiers.md`` described ``interpretation`` and ``reporting`` as
the experimental subsystems, neither of which had a single row. The same
revision used "submodule" as a tier when it is an export kind, not a promise.

Tiers live in ``docs/api_tiers.md``, which is the adopted policy a human owns.
This file is the evidence that policy is decided from.

Usage
-----
    python -m tools.api_inventory                  # regenerate the JSON
    python -m tools.api_inventory --check          # fail on drift, write nothing
"""
from __future__ import annotations

import argparse
import collections
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

__all__ = ["SCHEMA", "SCHEMA_VERSION", "build_inventory", "check", "main", "render"]

SCHEMA = "macroforecast.api_inventory"
# 3: rows carry a factual ``export_kind``; the subjective ``tier`` is gone.
SCHEMA_VERSION = 3

DEFAULT_OUTPUT = Path("docs/api_inventory.json")

# Files that decide whether a name counts as "documented as a top-level entry
# point". Only a literal ``mf.<name>`` mention counts; a
# ``from macroforecast.pipeline import Arm`` line teaches a submodule path and
# is deliberately not credited to the top-level surface.
DOC_SOURCES = ("README.md", "docs/guide/getting_started.md", "docs/index.md")


def _repo_root() -> Path:
    """Return the repository root, derived from the installed package."""

    import macroforecast as mf

    return Path(mf.__file__).parent.parent


def _documented_names(repo: Path) -> set[str]:
    """Names written as ``mf.<name>`` anywhere in the user-facing entry docs."""

    text = ""
    for rel in DOC_SOURCES:
        path = repo / rel
        if path.exists():
            text += path.read_text(encoding="utf-8")
    return set(re.findall(r"mf\.([A-Za-z_][A-Za-z0-9_]*)", text))


def _kind(obj: object) -> str:
    """Describe an object's kind.

    Deliberately identical to the previous revision so that values already
    quoted in ``docs/api_tiers.md`` (for example ``Split`` -> ``GenericAlias``)
    stay correct.
    """

    if inspect.isfunction(obj):
        return "function"
    return type(obj).__name__


def _supported_rows(mf: Any, documented: set[str]) -> list[dict[str, Any]]:
    """Build one row per name in ``mf.__all__``, using declared ownership."""

    lazy_exports: dict[str, str] = mf._LAZY_EXPORTS
    lazy_modules: tuple[str, ...] = mf._LAZY_MODULES

    rows: list[dict[str, Any]] = []
    for name in sorted(mf.__all__):
        obj = getattr(mf, name)
        is_module = name in lazy_modules

        if is_module:
            # A namespace export owns itself; there is nothing to corroborate.
            owner: str | None = None
            canonical = f"mf.{name}"
            in_owner_all: bool | None = None
            declared_module: str | None = None
            module_attr_agrees: bool | None = None
        else:
            # Authority 1: the root's own name -> home-submodule mapping.
            owner = lazy_exports[name].lstrip(".")
            canonical = f"mf.{owner}.{name}"

            # Authority 3: does the home submodule itself claim the name?
            home = importlib.import_module(f"macroforecast.{owner}")
            home_all = getattr(home, "__all__", None)
            in_owner_all = None if home_all is None else name in home_all

            # Authority 4: corroboration only. Type aliases report ``typing`` or
            # ``builtins`` and plain constants have no ``__module__`` at all, so
            # a disagreement is recorded, not treated as a failure.
            declared_module = getattr(obj, "__module__", None)
            if declared_module is None:
                module_attr_agrees = None
            else:
                home_pkg = f"macroforecast.{owner}"
                module_attr_agrees = declared_module == home_pkg or (
                    declared_module.startswith(f"{home_pkg}.")
                )

        rows.append(
            {
                "name": name,
                "surface": "supported",
                "export_kind": "namespace" if is_module else "lazy_export",
                "canonical": canonical,
                "owner": owner,
                "kind": _kind(obj),
                "in_docs": name in documented,
                "in_owner_all": in_owner_all,
                "declared_module": declared_module,
                "module_attr_agrees": module_attr_agrees,
            }
        )
    return rows


def _special_rows(mf: Any, documented: set[str]) -> list[dict[str, Any]]:
    """Deliberately public names that ``__all__`` and a ``dir()`` filter miss."""

    rows: list[dict[str, Any]] = []
    if hasattr(mf, "__version__"):
        rows.append(
            {
                "name": "__version__",
                "surface": "special",
                "canonical": "mf.__version__",
                "kind": _kind(mf.__version__),
                "value": mf.__version__,
                # Recorded on the same footing as a supported row: this surface
                # is referenced by the entry docs, so the file should say so
                # rather than leave the reader to assume it is undocumented.
                "in_docs": "__version__" in documented,
                "reason": (
                    "Stable, documented, and set directly in the root module. It "
                    "is absent from __all__ and, because the name is "
                    "underscore-prefixed, invisible to a dir() scan that filters "
                    "on a leading underscore."
                ),
            }
        )
    return rows


def _non_api_global_rows(mf: Any) -> list[dict[str, Any]]:
    """Public in ``dir(mf)``, absent from ``__all__``, and not API.

    These are stdlib names bound at root ``__init__`` scope. Their real
    provenance is the root module itself; any submodule attribution is an
    artefact of scanning ``dir()`` rather than a declaration.
    """

    public_dir = {name for name in dir(mf) if not name.startswith("_")}
    leaked = sorted(public_dir - set(mf.__all__))

    rows: list[dict[str, Any]] = []
    for name in leaked:
        obj = mf.__dict__.get(name)
        rows.append(
            {
                "name": name,
                "surface": "non_api_global",
                "canonical": f"mf.{name}",
                "kind": _kind(obj),
                "bound_in_root_globals": name in mf.__dict__,
                "declared_module": getattr(obj, "__module__", None),
                "reason": (
                    "Bound at macroforecast/__init__.py module scope by an "
                    "import statement the lazy loader needs. Not exported, not "
                    "documented, and not owned by any submodule."
                ),
            }
        )
    return rows


def build_inventory() -> dict[str, Any]:
    """Collect the full inventory. Deterministic; performs no writes."""

    import macroforecast as mf

    repo = _repo_root()
    documented = _documented_names(repo)

    supported = _supported_rows(mf, documented)
    special = _special_rows(mf, documented)
    non_api = _non_api_global_rows(mf)

    # Derived from the rows, not from the mappings, so that comparing the two
    # below is a real cross-check rather than a restatement.
    export_kinds = collections.Counter(row["export_kind"] for row in supported)
    public_dir = sum(1 for name in dir(mf) if not name.startswith("_"))

    # Every count below is derived from the rows above, so the file cannot
    # report a total that its own contents contradict.
    counts: dict[str, Any] = {
        "supported": len(supported),
        "lazy_exports": len(mf._LAZY_EXPORTS),
        "lazy_modules": len(mf._LAZY_MODULES),
        "export_kinds": {kind: export_kinds[kind] for kind in sorted(export_kinds)},
        "documented": sum(1 for row in supported if row["in_docs"]),
        "special": len(special),
        "non_api_globals": len(non_api),
        "public_dir": public_dir,
    }

    # Stated so a reader can check the arithmetic without re-deriving it, and so
    # a regression test has something to assert against.
    identities = {
        "supported == lazy_exports + lazy_modules": (
            counts["supported"] == counts["lazy_exports"] + counts["lazy_modules"]
        ),
        "export_kind rows == lazy_exports + lazy_modules": (
            export_kinds["lazy_export"] == counts["lazy_exports"]
            and export_kinds["namespace"] == counts["lazy_modules"]
        ),
        "public_dir == supported + non_api_globals": (
            counts["public_dir"] == counts["supported"] + counts["non_api_globals"]
        ),
    }

    # Honest exceptions rather than a forced invariant: record where the weaker
    # corroborating signal does not line up with declared ownership.
    module_attr_exceptions = sorted(
        row["name"] for row in supported if row["module_attr_agrees"] is False
    )
    module_attr_absent = sorted(
        row["name"]
        for row in supported
        if row["owner"] is not None and row["module_attr_agrees"] is None
    )
    owner_all_exceptions = sorted(
        row["name"] for row in supported if row["in_owner_all"] is False
    )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "package_version": mf.__version__,
        "authorities": [
            "macroforecast._LAZY_EXPORTS",
            "macroforecast._LAZY_MODULES",
            "<submodule>.__all__",
            "object.__module__ (corroboration only)",
        ],
        "counts": counts,
        "identities": identities,
        "corroboration": {
            "owner_all_exceptions": owner_all_exceptions,
            "module_attr_exceptions": module_attr_exceptions,
            "module_attr_absent": module_attr_absent,
        },
        "surfaces": {
            "supported": supported,
            "special": special,
            "non_api_globals": non_api,
        },
    }


def render(inventory: dict[str, Any] | None = None) -> str:
    """Serialise the inventory to the exact bytes that belong on disk."""

    if inventory is None:
        inventory = build_inventory()
    return json.dumps(inventory, indent=1) + "\n"


def check(path: str | Path = DEFAULT_OUTPUT) -> tuple[bool, list[str]]:
    """Return ``(ok, messages)`` for a drift check. Never writes."""

    out = Path(path)
    wanted = render()

    if not out.exists():
        return False, [f"missing: {out.as_posix()}"]

    current = out.read_text(encoding="utf-8")
    if current == wanted:
        return True, []
    return False, [f"changed: {out.as_posix()}"]


def _summary_lines(inventory: dict[str, Any]) -> list[str]:
    """Human-readable summary, kept separate from the machine-readable file."""

    counts = inventory["counts"]
    supported = inventory["surfaces"]["supported"]
    special_names = ", ".join(r["name"] for r in inventory["surfaces"]["special"])
    leaked_names = ", ".join(r["name"] for r in inventory["surfaces"]["non_api_globals"])
    lines = [
        f"supported (mf.__all__):    {counts['supported']}"
        f"  = {counts['lazy_exports']} lazy exports"
        f" + {counts['lazy_modules']} lazy modules",
        f"special (outside __all__): {counts['special']}  ({special_names or '-'})",
        f"public-but-not-API:        {counts['non_api_globals']}  ({leaked_names or '-'})",
        f"public dir(mf):            {counts['public_dir']}",
        "",
        "export kind over the supported surface:",
    ]
    for kind, number in counts["export_kinds"].items():
        lines.append(f"  {kind:14s} {number}")

    # Documentation evidence, not a tier: which names the entry docs actually
    # reference. What the package should promise is decided in docs/api_tiers.md.
    documented = [r["name"] for r in supported if r["in_docs"]]
    lines += ["", f"referenced as mf.<name> in the entry docs: {counts['documented']}"]
    lines += [f"  mf.{name}" for name in documented]
    for row in inventory["surfaces"]["special"]:
        if row.get("in_docs"):
            lines.append(f"  mf.{row['name']}  (special surface)")

    corroboration = inventory["corroboration"]
    lines += [
        "",
        "corroboration exceptions (recorded, not treated as errors):",
        f"  not in owner __all__: {corroboration['owner_all_exceptions'] or 'none'}",
        f"  __module__ disagrees: {corroboration['module_attr_exceptions'] or 'none'}",
        f"  __module__ absent:    {corroboration['module_attr_absent'] or 'none'}",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.api_inventory",
        description="Regenerate or check the committed top-level API inventory.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help=f"Inventory path (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed inventory differs. Writes nothing.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    output = Path(args.output)

    if args.check:
        ok, messages = check(output)
        if ok:
            print(f"[tools.api_inventory] {output} is up to date")
            return 0
        print(f"[tools.api_inventory] {output} is out of date", file=sys.stderr)
        for message in messages:
            print(message, file=sys.stderr)
        print(f"run: python -m tools.api_inventory {output.as_posix()}", file=sys.stderr)
        return 1

    inventory = build_inventory()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(inventory), encoding="utf-8")
    for line in _summary_lines(inventory):
        print(line)
    print(f"\nwrote {output.as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
