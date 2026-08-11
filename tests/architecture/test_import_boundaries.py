"""The layering is a claim about the package; this checks it is still true.

`macroforecast` is arranged so a lower layer never needs to know about a higher
one. The value of that is not tidiness: it is that `models` can be read and
tested without knowing what a pipeline is, and that `forecasting.run()` stays
usable on its own -- which is how every replication under `docs/replication` is
written. One upward import quietly ends both properties, and nothing else in the
suite would notice.

Levels below were derived by reading what each package actually imports, not by
taste (`filters`, `metrics`, `meta` and `tests` import nothing from the package
at all, so they sit at the bottom).

The test reads import statements rather than importing modules, so a violation
is reported as the file and line that introduced it.

## Known exceptions

`KNOWN_EXCEPTIONS` is currently empty: no module imports a layer above its own.
Two did until 2026-08-09, and each was fixed by moving the function down rather
than by tolerating the import -- the panel fingerprint out of `pipeline.run` and
into `data/identity.py`, the git/environment probe out of `output` and into
`meta/provenance.py`. `docs/architecture.md` records why each belongs there.

The list stays because a future exception needs somewhere to be written down,
and a second test fails if a listed entry stops occurring, so it cannot rot into
fiction.

## What this does not check

The contract is directional, not acyclic. Same-layer imports are allowed, so the
layer comparison below says nothing about a cycle *within* a layer, and layer 1
has one: `feature_engineering` -> `model_selection` -> `model_ensemble` ->
`models` -> `feature_engineering`. It is an accepted boundary rather than an open
defect -- those four are peers by design, and the edge that closes the loop is
function-local. `docs/architecture.md` records the decision; what keeps it
harmless is checked here, by importing each of the four on its own.

It also says nothing about a cycle *inside* one package, because every such
import is same-package rather than same-layer. `forecasting` had one --
`runner` -> `policies` -> `policies.recursive` -> `runner`, closed by
`recursive` reaching back for `runner._test_feature_builder` -- which survived
because the reverse import sat at the bottom of `recursive.py` and so happened
to find a half-initialised `runner` with that name already bound. That is a
cycle held open by statement order, and the last section of this file now
checks the `forecasting` module graph directly so it cannot be reintroduced.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pytest

import macroforecast

ROOT = Path(macroforecast.__file__).parent

#: Repo root, so a fresh interpreter imports the tracked source rather than
#: whatever happens to be installed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: package -> layer. Lower may not import higher.
LAYERS: dict[str, int] = {
    # 0: no intra-package dependencies
    "analysis": 0,
    "filters": 0,
    "meta": 0,
    "metrics": 0,
    "tests": 0,
    "data": 0,
    "window": 0,
    # 1: build on data/window
    "data_analysis": 1,
    "preprocessing": 1,
    "feature_engineering": 1,
    "models": 1,
    "model_selection": 1,
    "model_ensemble": 1,
    # 2: build on the modelling layer
    "feature_analysis": 2,
    "feature_diagnostic": 2,
    "interpretation": 2,
    "forecasting": 2,
    # 3: orchestration
    "forecast_analysis": 3,
    "forecast_diagnostic": 3,
    "pipeline": 3,
    # 4: consume a finished run
    "evaluation": 4,
    "output": 4,
    # 5: consume the outputs
    "reporting": 5,
}

#: (module, imported package) pairs that predate this test. See the docstring.
#: Empty since A1 (2026-08-09). Both entries were lower layers reaching up for a
#: function that had no business living where it lived:
#:
#:   data/vintage.py   -> pipeline.run._panel_fingerprint
#:   pipeline/run.py   -> output.collect_provenance
#:
#: A fingerprint is a property of the data and now lives in data/identity.py; a
#: git/environment probe is not artifact writing and now lives in
#: meta/provenance.py. Neither move changed behaviour -- the digest is byte-identical
#: and output.collect_provenance re-exports the same object.
#:
#: Keep this empty. An entry added here is a layering violation the project decided to
#: live with, so it needs a reason in the architecture document, not just a tuple.
KNOWN_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset()


def _layer_of(module_path: Path) -> tuple[str, int] | None:
    try:
        relative = module_path.relative_to(ROOT)
    except ValueError:
        return None
    top = relative.parts[0]
    if top.endswith(".py"):
        top = top[:-3]
    if top not in LAYERS:
        return None
    return top, LAYERS[top]


#: ``_resolve_relative`` returns this when a relative import walks up to
#: ``macroforecast`` itself, e.g. ``from .. import pipeline`` inside
#: ``macroforecast/data/vintage.py``. The package being reached is then one of the
#: imported NAMES rather than the module path, so the caller reads ``node.names``.
ROOT_MARKER = "<root>"


def _resolve_relative(module_path: Path, level: int, module: str | None) -> str | None:
    """Resolve a relative import to the ``macroforecast.<top>`` package it reaches.

    ``from ..pipeline import run_pipeline`` inside ``macroforecast/data/vintage.py``
    is an upward import to a *sibling top-level package*, and skipping every
    relative import — as this file did until 2026-08-09 — let exactly that through.
    The comment claimed relative imports stay inside their own layer; that was an
    assumption, not a property of Python.

    Returns the top-level package name, or ``None`` when the import cannot leave
    the current one (``from .local import x``, or a level that walks past the root).
    """
    try:
        relative = module_path.relative_to(ROOT)
    except ValueError:
        return None
    # Package parts of the importing module: macroforecast/<a>/<b>/mod.py -> [<a>, <b>]
    parts = list(relative.parts[:-1])
    # level=1 is "this package"; each extra level walks one package up.
    up = level - 1
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    if module:
        return (base + [module.split(".")[0]])[0]
    if base:
        # ``from . import x`` / ``from .. import x`` that still lands inside a
        # package: the package itself is what the import is rooted at.
        return base[0]
    # Walked all the way to ``macroforecast`` itself: the imported NAMES are the
    # top-level packages, so the caller has to read them off ``node.names``.
    return ROOT_MARKER


def _imported_packages(tree: ast.AST, module_path: Path) -> list[tuple[str, int]]:
    """Every ``macroforecast.<top>`` this module imports, with its line.

    Absolute and relative forms both count. A relative import is resolved against
    the importing module's own package path, so an upward one is reported as the
    package it actually reaches rather than skipped.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("macroforecast."):
                    found.append((alias.name.split(".")[1], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                top = _resolve_relative(module_path, node.level, node.module)
                if top == ROOT_MARKER:
                    # ``from .. import pipeline`` -- the packages are the names.
                    for alias in node.names:
                        if alias.name in LAYERS:
                            found.append((alias.name, node.lineno))
                elif top is not None:
                    found.append((top, node.lineno))
                continue
            if node.module and node.module.startswith("macroforecast."):
                found.append((node.module.split(".")[1], node.lineno))
    return found


def _source_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and "_vendor" not in p.parts
    )


def _violations() -> list[tuple[str, str, int, str, int, int]]:
    out = []
    for path in _source_files():
        here = _layer_of(path)
        if here is None:
            continue
        name, level = here
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"{path}: {exc}")
        rel = str(path.relative_to(ROOT))
        for imported, lineno in _imported_packages(tree, path):
            if imported not in LAYERS or imported == name:
                continue
            if LAYERS[imported] > level:
                out.append((rel, name, level, imported, LAYERS[imported], lineno))
    return out


def test_no_new_module_imports_a_layer_above_its_own() -> None:
    unexpected = [
        f"{rel}:{lineno} ({name}, layer {level}) imports {imported} (layer {ilevel})"
        for rel, name, level, imported, ilevel, lineno in _violations()
        if (rel, imported) not in KNOWN_EXCEPTIONS
    ]
    assert not unexpected, (
        "a lower layer imported a higher one, which makes it unusable on its own:\n  "
        + "\n  ".join(unexpected)
    )


def test_the_known_exceptions_are_still_real() -> None:
    """If one gets fixed, this fails -- so the list cannot rot into fiction."""
    present = {(rel, imported) for rel, _, _, imported, _, _ in _violations()}
    stale = sorted(KNOWN_EXCEPTIONS - present)
    assert not stale, (
        "these are listed as known exceptions but no longer occur; remove them "
        f"from KNOWN_EXCEPTIONS: {stale}"
    )


def test_every_declared_layer_exists() -> None:
    """A typo in LAYERS would silently exempt a package from the check above."""
    missing = [
        name for name in LAYERS
        if not (ROOT / name).is_dir() and not (ROOT / f"{name}.py").is_file()
    ]
    assert not missing, f"LAYERS names packages that do not exist: {missing}"


def test_no_top_level_package_is_unclassified() -> None:
    """New packages must be placed, or the boundary check stops covering them."""
    tops = {
        p.name if p.is_dir() else p.stem
        for p in ROOT.iterdir()
        if (p.is_dir() and (p / "__init__.py").exists()) or p.suffix == ".py"
    }
    unclassified = sorted(t for t in tops - set(LAYERS) if not t.startswith("_"))
    assert not unclassified, (
        "add these to LAYERS at the level their imports allow:\n  "
        + "\n  ".join(unclassified)
    )


#: The layer-1 loop recorded in ``docs/architecture.md``. Same-layer edges are
#: permitted, so ``_violations()`` is silent about these by design; what makes the
#: loop harmless is that the edge closing it is function-local.
LAYER_1_LOOP = ("feature_engineering", "model_selection", "model_ensemble", "models")


@pytest.mark.parametrize("package", LAYER_1_LOOP)
def test_each_package_in_the_layer_1_loop_imports_on_its_own(package: str) -> None:
    """A fresh interpreter, because an in-process import would not show the failure.

    Hoisting ``_sparse_ic.py``'s function-local ``model_selection`` import to
    module level closes the loop at import time. Nothing else here would notice:
    the edge is same-layer, so it is not a violation, and ``import macroforecast``
    pulls the four in an order that happens to work. It breaks only when one of
    them is the *first* thing imported -- which is what a reader doing
    ``import macroforecast.models`` to check an estimator actually does.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import macroforecast.{package}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"macroforecast.{package} no longer imports on its own, which is what a "
        f"module-level import cycle inside layer 1 looks like:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "module, level, imported, expected, why",
    [
        # macroforecast/data/vintage.py
        ("data/vintage.py", 1, "panel", "data",
         "from .panel -- stays inside data"),
        ("data/vintage.py", 2, "pipeline", "pipeline",
         "from ..pipeline -- reaches a SIBLING top-level package, the case that was skipped"),
        ("data/vintage.py", 2, "window", "window",
         "from ..window -- sibling, resolved by layer like any absolute import"),
        ("data/vintage.py", 1, None, "data",
         "from . import x -- still inside data"),
        ("data/vintage.py", 2, None, ROOT_MARKER,
         "from .. import x -- walks up to macroforecast itself, so the package "
         "being reached is one of the imported NAMES, not the module path"),
        # a deeper module, so the level arithmetic is exercised
        ("forecasting/policies/base.py", 1, "helpers", "forecasting",
         "from .helpers -- inside forecasting"),
        ("forecasting/policies/base.py", 2, "runner", "forecasting",
         "from ..runner -- still inside forecasting"),
        ("forecasting/policies/base.py", 3, "output", "output",
         "from ...output -- leaves forecasting entirely"),
    ],
)
def test_relative_imports_resolve_to_the_package_they_reach(
    module, level, imported, expected, why
):
    """The blind spot this file had until 2026-08-09.

    Every relative import was skipped on the assumption that relative meant
    same-layer. It does not: ``from ..pipeline import run_pipeline`` inside
    ``data/vintage.py`` is a relative import to a sibling top-level package, and the
    guard waved it through. These cases pin the resolver directly, because a
    regression here would show up as the guard silently passing rather than as a
    failure.
    """
    resolved = _resolve_relative(ROOT / module, level, imported)
    assert resolved == expected, why


def test_an_upward_relative_import_is_caught_end_to_end():
    """The guard must reject a lower layer reaching up, however it is written."""
    source = "from ..pipeline import run_pipeline\n"
    tree = ast.parse(source)
    found = _imported_packages(tree, ROOT / "data" / "vintage.py")
    assert found == [("pipeline", 1)], found

    # The `from .. import pipeline` spelling of the same violation.
    root_form = ast.parse("from .. import pipeline\n")
    assert _imported_packages(root_form, ROOT / "data" / "vintage.py") == [
        ("pipeline", 1)
    ], "an upward import written as `from .. import <pkg>` slipped through"

    lower = LAYERS["data"]
    upper = LAYERS["pipeline"]
    assert upper > lower, (
        "the fixture assumes pipeline sits above data; if the layer map changed, "
        "this test needs a different pair rather than deleting"
    )


# ---------------------------------------------------------------------------
# Module-level cycles inside one package
# ---------------------------------------------------------------------------

#: Checked at module granularity because it is the package being decomposed:
#: policy strategies and stage helpers are being lifted out of ``runner.py`` a
#: piece at a time, and a half-moved piece is exactly what leaves a module
#: importing the module it was moved out of.
CYCLE_FREE_PACKAGE = "forecasting"

#: Imported one at a time in a fresh interpreter. These three were the members
#: of the cycle, so they are the ones whose independence is worth pinning.
INDEPENDENTLY_IMPORTABLE = (
    "macroforecast.forecasting.runner",
    "macroforecast.forecasting.policies",
    "macroforecast.forecasting.policies.recursive",
)


def _module_name(path: Path) -> str:
    """``forecasting/policies/recursive.py`` -> its dotted module name."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["macroforecast", *parts])


def _without_type_checking(tree: ast.Module) -> ast.Module:
    """The same tree with ``if TYPE_CHECKING:`` bodies dropped.

    An annotation-only import does not execute, so it cannot close a cycle at
    run time. Counting one would make this file report a cycle that does not
    exist -- and the usual fix for a real cycle is to move an import *into* such
    a block, which would then look like no fix at all.
    """

    class _Pruner(ast.NodeTransformer):
        def visit_If(self, node: ast.If):
            test = node.test
            guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if guarded:
                # ``else:`` still runs, so it is kept.
                return [self.visit(stmt) for stmt in node.orelse]
            return self.generic_visit(node)

    return ast.fix_missing_locations(_Pruner().visit(tree))


def _imported_modules(path: Path, known: set[str]) -> set[str]:
    """Every module in ``known`` that ``path`` imports at run time.

    ``from a.b import c`` is recorded as reaching both ``a.b`` and ``a.b.c``,
    because the name may be a submodule rather than an attribute and only the
    filter against ``known`` can tell. Relative imports are resolved against the
    importing module's own package, so a future ``from . import base`` counts
    the same as the absolute spelling.
    """
    tree = _without_type_checking(ast.parse(path.read_text(encoding="utf-8")))
    here = _module_name(path)
    package = here if path.name == "__init__.py" else here.rsplit(".", 1)[0]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                prefix = ".".join(parts)
                module = f"{prefix}.{node.module}" if node.module else prefix
            elif node.module:
                module = node.module
            else:  # pragma: no cover -- ``import`` with no module and no level
                continue
            found.add(module)
            for alias in node.names:
                found.add(f"{module}.{alias.name}")
    return {name for name in found if name in known and name != here}


def _package_module_graph(package: str) -> dict[str, set[str]]:
    """module -> the modules it imports, restricted to ``package``."""
    files = sorted(
        p
        for p in (ROOT / package).rglob("*.py")
        if "__pycache__" not in p.parts and "_vendor" not in p.parts
    )
    known = {_module_name(p) for p in files}
    return {_module_name(p): _imported_modules(p, known) for p in files}


def _a_cycle_in(graph: dict[str, set[str]]) -> list[str] | None:
    """One import cycle's members, or ``None`` if the graph is acyclic."""
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        return list(exc.args[1])
    return None


def test_no_module_level_import_cycle_inside_forecasting() -> None:
    """A module in a cycle cannot be read, tested or imported by itself.

    ``runner`` -> ``policies`` -> ``policies.recursive`` -> ``runner`` was real
    until 2026-08-11 and is what this catches. Nothing else in the suite would:
    the edges are same-package, so ``_violations()`` is silent by design, and
    the cycle imported fine, because the reverse import was written at the
    bottom of ``recursive.py`` where the half-built ``runner`` module already
    had the one name it wanted.
    """
    cycle = _a_cycle_in(_package_module_graph(CYCLE_FREE_PACKAGE))
    assert cycle is None, (
        f"macroforecast.{CYCLE_FREE_PACKAGE} has a module-level import cycle: "
        + " -> ".join(cycle or [])
        + "\nMove the shared code down into the module both sides can import "
        "(as _test_feature_builder was moved to feature_stage.py) rather than "
        "importing it back from the module it was moved out of."
    )


def test_the_cycle_check_would_still_see_the_edge_that_was_removed() -> None:
    """Otherwise the check above could pass by reading nothing at all.

    A cycle test that silently stops finding imports -- a moved package, a
    spelling it does not resolve -- keeps passing forever and looks like a
    guarantee. So: the real forward edges must be visible, and putting the one
    deleted reverse edge back must bring the cycle back.
    """
    graph = _package_module_graph(CYCLE_FREE_PACKAGE)
    runner = "macroforecast.forecasting.runner"
    policies = "macroforecast.forecasting.policies"
    recursive = "macroforecast.forecasting.policies.recursive"

    missing = [m for m in (runner, policies, recursive) if m not in graph]
    assert not missing, f"the graph did not reach these modules at all: {missing}"
    assert policies in graph[runner], "runner -> policies is no longer seen"
    assert recursive in graph[policies], "policies -> policies.recursive is no longer seen"

    graph[recursive] = graph[recursive] | {runner}
    assert _a_cycle_in(graph) is not None, (
        "restoring policies.recursive -> runner did not produce a cycle, so this "
        "file is no longer able to detect the one it is here to prevent"
    )


def test_type_checking_only_imports_are_not_counted_as_runtime_edges() -> None:
    """The documented escape hatch has to actually work.

    A cycle broken by demoting an import to ``TYPE_CHECKING`` must read as
    broken here, or the fix would fail the test that asked for it.
    """
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from macroforecast.forecasting.runner import run\n"
        "else:\n"
        "    from macroforecast.forecasting.types import ForecastResult\n"
    )
    tree = _without_type_checking(ast.parse(source))
    reached = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "macroforecast.forecasting.runner" not in reached, (
        "an annotation-only import was counted as a runtime edge"
    )
    assert "macroforecast.forecasting.types" in reached, (
        "the else-branch import is real and must still be counted"
    )


@pytest.mark.parametrize("module", INDEPENDENTLY_IMPORTABLE)
def test_each_cycle_member_imports_on_its_own(module: str) -> None:
    """A fresh interpreter, each module first rather than reached via another.

    This is the property the cycle put at risk rather than the cycle itself:
    while it existed these still imported, but only because ``recursive`` asked
    for the one ``runner`` attribute that was already bound by the time it ran.
    Any later change to what ``recursive`` needs, or to where in ``runner`` it
    is defined, turns that into an ImportError -- and the failure would surface
    in whichever module a reader happened to import first, not at the edit.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{module} does not import on its own:\n{result.stderr}"
    )
