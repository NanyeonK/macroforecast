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

Two upward imports exist today. Both are function-local, so neither creates an
import cycle, and both are listed explicitly rather than tolerated silently --
this file is a ratchet against NEW violations, not a claim that the package is
already clean.

- `data/vintage.py` -> `pipeline.run._panel_fingerprint`. A layer-0 module
  reaching into a layer-3 **private** function. The fingerprinting belongs
  lower; moving it is a behaviour-preserving refactor left for its own change.
- `pipeline/run.py` -> `output.collect_provenance`. Less clear-cut: it is really
  a question of whether `output` sits above `pipeline` or beside it. Recorded so
  the decision is made deliberately rather than by accretion.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import macroforecast

ROOT = Path(macroforecast.__file__).parent

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
KNOWN_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset({
    ("data/vintage.py", "pipeline"),
    ("pipeline/run.py", "output"),
})


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
