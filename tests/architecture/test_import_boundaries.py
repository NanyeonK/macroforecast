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


def _imported_packages(tree: ast.AST) -> list[tuple[str, int]]:
    """Every ``macroforecast.<top>`` this module imports, with its line."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("macroforecast."):
                    found.append((alias.name.split(".")[1], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: stays inside its own layer
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
        for imported, lineno in _imported_packages(tree):
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
