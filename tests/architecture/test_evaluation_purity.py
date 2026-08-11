"""``pipeline/evaluate.py`` is arithmetic over a forecast frame; this checks it stays so.

Evaluation used to call ``load_fred_series()`` to turn a named subsample mask
into a boolean state series, which made scoring one *fixed* table depend on the
network, the FRED cache, and the filesystem: the same frame could score
differently on a Tuesday, or not at all on a train. Named masks are resolved
before evaluation now (``pipeline/evaluation_inputs.py``), and this file is the
ratchet on that.

It checks the property from both ends, because either alone is weak:

* statically -- the evaluator's source imports no loader and no I/O module, so
  the coupling cannot come back as an import someone forgot to look at;
* dynamically -- with ``load_fred_series`` replaced by something that raises, a
  full named-mask evaluation over already-resolved inputs still runs, twice,
  identically. That is the property users actually have, and it is the one a
  "moved the call one function down" refactor would fail.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    EvalSpec,
    SubsampleWindow,
    evaluate,
    resolve_evaluation_inputs,
)

# ``macroforecast.pipeline.evaluate`` is two things: the submodule, and the
# FUNCTION the package re-exports under the same name. ``mf.pipeline.evaluate``
# resolves to whichever the import order left on the package, so the module is
# fetched by name here rather than off the attribute -- reading ``__file__`` from
# the attribute happens to work only while something else has already imported
# the submodule, and fails on a fresh import.
eval_mod = importlib.import_module("macroforecast.pipeline.evaluate")
EVALUATOR = Path(eval_mod.__file__)

#: Modules an evaluator has no business importing: the package's own data layer,
#: and the stdlib/third-party doors to disk, network, and process state. Import
#: any of these and "evaluating this frame" stops being a function of the frame.
FORBIDDEN_IMPORTS = frozenset(
    {
        "macroforecast.data",
        "glob",
        "http",
        "io",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "tempfile",
        "urllib",
    }
)

#: Names that mean I/O wherever they appear, including behind an alias or inside
#: a function body where an import check would not look.
FORBIDDEN_CALLS = frozenset(
    {
        "load_fred_series",
        "open",
        "read_csv",
        "read_parquet",
        "to_csv",
        "to_parquet",
        "urlopen",
    }
)


def _evaluator_tree() -> ast.AST:
    return ast.parse(EVALUATOR.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


def _forbidden(module: str) -> str | None:
    """The forbidden root *module* sits under, if any (``os.path`` -> ``os``)."""
    parts = module.split(".")
    for depth in range(1, len(parts) + 1):
        candidate = ".".join(parts[:depth])
        if candidate in FORBIDDEN_IMPORTS:
            return candidate
    return None


def test_the_evaluator_imports_no_loader_and_no_io_module() -> None:
    violations = [
        f"{EVALUATOR.name}:{lineno} imports {module!r} (forbidden: {root!r})"
        for module, lineno in _imported_modules(_evaluator_tree())
        if (root := _forbidden(module)) is not None
    ]
    assert not violations, (
        "pipeline/evaluate.py reached for data or I/O; evaluating a fixed "
        "forecast frame must not depend on the network, the cache, or the "
        "filesystem:\n  " + "\n  ".join(violations)
    )


def test_the_evaluator_never_names_a_loader() -> None:
    """An import check alone misses ``mf.data.load_fred_series(...)`` written inline."""
    tree = _evaluator_tree()
    named: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            named.append(f"{EVALUATOR.name}:{node.lineno} uses {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
            named.append(f"{EVALUATOR.name}:{node.lineno} uses .{node.attr}")
    assert not named, (
        "pipeline/evaluate.py names an I/O entry point:\n  " + "\n  ".join(named)
    )


def test_the_evaluator_module_exposes_no_loader_attribute() -> None:
    """The symbol tests used to monkeypatch is gone, not merely unused."""
    assert not hasattr(eval_mod, "load_fred_series")


def test_the_public_evaluate_name_is_the_callable_not_the_module() -> None:
    """The overloaded name resolves the way callers depend on, whatever imported first.

    ``from macroforecast.pipeline import evaluate`` must give the function --
    users call it -- while the module of the same name stays reachable by import.
    Asserting both keeps this file honest about which one it is reading, and
    would catch a re-export that stopped shadowing the submodule.
    """
    assert callable(evaluate)
    assert evaluate is eval_mod.evaluate
    assert evaluate is not eval_mod
    assert eval_mod.__name__ == "macroforecast.pipeline.evaluate"
    # Signature, not just callability: ``inputs`` is the seam this file exists for.
    assert "inputs" in inspect.signature(evaluate).parameters


RESOLVER_MODULE = "macroforecast.pipeline.evaluation_inputs"


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _type_checking_import_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            lines.update(
                child.lineno
                for child in ast.walk(node)
                if isinstance(child, (ast.Import, ast.ImportFrom))
            )
    return lines


def test_the_evaluator_reaches_the_resolver_only_for_type_checking() -> None:
    """The dependency runs resolver -> evaluator; back the other way it is types only.

    The resolver imports the evaluator's date/frequency helpers at run time so
    that the rule choosing ``USREC`` vs ``USRECQ`` has exactly one definition.
    That is only safe while the evaluator's reference to the resolver stays
    inside ``if TYPE_CHECKING`` -- a runtime import there would both close the
    cycle and put a FRED loader back in the evaluator's import graph.
    """
    tree = _evaluator_tree()
    guarded = _type_checking_import_lines(tree)
    unguarded = [
        f"{EVALUATOR.name}:{node.lineno} imports {node.module!r} at run time"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == RESOLVER_MODULE
        and node.lineno not in guarded
    ]
    assert not unguarded, "\n  ".join(unguarded)


def test_the_shared_date_rules_have_one_definition() -> None:
    """Same objects, not copies: the resolver cannot drift from the evaluator."""
    inputs_mod = importlib.import_module(RESOLVER_MODULE)
    for helper in ("_target_mask_frequency", "_series_summary", "_eval_subsamples"):
        assert getattr(inputs_mod, helper) is getattr(eval_mod, helper), helper
    assert inputs_mod._SUBSAMPLE_DATE_COLUMN == eval_mod._SUBSAMPLE_DATE_COLUMN


def _master(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for origin, date in enumerate(dates):
        actual = 1.0 + 0.05 * origin + 0.1 * np.sin(origin / 5.0)
        rows.extend(
            [
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "AR", "prediction": actual + 0.30, "actual": actual,
                },
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "OLS", "prediction": actual + 0.08 + 0.01 * (origin % 3),
                    "actual": actual,
                },
            ]
        )
    return pd.DataFrame(rows)


def _fred_bundle(series_id: str, index: pd.DatetimeIndex, values: list[int]):
    panel = pd.DataFrame({series_id: values}, index=index)
    panel.index.name = "date"
    return mf.data.DataBundle(
        panel=panel,
        metadata={
            "dataset": "fred_series",
            "series_id": series_id,
            "artifact": {
                "source_url": f"https://example.test/{series_id}.csv",
                "local_path": f"/tmp/{series_id}.csv",
                "file_sha256": f"sha-{series_id}",
                "cache_hit": True,
            },
        },
    )


def test_evaluation_runs_and_repeats_identically_once_the_loader_is_dead(monkeypatch):
    """The property the refactor exists for: resolved inputs + a broken loader still score.

    Resolution happens once, up front. After that the loader is replaced with a
    function that raises on sight -- standing in for an unplugged network, a
    purged cache, or a FRED outage -- and the SAME frame is evaluated twice. Both
    runs must succeed and agree exactly, tables and mask provenance alike.
    """
    import macroforecast.pipeline.evaluation_inputs as inputs_mod

    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    master = _master(dates)
    spec = SimpleNamespace(
        evaluation=EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={
                "recession": SubsampleWindow(mask="nber_recession"),
                "expansion": SubsampleWindow(mask="nber_expansion"),
            },
        ),
        combinations=(),
        arms=(),
        seed=42,
    )

    monkeypatch.setattr(
        inputs_mod,
        "load_fred_series",
        lambda series_id, **_kwargs: _fred_bundle(
            series_id, dates, [1 if idx < 36 else 0 for idx in range(len(dates))]
        ),
    )
    inputs = resolve_evaluation_inputs(master, spec)

    def dead_loader(*_args, **_kwargs):
        raise AssertionError(
            "evaluate() reached a data loader; evaluation must be pure over the "
            "forecast frame and its already-resolved inputs"
        )

    monkeypatch.setattr(inputs_mod, "load_fred_series", dead_loader)
    monkeypatch.setattr(mf.data, "load_fred_series", dead_loader)
    monkeypatch.setattr(mf.data.loaders, "load_fred_series", dead_loader)

    first = evaluate(master, spec, inputs=inputs)
    second = evaluate(master, spec, inputs=inputs)

    assert not first["accuracy"].empty
    assert set(first["accuracy"]["subsample"]) == {"recession", "expansion"}
    for table in ("accuracy", "significance", "mcs", "density", "calibration"):
        pd.testing.assert_frame_equal(first[table], second[table])
    attr = "macroforecast_subsample_provenance"
    assert first["forecasts"].attrs[attr] == second["forecasts"].attrs[attr]
    assert first["forecasts"].attrs[attr]["recession"]["mask_summary"]["series_id"] == "USREC"


def test_a_named_mask_without_resolved_inputs_says_so_instead_of_loading(monkeypatch):
    """No silent fallback: the evaluator asks the caller to resolve, it does not fetch."""
    import macroforecast.pipeline.evaluation_inputs as inputs_mod

    def dead_loader(*_args, **_kwargs):
        raise AssertionError("evaluate() must not load anything")

    monkeypatch.setattr(inputs_mod, "load_fred_series", dead_loader)
    monkeypatch.setattr(mf.data, "load_fred_series", dead_loader)

    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    spec = SimpleNamespace(
        evaluation=EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={"recession": SubsampleWindow(mask="nber_recession")},
        ),
        combinations=(),
        arms=(),
        seed=42,
    )

    with pytest.raises(ValueError, match="resolve_evaluation_inputs"):
        evaluate(_master(dates), spec)
