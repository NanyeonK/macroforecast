"""F-062: the selection-history sidecar is RFC 8259 JSON, and legacy ones still load.

``append_origin_selection_records`` encoded each record with ``json.dumps`` and its
default ``allow_nan=True``, so a non-finite leaf was written as a bare ``NaN`` /
``Infinity`` / ``-Infinity`` token. Those are ECMAScript literals, not JSON: a
strict RFC 8259 parser (R's ``jsonlite``, Go's ``encoding/json``, ``jq``) rejects
the line outright, so a sidecar that Python round-tripped fine was unreadable to
every other tool.

The reachable case is narrow and worth stating exactly. Records arrive here FLAT
and string-keyed -- ``runner._flatten_scalar_history_items`` descends mappings
only and keeps only already-scalar entries, so a list-valued param never reaches
the writer. What can still arrive non-finite is one of a record's own scalar
FIELDS: ``value``, from a fit-params entry that is a non-finite ``float`` (a
user-supplied hyperparameter, say) or a custom scalar whose ``.item()`` returns
one, or ``score``, read from a fitted step's ``scores`` mapping. One such leaf is
enough to emit a bare token and break the line. The nested-container cases
exercised below are BOUNDARY HARDENING for caller-supplied records, not a claim
that such a record has been seen on disk.

Three separate things are pinned below, because F-062 is a boundary fix and a
boundary has two sides plus a failure policy.

*Writing* is now strict: the record is sanitized BEFORE encoding (non-finite
numeric leaves become JSON ``null``, at any nesting depth, through Python floats,
numpy scalars and single-element arrays) and then encoded with
``allow_nan=False``. Every test here reads the bytes back through a decoder whose
``parse_constant`` RAISES, so a test cannot pass merely because Python's own
parser is tolerant of the tokens the fix exists to eliminate.

*Reading* stays deliberately lenient, and that asymmetry is the compatibility
contract: sidecars already on disk are full of bare ``NaN``, and refusing them
would invalidate every legacy checkpoint directory for a defect in the writer.
They keep loading, normalized to missing, and a legacy directory still resumes.

*Failing* is CLOSED: an unencodable record raises, cleans up its temp file and
leaves any previous final sidecar untouched. ``checkpoint.py``'s module docstring
carries the argument for why that is safe -- it rests entirely on the runner
writing the sidecar BEFORE the origin's parquet, so the failure lands while the
origin is still incomplete and recomputable. That order is itself pinned here,
behaviourally and structurally.
"""
from __future__ import annotations

import ast
import json
import sys
import textwrap
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import checkpoint as ckpt
from macroforecast.forecasting import runner as runner_mod


# --------------------------------------------------------------------------- #
# A decoder that refuses what the fix exists to stop emitting
# --------------------------------------------------------------------------- #
def _reject_constant(name: str) -> object:
    """``parse_constant`` hook: any bare non-finite token fails the test."""
    raise AssertionError(
        f"sidecar contains the non-standard JSON constant {name!r}; RFC 8259 has "
        "no NaN/Infinity literal, so a strict third-party parser rejects this line"
    )


_STRICT_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _strict_lines(path: Path) -> list[dict[str, Any]]:
    """Every nonblank sidecar line, parsed as strict RFC 8259 JSON."""
    text = path.read_text(encoding="utf-8")
    return [_STRICT_DECODER.decode(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _records(origin_pos: int) -> list[dict[str, Any]]:
    """One healthy lean forecast row, so a sidecar is never an orphan."""
    return [
        {
            "target": "y",
            "horizon": 1,
            "origin": origin_pos,
            "origin_pos": origin_pos,
            "date": pd.Timestamp("2000-01-31"),
            "model": "ridge",
            "prediction": float(origin_pos),
            "actual": 0.0,
        }
    ]


def _selection(origin_pos: int, **overrides: Any) -> dict[str, Any]:
    """One selection-history record in the shape the runner emits."""
    record: dict[str, Any] = {
        "target": "y",
        "arm": "ridge",
        "horizon": 1,
        "origin_pos": origin_pos,
        "kind": "feature",
        "name": "x1",
        "model": "ridge",
        "score": 1.0,
        "source": "screen",
    }
    record.update(overrides)
    return record


def _cell(tmp_path: Path, *positions: int) -> Path:
    """A checkpoint directory with one healthy parquet per origin position."""
    directory = tmp_path / "h1"
    for position in positions:
        ckpt.append_origin_records(directory, position, _records(position))
    return directory


def _sidecar_path(directory: Path, origin_pos: int) -> Path:
    return directory / f"origin_{origin_pos}_selection.jsonl"


def _legacy_sidecar(directory: Path, origin_pos: int) -> Path:
    """A sidecar exactly as the PRE-F-062 writer produced it: bare ``NaN``."""
    path = _sidecar_path(directory, origin_pos)
    path.write_text(
        json.dumps(_selection(origin_pos, score=float("nan")), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _identity() -> ckpt.CheckpointRunIdentity:
    return ckpt.CheckpointRunIdentity(
        digest="f" * 64,
        complete=True,
        opaque_fields=(),
        components={"model": "ridge"},
    )


def _tmp_leftovers(directory: Path) -> list[Path]:
    """Crashed-write dotfiles the writer promised to clean up."""
    return sorted(path for path in directory.glob(".*") if path.is_file())


# --------------------------------------------------------------------------- #
# Writing: non-finite leaves become JSON null, at every depth and every dtype
# --------------------------------------------------------------------------- #
def test_python_non_finite_becomes_json_null(tmp_path: Path) -> None:
    """A Python ``inf``/``nan`` leaf strict-parses, and lands as null not a string."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [
            _selection(0, score=float("inf")),
            _selection(0, name="x2", score=float("-inf")),
            _selection(0, name="x3", score=float("nan")),
        ],
    )

    lines = _strict_lines(_sidecar_path(directory, 0))

    assert [line["score"] for line in lines] == [None, None, None]
    # Not "Infinity", not "nan": a string here would silently become a str column.
    assert all(not isinstance(line["score"], str) for line in lines)


def test_finite_values_are_left_alone(tmp_path: Path) -> None:
    """Sanitizing must not touch the values that were always encodable."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [_selection(0, score=0.25, value=-3, name="x1", kind="feature")],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["score"] == 0.25
    assert line["value"] == -3
    assert line["name"] == "x1"


def test_numpy_non_finite_scalars_are_sanitized(tmp_path: Path) -> None:
    """``np.float32`` is NOT a ``float`` subclass, so it needs its own handling."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [
            _selection(0, score=np.float32("inf")),
            _selection(0, name="x2", score=np.float64("nan")),
            _selection(0, name="x3", score=np.float16("-inf")),
            _selection(0, name="x4", score=np.array([np.inf])),
        ],
    )

    lines = _strict_lines(_sidecar_path(directory, 0))

    assert [line["score"] for line in lines] == [None, None, None, None]


def test_nested_non_finite_leaf_is_sanitized(tmp_path: Path) -> None:
    """Boundary hardening: a non-finite leaf is neutralized at any nesting depth.

    The runner's own records are flat, so this shape is not one it produces; the
    walk exists because this function accepts caller-supplied records and a leaf
    missed at depth would put a bare token back on the line.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [
            _selection(
                0,
                kind="param",
                value={
                    "outer": {
                        "levels": [1.0, float("nan"), np.float32("inf")],
                        "nested": ({"deep": float("-inf")},),
                    }
                },
            )
        ],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    outer = line["value"]["outer"]
    assert outer["levels"] == [1.0, None, None]
    assert outer["nested"][0]["deep"] is None


def test_numpy_longdouble_does_not_recurse(tmp_path: Path) -> None:
    """``np.longdouble.item()`` returns another ``np.longdouble`` -- a fixed point.

    The pre-F-062 writer handed that straight back to ``json`` from
    ``_json_default``, which handed the same object back to ``_json_default``,
    until the interpreter gave up with ``RecursionError`` -- for a FINITE value
    too. Unboxing has to terminate rather than trust ``.item()``.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [
            _selection(0, score=np.longdouble("inf")),
            _selection(0, name="x2", score=np.longdouble("2.5")),
        ],
    )

    non_finite, finite = _strict_lines(_sidecar_path(directory, 0))

    assert non_finite["score"] is None
    assert finite["score"] == pytest.approx(2.5)


def test_timestamp_keeps_its_iso_spelling(tmp_path: Path) -> None:
    """``pd.Timestamp`` encoded before F-062 and must encode identically now."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [_selection(0, kind="param", value={"origin": pd.Timestamp("2003-01-31")})],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"]["origin"] == "2003-01-31T00:00:00"


def test_unrecognised_leaf_keeps_its_pre_fix_rendering(tmp_path: Path) -> None:
    """Sanitizing hands anything it does not know to the unchanged ``default`` hook.

    A multi-element array rendered as ``str()`` before F-062. The sanitizer must
    not start coercing such values itself -- it returns them untouched, so this is
    still ``_json_default``'s rendering and not a new one.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [_selection(0, kind="param", value={"grid": np.array([1.0, 2.0])})],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"]["grid"] == str(np.array([1.0, 2.0]))


class _CustomMapping(Mapping):
    """A ``Mapping`` that is not a ``dict``, so ``json`` never treats it as one."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"_CustomMapping({self._data!r})"


def test_custom_mapping_is_traversed_not_stringified(tmp_path: Path) -> None:
    """A rendering CHANGE, documented rather than claimed away.

    ``json`` treats only a real ``dict`` as an object, so before F-062 a non-``dict``
    ``Mapping`` fell through to ``_json_default``'s ``str()`` and landed as a repr
    string with any non-finite leaf frozen inside it. The boundary walk traverses
    any ``Mapping``, so it is now rebuilt as a JSON object and those leaves are
    neutralized. That is the point of hardening the boundary, but it means
    "already-encodable values are untouched" holds for the shipped flat records and
    the scalar, timestamp and array cases pinned above -- not universally.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [
            _selection(
                0, kind="param", value=_CustomMapping({"a": 1, "b": float("nan")})
            )
        ],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"] == {"a": 1, "b": None}


def test_non_finite_mapping_key_stays_a_quoted_string(tmp_path: Path) -> None:
    """A non-finite KEY already encoded legally, so strictness must not break it.

    ``json.dumps({float("nan"): 1})`` yields ``{"NaN": 1}`` -- a quoted string key,
    which is valid RFC 8259 -- but it routes that spelling through the same
    ``allow_nan`` switch as a non-finite VALUE. Turning the switch off without
    handling keys would start REFUSING a line that was standards-clean all along.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory,
        0,
        [_selection(0, kind="param", value={float("nan"): 1, float("inf"): 2})],
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"] == {"NaN": 1, "Infinity": 2}


def test_colliding_mapping_keys_are_refused_not_silently_merged(
    tmp_path: Path,
) -> None:
    """Rewriting a non-finite key can collide with a key that is already a string.

    ``{float("nan"): 1, "NaN": 2}`` would collapse to ``{"NaN": 2}`` under a plain
    dict comprehension, dropping a value with nothing on disk to show for it --
    and ``json`` REFUSED that mapping before F-062 anyway, since ``sort_keys``
    cannot order ``str`` against ``float``. Silently merging would be a regression
    on top of a lie, so the record is refused whole.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(directory, 0, [_selection(0, score=1.0)])
    before = _sidecar_path(directory, 0).read_bytes()

    with pytest.raises(ValueError, match="both encode to"):
        ckpt.append_origin_selection_records(
            directory,
            0,
            [_selection(0, kind="param", value={float("nan"): 1, "NaN": 2})],
        )

    assert _sidecar_path(directory, 0).read_bytes() == before
    assert _tmp_leftovers(directory) == []


def test_two_distinct_nan_keys_are_refused(tmp_path: Path) -> None:
    """``nan != nan``, so a dict really can hold two of them, and both rewrite alike."""
    directory = _cell(tmp_path, 0)

    with pytest.raises(ValueError, match="both encode to"):
        ckpt.append_origin_selection_records(
            directory,
            0,
            [_selection(0, kind="param", value={float("nan"): 1, float("nan"): 2})],
        )

    assert not _sidecar_path(directory, 0).exists()
    assert _tmp_leftovers(directory) == []


@pytest.mark.parametrize("key", [np.int64(7), np.float32("nan"), np.float32(1.5)])
def test_unsupported_numpy_key_stays_rejected(tmp_path: Path, key: Any) -> None:
    """A key ``json`` never accepted must not become acceptable via this fix.

    ``json`` raises ``TypeError`` for a numpy-typed key, and normalizing keys
    through the numeric unboxer would have quietly turned that into a written
    record -- WIDENING what the sidecar accepts while claiming only to change what
    it emits. Only non-finite ``float`` keys (``np.float64`` included, being a real
    ``float`` subclass) are rewritten; these are handed to the encoder untouched.
    """
    directory = _cell(tmp_path, 0)

    with pytest.raises(TypeError, match="keys must be"):
        ckpt.append_origin_selection_records(
            directory, 0, [_selection(0, kind="param", value={key: "a"})]
        )

    assert not _sidecar_path(directory, 0).exists()
    assert _tmp_leftovers(directory) == []


def test_numpy_float64_non_finite_key_still_encodes(tmp_path: Path) -> None:
    """``np.float64`` IS a ``float`` subclass, so ``json`` already keyed on it."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(
        directory, 0, [_selection(0, kind="param", value={np.float64("nan"): 1})]
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"] == {"NaN": 1}


def test_records_are_not_reordered_or_collapsed(tmp_path: Path) -> None:
    """Sanitizing is per-leaf; it must not dedupe or reorder selection rows."""
    directory = _cell(tmp_path, 0)
    written = [
        _selection(0, name="x1", score=float("nan")),
        _selection(0, name="x2", score=float("nan")),
        _selection(0, name="x1", score=float("nan")),
    ]
    ckpt.append_origin_selection_records(directory, 0, written)

    lines = _strict_lines(_sidecar_path(directory, 0))

    assert [line["name"] for line in lines] == ["x1", "x2", "x1"]


def test_repeated_sibling_object_is_not_mistaken_for_a_cycle(tmp_path: Path) -> None:
    """Cycle detection tracks the CURRENT PATH, not every object already seen."""
    directory = _cell(tmp_path, 0)
    shared = {"alpha": float("nan")}
    ckpt.append_origin_selection_records(
        directory, 0, [_selection(0, kind="param", value=[shared, shared])]
    )

    (line,) = _strict_lines(_sidecar_path(directory, 0))

    assert line["value"] == [{"alpha": None}, {"alpha": None}]


# --------------------------------------------------------------------------- #
# Failing: CLOSED. Nothing is coerced into writability, and nothing is stranded.
# --------------------------------------------------------------------------- #
def test_unencodable_record_raises_and_writes_no_sidecar(tmp_path: Path) -> None:
    """Mixed-type keys defeat ``sort_keys``; the write must fail, not degrade."""
    directory = _cell(tmp_path, 0)

    with pytest.raises(TypeError):
        ckpt.append_origin_selection_records(
            directory,
            0,
            [_selection(0, kind="param", value={1: "a", "b": float("inf")})],
        )

    assert not _sidecar_path(directory, 0).exists()
    assert _tmp_leftovers(directory) == []


def test_cyclic_record_raises_rather_than_nulling_the_back_reference(
    tmp_path: Path,
) -> None:
    """A cycle is unencodable in JSON, so the record is refused as a whole.

    The walker must neither spin on it nor quietly null the back-reference, which
    would report success for a record that differs from the one in memory.
    """
    directory = _cell(tmp_path, 0)
    cyclic: dict[str, Any] = {"levels": [float("nan")]}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="circular reference"):
        ckpt.append_origin_selection_records(
            directory, 0, [_selection(0, kind="param", value=cyclic)]
        )

    assert not _sidecar_path(directory, 0).exists()
    assert _tmp_leftovers(directory) == []


def test_failed_write_preserves_an_earlier_final_sidecar(tmp_path: Path) -> None:
    """A refused record must not destroy history a previous write got right."""
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(directory, 0, [_selection(0, score=1.0)])
    before = _sidecar_path(directory, 0).read_bytes()

    with pytest.raises(TypeError):
        ckpt.append_origin_selection_records(
            directory, 0, [_selection(0, kind="param", value={1: "a", "b": 2})]
        )

    assert _sidecar_path(directory, 0).read_bytes() == before
    assert _tmp_leftovers(directory) == []


class _ExplodingRecord(Mapping):
    """A record whose iteration raises a NON-``OSError``, mid-write."""

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("selection record exploded mid-write")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("selection record exploded mid-write")


def test_non_oserror_failure_cleans_temp_and_preserves_final(tmp_path: Path) -> None:
    """Cleanup was ``except OSError``, so any other failure leaked a dotfile.

    The temp file lives in the checkpoint directory itself, and nothing ever
    removes a leaked one. Its dot-prefixed name matches neither the
    ``origin_*.parquet`` nor the ``origin_*_selection.jsonl`` scan, so it does not
    corrupt resume accounting -- it just accumulates permanently in a directory
    presented to the user as ours.
    """
    directory = _cell(tmp_path, 0)
    ckpt.append_origin_selection_records(directory, 0, [_selection(0, score=1.0)])
    before = _sidecar_path(directory, 0).read_bytes()

    with pytest.raises(RuntimeError, match="exploded mid-write"):
        ckpt.append_origin_selection_records(directory, 0, [_ExplodingRecord()])

    assert _tmp_leftovers(directory) == []
    # The previous good sidecar is untouched: a failed write replaces nothing.
    assert _sidecar_path(directory, 0).read_bytes() == before


def test_strict_write_is_silent_on_ordinary_records(tmp_path: Path) -> None:
    """The writer either succeeds or raises. It never warns and continues."""
    directory = _cell(tmp_path, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ckpt.append_origin_selection_records(
            directory,
            0,
            [_selection(0, score=float("nan"), value={"alpha": np.float32("inf")})],
        )

    assert _strict_lines(_sidecar_path(directory, 0))


# --------------------------------------------------------------------------- #
# The runner order the failure policy rests on
# --------------------------------------------------------------------------- #
def _run(checkpoint_path: Path):
    """The cheap ridge cell the F-058/F-060 suites use, with sidecars enabled."""
    n = 48
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    panel = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )
    return mf.forecasting.run(
        panel,
        "ridge",
        window=mf.window.spec(
            estimation=mf.window.estimation_expanding(min_size=24),
            val=mf.window.val_last_block(size=8),
            test=mf.window.test_origins(horizon=1, step=6),
        ),
        features=mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2]),
        params={"alpha": 0.01},
        save_models=False,
        checkpoint_path=checkpoint_path,
        selection_history=True,
    )


def test_sidecar_failure_leaves_the_origin_incomplete_and_recomputable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-closed argument, end to end.

    A refused record aborts the origin before its parquet exists, nothing is marked
    completed, and the identical re-run computes it normally.
    """
    cell = tmp_path / "cell"

    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("strict sidecar encoding refused this record")

    monkeypatch.setattr(runner_mod, "append_origin_selection_records", _refuse)
    with pytest.raises(ValueError, match="strict sidecar encoding refused"):
        _run(cell)

    hdir = cell / "h1"
    # Non-vacuous: the run got as far as claiming the directory.
    assert ckpt.checkpoint_identity_path(hdir).exists()
    assert ckpt.final_origin_files(hdir) == []
    assert ckpt.completed_origin_positions(hdir) == set()

    monkeypatch.undo()
    frame = _run(cell).to_frame()

    assert not frame.empty
    assert ckpt.completed_origin_positions(hdir)
    assert _strict_lines(sorted(hdir.glob("origin_*_selection.jsonl"))[0])


_WRITE_CALLS = ("append_origin_selection_records", "append_origin_records")
#: Every runner function that writes a sidecar today. Discovery below compares
#: against this, so a THIRD call site fails until someone reviews its ordering.
_LIVE_SITES = ("run", "_run_vintage_aware")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_TRIES = (ast.Try, getattr(ast, "TryStar", ast.Try))
_WITHS = (ast.With, ast.AsyncWith)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every node under ``node``, not descending into a nested function."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, (*_FUNCTIONS, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(child))


def _write_calls(node: ast.AST) -> list[str]:
    """The checkpoint-write calls ``node`` makes itself, in source order."""
    found = sorted(
        (call.lineno, call.col_offset, call.func.id)
        for call in _own_nodes(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in _WRITE_CALLS
    )
    return [name for *_, name in found]


def _swallows_exceptions(node: ast.AST) -> bool:
    """Does ``node.body`` run under something that can discard an exception?"""
    if isinstance(node, _TRIES):
        return bool(node.handlers)
    if isinstance(node, _WITHS):
        return any(_is_suppress(item.context_expr) for item in node.items)
    return False


def _is_suppress(expr: ast.AST) -> bool:
    """``contextlib.suppress(...)`` or a ``suppress(...)`` imported by name.

    Deliberately OVER-BROAD: any attribute call ending in ``.suppress`` counts, so
    an unrelated ``foo.suppress(...)`` would also trip this. That direction is the
    safe one -- a false positive is a loud test failure someone reads, whereas
    resolving the real binding would mean import tracking this test does not want.
    """
    if not isinstance(expr, ast.Call):
        return False
    return (isinstance(expr.func, ast.Name) and expr.func.id == "suppress") or (
        isinstance(expr.func, ast.Attribute) and expr.func.attr == "suppress"
    )


def _assert_sidecar_is_written_first(tree: ast.Module, expected: Sequence[str]) -> None:
    """The whole structural contract, checkable against any parsed module.

    Sites are DISCOVERED (every function that calls the sidecar writer itself)
    rather than named, then compared against ``expected``, so a new call site or a
    deleted one fails here instead of silently going unchecked.

    They are held as a LIST, not keyed by name: two functions can share a name (a
    conditional definition, a method and a free function), and a dict would silently
    keep one of them, checking half of what it claims to. Labels carry the line
    number so a failure names the right one.
    """
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, _FUNCTIONS)
        and "append_origin_selection_records" in _write_calls(node)
    ]
    labels = sorted(f"{node.name}:{node.lineno}" for node in sites)
    assert sorted(node.name for node in sites) == sorted(expected), f"live sites: {labels}"

    for function in sites:
        label = f"{function.name}:{function.lineno}"
        assert _write_calls(function) == list(_WRITE_CALLS), label
        for node in _own_nodes(function):
            if _swallows_exceptions(node):
                caught = [call for stmt in node.body for call in _write_calls(stmt)]
                assert caught == [], f"{label}: {caught} under {type(node).__name__}"


def test_every_live_call_site_writes_the_sidecar_before_the_parquet() -> None:
    """Pins the ordering the fail-closed policy rests on, structurally.

    ``run`` is covered behaviourally above; ``_run_vintage_aware`` needs a vintage
    panel to reach, so it would otherwise go unpinned. Working from the parsed tree
    means reformatting, or a comment naming either function, cannot move this.

    Three properties matter. If the parquet were written FIRST, a refused sidecar
    would leave a COMPLETED origin whose history can never be rewritten, because a
    completed origin is skipped on every later run. If either call sat under a
    handler or a ``contextlib.suppress``, a swallowed failure would reach the same
    end. And if a third call site appeared, neither check would have looked at it.
    """
    tree = ast.parse(Path(runner_mod.__file__).read_text(encoding="utf-8"))

    _assert_sidecar_is_written_first(tree, _LIVE_SITES)


_HEALTHY = """
    def run():
        append_origin_selection_records(d, p, recs)
        append_origin_records(d, p, rows)
"""

_MUTANTS = {
    "parquet written first": """
        def run():
            append_origin_records(d, p, rows)
            append_origin_selection_records(d, p, recs)
    """,
    "sidecar under except": """
        def run():
            try:
                append_origin_selection_records(d, p, recs)
            except Exception:
                pass
            append_origin_records(d, p, rows)
    """,
    "sidecar under except*": """
        def run():
            try:
                append_origin_selection_records(d, p, recs)
            except* ValueError:
                pass
            append_origin_records(d, p, rows)
    """,
    "sidecar under contextlib.suppress": """
        def run():
            with contextlib.suppress(Exception):
                append_origin_selection_records(d, p, recs)
            append_origin_records(d, p, rows)
    """,
    "sidecar under bare suppress": """
        def run():
            with suppress(Exception):
                append_origin_selection_records(d, p, recs)
            append_origin_records(d, p, rows)
    """,
    "sidecar call deleted": """
        def run():
            append_origin_records(d, p, rows)
    """,
    "third call site added": """
        def run():
            append_origin_selection_records(d, p, recs)
            append_origin_records(d, p, rows)

        def run_extra():
            append_origin_selection_records(d, p, recs)
            append_origin_records(d, p, rows)
    """,
}


def test_the_call_order_check_accepts_a_healthy_module() -> None:
    """Positive control: the mutant test below would pass vacuously without it."""
    _assert_sidecar_is_written_first(ast.parse(textwrap.dedent(_HEALTHY)), ["run"])


#: ``except*`` is 3.11+ SYNTAX, so on 3.10 ``ast.parse`` raises ``SyntaxError``
#: before the check under test is ever reached -- an ERROR, not a rejection, and
#: this package supports 3.10 (``requires-python = ">=3.10"``, and ``ci-core``
#: runs the 3.10 matrix). The mutant lives in a string, so the file still IMPORTS
#: on 3.10; only parsing it must be skipped there. Coverage is kept on 3.11+.
_TRYSTAR_ONLY = frozenset({"sidecar under except*"})
_NEEDS_TRYSTAR = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="`except*` is Python 3.11+ syntax; ast.parse() cannot read this mutant",
)
_MUTANT_PARAMS = [
    pytest.param(name, marks=[_NEEDS_TRYSTAR] if name in _TRYSTAR_ONLY else [])
    for name in sorted(_MUTANTS)
]


@pytest.mark.parametrize("mutant", _MUTANT_PARAMS)
def test_the_call_order_check_rejects_each_mutant(mutant: str) -> None:
    """Every way the contract can break must actually fail the check above."""
    tree = ast.parse(textwrap.dedent(_MUTANTS[mutant]))

    with pytest.raises(AssertionError):
        _assert_sidecar_is_written_first(tree, ["run"])


# --------------------------------------------------------------------------- #
# Reading: legacy directories keep working, which is the whole compatibility cost
# --------------------------------------------------------------------------- #
def test_legacy_bare_nan_sidecar_loads_as_missing(tmp_path: Path) -> None:
    """Sidecars already on disk hold bare ``NaN``; refusing them would be a wipe."""
    directory = _cell(tmp_path, 0)
    _legacy_sidecar(directory, 0)

    frame = ckpt.load_selection_history_frame(directory)

    assert len(frame) == 1
    assert frame.loc[0, "name"] == "x1"
    assert pd.isna(frame.loc[0, "score"])
    assert not isinstance(frame.loc[0, "score"], str)


def test_legacy_and_strict_sidecars_coexist(tmp_path: Path) -> None:
    """A directory half-written before the fix and half after must load whole."""
    directory = _cell(tmp_path, 0, 1)
    _legacy_sidecar(directory, 0)
    ckpt.append_origin_selection_records(
        directory, 1, [_selection(1, name="x2", score=float("nan"))]
    )

    frame = ckpt.load_selection_history_frame(directory)

    assert sorted(frame["origin_pos"]) == [0, 1]
    assert frame["score"].isna().all()
    # Only the post-fix sidecar is strict; the legacy one is deliberately not.
    _strict_lines(_sidecar_path(directory, 1))


def test_legacy_sidecar_still_fails_closed_on_real_corruption(tmp_path: Path) -> None:
    """Tolerating bare constants must not re-open the F-061 partial-read hole."""
    directory = _cell(tmp_path, 0)
    path = _sidecar_path(directory, 0)
    path.write_text(
        json.dumps(_selection(0, score=float("nan")), sort_keys=True)
        + "\n{ truncated mid-write",
        encoding="utf-8",
    )

    with pytest.raises(ckpt.CheckpointCorruptionError) as excinfo:
        ckpt.load_selection_history_frame(directory)

    assert excinfo.value.line == 2
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# --------------------------------------------------------------------------- #
# Identity: this is a writer fix, so nothing about resume may move
# --------------------------------------------------------------------------- #
def test_legacy_sidecar_does_not_prevent_resume(tmp_path: Path) -> None:
    """A legacy sidecar is no reason to refuse a directory that is otherwise ours.

    ``CHECKPOINT_IDENTITY_VERSION`` itself is pinned by
    ``test_checkpoint_resume_identity.test_identity_version_is_two``, so restating
    it here would only duplicate that guard. What is new to F-062 is that a
    directory holding a PRE-fix sidecar still resumes, which is the compatibility
    cost of keeping the reader lenient.
    """
    directory = _cell(tmp_path, 0)
    identity = _identity()
    ckpt.write_checkpoint_identity(directory, identity)
    _legacy_sidecar(directory, 0)

    assert ckpt.resolve_checkpoint_resume(directory, identity) == {0}
