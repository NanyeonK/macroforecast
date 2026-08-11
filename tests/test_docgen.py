from __future__ import annotations

import pathlib
import subprocess
import sys
from pathlib import Path

import pytest

import tools.docgen as docgen
from tools.docgen import renderer


def test_tools_docgen_imports_cleanly() -> None:
    assert "write_all" in docgen.__all__
    assert docgen.collect_pages()


def test_docgen_uses_source_declared_module_exports(monkeypatch) -> None:
    import macroforecast.model_selection as model_selection

    page = Path("model_selection.md")
    baseline = docgen.collect_pages()[page]
    monkeypatch.setattr(
        model_selection,
        "__all__",
        [*model_selection.__all__, "OptionalRuntimeOnly"],
    )
    monkeypatch.setattr(
        model_selection,
        "OptionalRuntimeOnly",
        lambda: None,
        raising=False,
    )

    assert docgen.collect_pages()[page] == baseline


def test_docgen_module_pages_have_source_declared_exports() -> None:
    missing = [
        page.module
        for page in renderer.MODULE_PAGES
        if renderer._source_declared_all(renderer._module(page)) is None
    ]

    assert not missing




# No version skip. The renderer no longer takes classification, signatures, or
# summaries from the running interpreter, so byte identity holds on every supported
# CPython -- 3.10 included, where the old skip's stated reasons (pandas path reprs and
# alias classification) were exactly the drift this lane fixed. Removing the skip makes
# this test enforce the contract that was demonstrated across 3.10-3.14 rather than
# assume it above 3.11.
def test_docgen_check_passes_on_committed_reference_tree() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.docgen", "--check", "docs/reference"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_docgen_check_fails_on_perturbed_page(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    docgen.write_all(reference)
    page = reference / "index.md"
    page.write_text(page.read_text(encoding="utf-8") + "\nperturbation\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "tools.docgen", "--check", str(reference)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "changed: index.md" in result.stderr


# --------------------------------------------------------------------------- #
# D-001: nothing rendered may come from the interpreter instead of the source
# --------------------------------------------------------------------------- #

def test_public_type_alias_is_classified_as_a_type_alias() -> None:
    """``Split = tuple[np.ndarray, np.ndarray]`` is not something to construct.

    The interpreter disagrees with itself about that: ``Split`` answers
    ``inspect.isclass`` on 3.10, ``callable`` on 3.12, and reports a different
    ``inspect.signature`` again on 3.14 — so the generated page moved without the source
    moving. Deciding it structurally removes the interpreter from the answer.
    """
    from macroforecast import window

    from tools.docgen.renderer import _object_kind

    assert _object_kind(window.Split) == "type alias"


def test_union_type_alias_is_classified_as_a_type_alias() -> None:
    """The other shape of the same problem: ``NJobs`` moved between callable and data."""
    from macroforecast import meta

    from tools.docgen.renderer import _object_kind

    assert _object_kind(meta.NJobs) == "type alias"


def test_type_alias_summary_names_the_target_not_the_interpreter() -> None:
    """A union alias's ``__doc__`` is prose about the interpreter's alias machinery —
    "Represent a PEP 604 union type" on 3.12, "Represent a union type" on 3.14."""
    from macroforecast import window

    from tools.docgen.renderer import _summary

    summary = _summary(window.Split)

    assert summary.startswith("Type alias for ")
    assert "PEP 604" not in summary
    assert "Represent a" not in summary


def test_union_alias_text_uses_one_canonical_spelling() -> None:
    """``typing.Union[int, X]`` reprs as itself on 3.12 and as ``int | X`` on 3.14.

    Rebuilding from ``get_origin``/``get_args`` picks one spelling here instead of
    inheriting whichever the running interpreter prefers.
    """
    import typing

    from tools.docgen.renderer import _alias_text

    assert _alias_text(typing.Union[int, str]) == "int | str"
    assert _alias_text(int | None) == "int | None"


def test_alias_text_exact_renderings() -> None:
    """The four shapes the reference actually contains, spelled out.

    Each is a place the interpreter's own repr would have leaked something: ``NoneType``
    for the optional, an unquoted ``auto`` for the literal, ``<class 'numpy.ndarray'>``
    for the class leaves, and ``Ellipsis`` for the callable's parameter list.
    """
    import typing

    import numpy as np

    from macroforecast import meta, metrics, window
    from tools.docgen.renderer import _alias_text

    assert _alias_text(int | None) == "int | None"
    assert _alias_text(typing.Literal["auto"]) == "Literal['auto']"
    assert _alias_text(meta.NJobs) == "int | Literal['auto']"
    assert _alias_text(tuple[np.ndarray, np.ndarray]) == "tuple[numpy.ndarray, numpy.ndarray]"
    assert _alias_text(window.Split) == "tuple[numpy.ndarray, numpy.ndarray]"
    assert _alias_text(metrics.MetricLike) == "str | Callable[..., float]"


def test_generated_alias_assignments_have_no_repr_artifacts() -> None:
    """Scan the rendered pages, not just the helper.

    A helper that returns the right string is worth little if some other path writes the
    raw repr. These three artifacts are what the interpreter emits when it is asked
    directly, so their absence from every alias line is the evidence that nothing takes
    that shortcut.
    """
    import tools.docgen as docgen

    artifacts = ("<class ", "NoneType", "Ellipsis")
    offenders: list[tuple[str, str, str]] = []
    for page, text in docgen.collect_pages().items():
        for line in text.splitlines():
            if "type alias" not in line and " = " not in line:
                continue
            for artifact in artifacts:
                if artifact in line:
                    offenders.append((page.name, artifact, line.strip()[:90]))

    assert not offenders, offenders


def test_frozenset_value_summary_comes_from_the_value() -> None:
    """CPython rewrote ``frozenset.__doc__`` in 3.13, so a public frozenset constant's
    summary changed from "frozenset() -> empty frozenset object" to "Build an immutable
    unordered collection of unique elements." without the constant changing."""
    from macroforecast.metrics import DENSITY_METRIC_NAMES

    from tools.docgen.renderer import _summary

    summary = _summary(DENSITY_METRIC_NAMES)

    assert summary.startswith("A `frozenset` of ")
    assert str(len(DENSITY_METRIC_NAMES)) in summary
    assert "empty frozenset object" not in summary
    assert "Build an immutable" not in summary


@pytest.mark.parametrize(
    "value, expected",
    [
        (frozenset(), "An empty `frozenset`."),
        (set(), "An empty `set`."),
        (frozenset({"a", "b"}), "A `frozenset` of 2 values of str."),
        (frozenset({1}), "A `frozenset` of 1 value of int."),
    ],
)
def test_builtin_value_summaries_are_stable_and_truthful(value, expected) -> None:
    from tools.docgen.renderer import _summary

    assert _summary(value) == expected


def test_verification_counts_use_the_same_classifier_as_the_pages() -> None:
    """The count page kept a second copy of the classification rule, so it still drifted
    after every page it counts had stopped: on 3.10 ``Split`` counted as a class and on
    3.14 ``NJobs`` counted as data."""
    from macroforecast import meta, window

    from tools.docgen.renderer import _object_kind

    for alias in (window.Split, meta.NJobs):
        kind = _object_kind(alias)
        assert kind not in ("class", "callable", "function"), (
            "an alias must not be counted in a callable/class bucket"
        )


# --------------------------------------------------------------------------- #
# D-002 / D-003: the retired tools refuse before doing anything
# --------------------------------------------------------------------------- #

_RETIRED = (
    "generate_fred_dataset_docs.py",
    "gen_standalone_docs.py",
    "gen_encyclopedia_docs.py",
    "audit_docs_vs_code.py",
)

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


@pytest.mark.parametrize("script", _RETIRED)
def test_retired_tool_exits_non_zero_and_writes_nothing(script, tmp_path) -> None:
    """Old command lines must fail loudly rather than half-work.

    Run from a sentinel directory with the arguments the removed implementations used to
    take, so the refusal is proven against a real invocation and not just an import.
    """
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep")
    before = {path.name: path.read_bytes() for path in sentinel.rglob("*") if path.is_file()}

    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS / script),
            "--layer", "L3",
            "--dry-run",
            "--root", "docs",
            "--out", "generated.md",
        ],
        cwd=sentinel,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, f"{script} must not report success"
    after = {path.name: path.read_bytes() for path in sentinel.rglob("*") if path.is_file()}
    assert after == before, f"{script} touched the sentinel tree"


@pytest.mark.parametrize("script", _RETIRED)
def test_retired_tool_names_a_supported_replacement(script, tmp_path) -> None:
    """A refusal that does not say what to run instead is a dead end."""
    result = subprocess.run(
        [sys.executable, str(_TOOLS / script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    message = result.stdout + result.stderr
    assert "retired" in message.lower()
    assert "python -m tools.docgen" in message


def test_retired_fred_generator_states_the_harm_it_would_actually_cause() -> None:
    """The old body wrote to ``docs/fred_dataset/``, not to ``docs/datasets/``.

    So it could not have overwritten the current guides, and a refusal claiming it would
    is wrong about its own subject. What it would do is fetch or reuse external data,
    write date-dependent artifacts under ``build/fred_dataset_sources/``, and leave a
    stale competing tree beside the real guides -- the more confusing outcome, because
    both would then exist with nothing marking which is current.
    """
    result = subprocess.run(
        [sys.executable, str(_TOOLS / "generate_fred_dataset_docs.py")],
        capture_output=True,
        text=True,
    )

    message = result.stdout + result.stderr
    assert "build/fred_dataset_sources/" in message
    assert "docs/fred_dataset/" in message
    assert "external data" in message
    assert "overwrite" not in message, "the old tool cannot overwrite the current guides"


def test_retired_fred_generator_separates_guides_from_the_api_reference() -> None:
    """The dataset guides are hand-maintained prose; ``tools.docgen`` owns only the API
    reference. Conflating the two is how a caller would reach for the wrong command."""
    result = subprocess.run(
        [sys.executable, str(_TOOLS / "generate_fred_dataset_docs.py")],
        capture_output=True,
        text=True,
    )

    message = result.stdout + result.stderr
    assert "hand-maintained" in message
    assert "docs/datasets/" in message
    assert "docs/reference/" in message


@pytest.mark.parametrize("script", _RETIRED)
def test_retired_tool_imports_safely(script) -> None:
    """Importing must not reach the removed package surfaces the old bodies used."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy, sys; sys.path.insert(0, {str(_TOOLS)!r}); "
            f"__import__({script.removesuffix('.py')!r})",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", _RETIRED)
def test_retired_tool_is_importable_as_a_package_module(script) -> None:
    """``import tools.gen_standalone_docs`` from the repo root must work.

    A plain ``from _retired_doc_tool import retire`` resolves when the script is run by
    path -- Python puts ``tools/`` on ``sys.path`` -- and fails when the same file is
    imported as part of the ``tools`` package, where only the repo root is on the path.
    Both contexts are supported, so both are tested.
    """
    module = "tools." + script.removesuffix(".py")
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", _RETIRED)
def test_retired_tool_message_claims_only_what_is_true(script, tmp_path) -> None:
    """The refusal reads its own arguments, so it cannot claim nothing was read.

    What it can truthfully say is the part that matters to a caller: nothing was written
    and nothing was fetched.
    """
    result = subprocess.run(
        [sys.executable, str(_TOOLS / script), "--legacy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    message = result.stdout + result.stderr
    assert "No documentation or data file was written" in message
    assert "no network fetch" in message
    assert "Nothing was read" not in message


def _verification_counts() -> dict[str, int]:
    """The displayed rows of the reference-verification counts table."""
    import re

    import tools.docgen as docgen

    page = docgen.collect_pages()[pathlib.Path("reference_verification.md")]
    counts: dict[str, int] = {}
    for line in page.splitlines():
        match = re.fullmatch(r"\| ([^|]+?) \| (\d+) \|", line.strip())
        if match:
            counts[match.group(1).strip()] = int(match.group(2))
    return counts


def test_verification_categories_partition_the_public_symbols() -> None:
    """The category rows must add up to the total they claim to break down.

    They did not: aliases fell through to the ``else`` branch and were counted as
    "Data/module symbols" while their own pages said ``type alias`` — the table
    contradicting the pages it summarises. A sum check is the invariant that catches any
    future category added to ``_object_kind`` without a row here.
    """
    counts = _verification_counts()

    total = counts["Public symbols across module pages"]
    categories = (
        counts["Callable/function symbols"]
        + counts["Class symbols"]
        + counts["Type alias symbols"]
        + counts["Data/module symbols"]
    )

    assert categories == total, counts


def test_type_aliases_are_counted_as_aliases_and_not_as_data() -> None:
    """The count has to move with the classification, not lag behind it."""
    from macroforecast import meta, window
    from tools.docgen.renderer import _object_kind

    counts = _verification_counts()

    known_aliases = [window.Split, meta.NJobs]
    assert all(_object_kind(alias) == "type alias" for alias in known_aliases)
    assert counts["Type alias symbols"] >= len(known_aliases)
    assert counts["Data/module symbols"] >= 0
