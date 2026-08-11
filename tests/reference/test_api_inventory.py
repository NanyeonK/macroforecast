"""The inventory's ownership claims must come from declarations, not from dir().

Every test here fails against the previous ``dir()``-scanning revision of
``tools/api_inventory.py``: that module wrote its output at import time, keyed
its rows off ``dir(mf)`` rather than ``mf.__all__``, and resolved an owner by
taking whichever hard-coded submodule happened to expose the name first.
"""
from __future__ import annotations

import importlib
import typing
from pathlib import Path

import macroforecast as mf

import tools.api_inventory as api_inventory

COMMITTED = Path("docs/api_inventory.json")

# The four Literal aliases and their declared home submodule.
LITERAL_ALIASES = {
    "MetadataLevel": "meta",
    "StageDefaultScope": "meta",
    "RegimeDirection": "data",
    "SamePeriodPolicy": "data",
}


def _surfaces() -> dict[str, list[dict[str, object]]]:
    return api_inventory.build_inventory()["surfaces"]


def _names(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["name"]) for row in rows}


def _public_dir() -> set[str]:
    return {name for name in dir(mf) if not name.startswith("_")}


def test_importing_the_tool_writes_nothing() -> None:
    """Importing the module must be a pure import, not a build step."""

    before_bytes = COMMITTED.read_bytes()
    before_mtime = COMMITTED.stat().st_mtime_ns

    importlib.reload(api_inventory)

    assert COMMITTED.read_bytes() == before_bytes
    assert COMMITTED.stat().st_mtime_ns == before_mtime


def test_supported_surface_is_exactly_top_level_all() -> None:
    inventory = api_inventory.build_inventory()
    supported = inventory["surfaces"]["supported"]

    assert _names(supported) == set(mf.__all__)
    assert inventory["counts"]["supported"] == len(mf.__all__)


def test_surfaces_are_disjoint_and_account_for_public_dir() -> None:
    surfaces = _surfaces()
    supported = _names(surfaces["supported"])
    special = _names(surfaces["special"])
    non_api = _names(surfaces["non_api_globals"])

    assert supported & special == set()
    assert supported & non_api == set()
    assert special & non_api == set()

    assert supported | non_api == _public_dir()

    assert "__version__" in special
    assert "__version__" not in supported
    assert "__version__" not in non_api


def test_non_api_globals_surface_is_empty() -> None:
    """Nothing is bound at root scope outside ``__all__`` any more.

    Before this cleanup the surface held ``Any``, ``annotations`` and
    ``import_module``. ``__version__`` is unaffected: it is public on purpose,
    underscore-prefixed, and carried by the separate ``special`` surface.
    """

    inventory = api_inventory.build_inventory()

    assert inventory["surfaces"]["non_api_globals"] == []
    assert inventory["counts"]["non_api_globals"] == 0
    assert inventory["counts"]["public_dir"] == inventory["counts"]["supported"]
    assert _public_dir() == set(mf.__all__)

    for name in ("Any", "annotations", "import_module"):
        assert not hasattr(mf, name), name


def test_non_api_globals_are_root_globals_not_submodule_owned() -> None:
    """The empty surface above is a measurement, not a query that stopped working.

    Binding a name at root module scope is exactly what an unaliased
    ``from typing import Any`` does, so injecting one reproduces the condition
    the surface exists to catch -- and proves the preceding test is not vacuous.
    """

    assert _surfaces()["non_api_globals"] == []

    setattr(mf, "leaked_probe", object())
    try:
        rows = _surfaces()["non_api_globals"]
        names = _names(rows)
        public_while_leaked = _public_dir()
    finally:
        delattr(mf, "leaked_probe")

    assert names == {"leaked_probe"}
    assert names == public_while_leaked - set(mf.__all__)

    (row,) = rows
    assert row["surface"] == "non_api_global"
    assert row["bound_in_root_globals"] is True
    # A root global is owned by the root module, so its canonical path can
    # never carry a submodule segment.
    assert row["canonical"] == "mf.leaked_probe"

    # The probe is a plain assignment, not an import -- so a row must not state
    # a provenance the query never established. The three names this surface
    # used to hold happened to be imports; that is history, not a rule, and it
    # belongs in the module docstring rather than in generated per-row data.
    reason = str(row["reason"])
    assert "import statement" not in reason, reason
    assert "lazy loader" not in reason, reason
    assert "module scope" in reason

    assert _surfaces()["non_api_globals"] == []


def test_canonical_owner_matches_root_lazy_exports() -> None:
    for row in _surfaces()["supported"]:
        name = str(row["name"])
        owner = row["owner"]
        if owner is None:
            assert name in mf._LAZY_MODULES
            assert row["canonical"] == f"mf.{name}"
        else:
            assert f".{owner}" == mf._LAZY_EXPORTS[name]
            assert row["canonical"] == f"mf.{owner}.{name}"


def test_axis_contribution_is_owned_by_analysis() -> None:
    row = next(
        row for row in _surfaces()["supported"] if row["name"] == "axis_contribution"
    )

    assert row["owner"] == "analysis"
    assert row["canonical"] == "mf.analysis.axis_contribution"


def test_every_lazy_export_is_declared_in_its_home_all() -> None:
    inventory = api_inventory.build_inventory()

    for row in inventory["surfaces"]["supported"]:
        if row["owner"] is not None:
            assert row["in_owner_all"] is True, row["name"]

    assert inventory["corroboration"]["owner_all_exceptions"] == []


def test_literal_aliases_are_declared_submodule_api() -> None:
    inventory = api_inventory.build_inventory()
    rows = {str(row["name"]): row for row in inventory["surfaces"]["supported"]}
    exceptions = inventory["corroboration"]["module_attr_exceptions"]

    for name, owner in LITERAL_ALIASES.items():
        row = rows[name]

        assert row["owner"] == owner
        assert row["in_owner_all"] is True
        # A Literal alias reports typing as its __module__, so the disagreement
        # is recorded as a documented exception rather than an error.
        assert name in exceptions


def test_module_attr_exceptions_are_recorded_not_dropped() -> None:
    corroboration = api_inventory.build_inventory()["corroboration"]

    assert set(corroboration["module_attr_exceptions"]) == {
        "MetadataLevel",
        "RegimeDirection",
        "SamePeriodPolicy",
        "StageDefaultScope",
        "Split",
    }
    assert set(corroboration["module_attr_absent"]) == {
        "DEFAULT_RANDOM_SEED",
        "MODEL_ENSEMBLE_BASE_ESTIMATORS",
        "MODEL_ENSEMBLE_SPECS",
    }


def test_counts_identities_all_hold() -> None:
    inventory = api_inventory.build_inventory()
    counts = inventory["counts"]
    surfaces = inventory["surfaces"]

    assert all(inventory["identities"].values()), inventory["identities"]

    # Re-derive each identity from the rows rather than trusting the flags.
    supported = surfaces["supported"]
    non_api = surfaces["non_api_globals"]
    module_rows = [row for row in supported if row["owner"] is None]
    symbol_rows = [row for row in supported if row["owner"] is not None]

    assert len(supported) == len(symbol_rows) + len(module_rows)
    assert len(symbol_rows) == len(mf._LAZY_EXPORTS)
    assert len(module_rows) == len(mf._LAZY_MODULES)
    assert len(_public_dir()) == len(supported) + len(non_api)

    export_kinds: dict[str, int] = {}
    for row in supported:
        kind = str(row["export_kind"])
        export_kinds[kind] = export_kinds.get(kind, 0) + 1

    assert export_kinds == counts["export_kinds"]
    assert export_kinds["lazy_export"] == len(mf._LAZY_EXPORTS)
    assert export_kinds["namespace"] == len(mf._LAZY_MODULES)
    assert sum(export_kinds.values()) == len(supported)
    assert counts["documented"] == sum(1 for row in supported if row["in_docs"])


def test_export_kind_is_factual() -> None:
    """export_kind restates a declaration; it never encodes a judgement."""

    for row in _surfaces()["supported"]:
        name = str(row["name"])
        kind = row["export_kind"]

        assert kind in {"lazy_export", "namespace"}

        if kind == "namespace":
            assert row["owner"] is None
            assert name in mf._LAZY_MODULES
            assert name not in mf._LAZY_EXPORTS
        else:
            assert row["owner"] is not None
            assert name in mf._LAZY_EXPORTS
            assert name not in mf._LAZY_MODULES


def test_inventory_states_no_stability_tier() -> None:
    """A tier is policy. The generator must not author one.

    An earlier revision hard-coded ``meta`` as experimental, so seven ``meta``
    aliases were labelled EXPERIMENTAL while docs/api_tiers.md named
    ``interpretation`` and ``reporting`` -- which had no rows at all.
    """

    inventory = api_inventory.build_inventory()

    for rows in inventory["surfaces"].values():
        for row in rows:
            assert "tier" not in row, row["name"]

    assert "tiers" not in inventory["counts"]
    assert not hasattr(api_inventory, "EXPERIMENTAL_OWNERS")
    assert not hasattr(api_inventory, "_tier")


def test_kind_never_reports_a_private_implementation_type() -> None:
    """A private type name is an implementation detail, not a fact to record.

    ``typing.Any`` is a ``_SpecialForm`` instance on 3.10 and a class with
    metaclass ``_AnyMeta`` on 3.11+, so recording the raw type name made the
    committed inventory unmatchable on 3.10 while 3.11 and 3.12 agreed.
    """

    offenders = [
        (row["surface"], row["name"], row["kind"])
        for rows in api_inventory.build_inventory()["surfaces"].values()
        for row in rows
        if str(row["kind"]).startswith("_")
    ]

    assert offenders == []


def test_kind_normalizes_private_types_generally() -> None:
    """The rule is general, not a special case for ``typing.Any``."""

    class _Hidden:
        pass

    assert not api_inventory._kind(_Hidden).startswith("_")
    assert not api_inventory._kind(_Hidden()).startswith("_")

    # Both are typing constructs whose private class names differ by
    # interpreter, so they must collapse to the same recorded kind.
    assert api_inventory._kind(typing.Any) == api_inventory._kind(
        typing.Literal["a", "b"]
    )


def test_kind_preserves_public_type_names() -> None:
    """Normalizing must not delete facts that are already stable."""

    # docs/api_tiers.md quotes this kind for mf.Split.
    assert api_inventory._kind(list[tuple[int, int]]) == "GenericAlias"
    assert api_inventory._kind(lambda: None) == "function"
    assert api_inventory._kind(1) == "int"


def test_version_surface_records_documentation_evidence() -> None:
    row = next(row for row in _surfaces()["special"] if row["name"] == "__version__")

    assert row["in_docs"] is True


def test_schema_is_declared() -> None:
    inventory = api_inventory.build_inventory()

    assert inventory["schema"] == "macroforecast.api_inventory"
    assert inventory["schema_version"] == 3
    assert inventory["package_version"] == mf.__version__


def test_render_is_deterministic() -> None:
    assert api_inventory.render() == api_inventory.render()


def test_check_passes_against_committed_inventory() -> None:
    assert api_inventory.check(COMMITTED) == (True, [])


def test_check_detects_drift_and_does_not_rewrite(tmp_path: Path) -> None:
    drifted = tmp_path / "api_inventory.json"
    drifted.write_text(COMMITTED.read_text(encoding="utf-8") + " ", encoding="utf-8")
    before = drifted.read_bytes()

    ok, messages = api_inventory.check(drifted)

    assert ok is False
    assert len(messages) == 1
    assert messages[0].startswith("changed: ")
    # A check must never repair what it is checking.
    assert drifted.read_bytes() == before


def test_check_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    ok, messages = api_inventory.check(missing)

    assert ok is False
    assert messages == [f"missing: {missing.as_posix()}"]


def test_cli_check_exits_zero_and_nonzero(tmp_path: Path) -> None:
    assert api_inventory.main([str(COMMITTED), "--check"]) == 0

    drifted = tmp_path / "api_inventory.json"
    drifted.write_text(COMMITTED.read_text(encoding="utf-8") + " ", encoding="utf-8")
    before = drifted.read_bytes()

    assert api_inventory.main([str(drifted), "--check"]) == 1
    assert drifted.read_bytes() == before
