"""v1.0 release-gate test: every operational (layer, sub-layer, axis,
option) tuple must have a Tier-1-complete :class:`OptionDoc` entry.

While layer-by-layer content PRs are still in flight, the test reports
gaps per layer rather than failing globally so progress is visible. A
layer is "v1.0-ready" once its row in ``_V1_REQUIRED_LAYERS`` flips
from ``False`` to ``True``; flipping all rows to ``True`` is the v1.0
release criterion.
"""
from __future__ import annotations

import pytest

from macrocast.scaffold import introspect
from macrocast.scaffold.option_docs import OPTION_DOCS


# v1.0 release flips every value to True. Layer-by-layer content PRs
# update this map as their docs land.
_V1_REQUIRED_LAYERS: dict[str, bool] = {
    "l0": True,   # PR-A1: 6 entries
    "l1": True,    # PR-A2 + extension: 107 entries
    "l1_5": True,  # PR-A3: 50 entries (diagnostics module)
    "l2": True,    # PR-A3: 48 entries (5-stage pipeline + scopes)
    "l2_5": True,  # PR-A3: 42 entries (diagnostics module)
    "l3": True,    # PR-A3: 37 ops
    "l3_5": True,  # PR-A3: 49 entries (diagnostics module)
    "l4": True,    # PR-A4: 49 entries
    "l4_5": True,  # PR-A4: 42 entries (diagnostics module)
    "l5": True,    # PR-A5: 30 entries (5 metric lists)
    "l6": True,    # PR-A5: 11 tests (DM/GW/DMP, GR/RS, MCS/SPA/RC/StepM)
    "l7": True,    # PR-A5: 15 entries (L7.B export shape)
    "l8": True,    # PR-A5: 62 entries (export/saved/provenance/granularity)
}


def _missing_for_layer(layer_id: str) -> list[tuple[str, str, str, str]]:
    missing: list[tuple[str, str, str, str]] = []
    for key in introspect.operational_options(layer_id):
        doc = OPTION_DOCS.get(key)
        if doc is None or not doc.is_tier1_complete():
            missing.append(key)
    return missing


@pytest.mark.parametrize("layer_id,required", sorted(_V1_REQUIRED_LAYERS.items()))
def test_layer_option_docs_complete_when_required(layer_id, required):
    """When ``required = True`` (i.e. the layer's content PR has landed),
    every operational option must have a Tier-1 OptionDoc entry."""

    missing = _missing_for_layer(layer_id)
    if not required:
        # Skip with a visible note so progress is trackable.
        pytest.skip(
            f"{layer_id}: docs not v1.0-required yet; {len(missing)} entries missing."
        )
    assert not missing, (
        f"{layer_id}: {len(missing)} operational options missing Tier-1 OptionDoc.\n"
        f"First 5 missing keys: {missing[:5]}"
    )


def test_registry_has_no_orphan_entries():
    """Every entry in OPTION_DOCS must point at a real (layer, sub-layer,
    axis, option) tuple in the schema. Catches stale docs lingering after
    a schema change removes an option."""

    orphans: list[tuple[str, str, str, str]] = []
    for key in OPTION_DOCS:
        layer_id, sublayer_id, axis_name, option_value = key
        operational = set(introspect.operational_options(layer_id))
        if key not in operational:
            orphans.append(key)
    assert not orphans, f"OptionDoc registry has orphans: {orphans}"


def test_v1_quality_floor():
    """Beyond the Tier-1 'every field is non-empty' gate, v1.0 docs must
    meet a substantive-content quality floor:

    * description >= 80 characters
    * when_to_use >= 30 characters
    * at least one academic / design reference

    Passing this gate is the precondition for the author validation
    gauntlet -- if every entry meets these minimums the author can
    spot-check rather than line-edit."""

    from macrocast.scaffold.option_docs import OPTION_DOCS

    short_desc: list[tuple] = []
    short_when: list[tuple] = []
    no_refs: list[tuple] = []
    for key, doc in OPTION_DOCS.items():
        if len(doc.description) < 80:
            short_desc.append((key, len(doc.description)))
        if len(doc.when_to_use) < 30:
            short_when.append((key, len(doc.when_to_use)))
        if not doc.references:
            no_refs.append(key)
    issues: list[str] = []
    if short_desc:
        issues.append(f"description<80 chars on {len(short_desc)} entries; first 5: {short_desc[:5]}")
    if short_when:
        issues.append(f"when_to_use<30 chars on {len(short_when)} entries; first 5: {short_when[:5]}")
    if no_refs:
        issues.append(f"no references on {len(no_refs)} entries; first 5: {no_refs[:5]}")
    assert not issues, "v1.0 quality floor violations:\n  - " + "\n  - ".join(issues)


def test_v1_release_gate_summary():
    """Aggregate progress meter -- prints the per-layer completion ratio
    so the v1.0 release-gate dashboard can read it from pytest output."""

    total_required = 0
    total_complete = 0
    for layer_id in introspect.list_layers():
        required = introspect.operational_options(layer_id)
        complete = sum(
            1
            for key in required
            if (doc := OPTION_DOCS.get(key)) is not None and doc.is_tier1_complete()
        )
        total_required += len(required)
        total_complete += complete
    coverage = (total_complete / total_required) if total_required else 1.0
    print(
        f"\n[v1.0 release gate] OptionDoc coverage: "
        f"{total_complete} / {total_required} ({coverage:.1%})"
    )
    # Always passes; this is a progress report, not a gate. The gate
    # lives in ``test_layer_option_docs_complete_when_required``.
    assert total_required >= 0
