"""Emit the source-committed Encyclopedia tree under ``docs/encyclopedia/``.

Replaces the legacy ``docs/reference/`` auto-emit (one giant per-layer RST
page per layer) with an encyclopedia-style markdown tree where every
layer / sub-layer / axis has a discoverable page or anchor:

* ``index.md``                 -- landing + global counts + 3 browse links.
* ``browse_by_layer.md``       -- L0..L8 + diagnostics, table form.
* ``browse_by_axis.md``        -- every axis A-Z, link to per-axis page.
* ``browse_by_option.md``      -- every option *value* A-Z (model families,
                                  L3 ops, file formats, ...).
* ``public_api.md``            -- curated public Python API surface
                                  (preserved from the legacy reference/).
* ``<layer_id>/index.md``      -- per-layer landing with sub-layers ×
                                  axes summary table.
* ``<layer_id>/axes/<axis>.md``-- per-axis page listing every option with
                                  full OptionDoc body when registered, or
                                  a "TBD: option doc not yet authored"
                                  placeholder when missing.

Generated entirely from ``macroforecast.scaffold.introspect`` (which walks
each layer's ``LayerImplementationSpec`` plus the L3/L4/L6/L7 fallback
manual axis lists) and the ``OPTION_DOCS`` registry. No content is
authored here -- this module is a pure renderer. CI re-runs it on every
push and diffs the output against ``docs/encyclopedia/`` to enforce that
the encyclopedia stays in sync with the schema.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from . import introspect
from .introspect import AxisInfo, LayerInfo, OptionInfo
from .option_docs import OPTION_DOCS, OptionDoc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _layer_display(layer_id: str) -> str:
    """``l1_5`` -> ``L1.5``, ``l4`` -> ``L4``."""

    return layer_id.upper().replace("_", ".")


def _slug(name: str) -> str:
    """File-safe identifier; option values may contain ``+`` or ``/``."""

    return (
        name.replace("+", "_plus_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
    )


def _status_badge(status: str) -> str:
    if status == "operational":
        return "operational"
    if status == "future":
        return "future"
    return status


# ---------------------------------------------------------------------------
# Per-op page helpers (Cycle 22 POC)
# ---------------------------------------------------------------------------


def _op_page_rel_path(*, layer_id: str, axis: str, option: str) -> str:
    """Relative path from the axis page to the per-op page.

    Axis page lives at ``<layer_id>/axes/<axis_slug>.md``.
    Per-op page lives at ``<layer_id>/<axis_slug>/<option_slug>.md``.
    Relative path: ``../<axis_slug>/<option_slug>.md``.
    """
    axis_slug = _slug(axis)
    option_slug = _slug(option)
    return f"../{axis_slug}/{option_slug}.md"


def _render_op_page(layer_id: str, axis: str, option: OptionInfo, doc: OptionDoc) -> str:
    """Render the dedicated per-op encyclopedia page for a function-op.

    Page location: ``<layer_id>/<axis_slug>/<option_slug>.md``

    Includes:
      - Header with layer / axis / sublayer context
      - ## Function signature  (Python code block)
      - ## Parameters         (table from OptionDoc.parameters)
      - ## Behavior           (from OptionDoc.description + when_to_use)
      - ## In recipe context  (YAML example stub or from OptionDoc.examples)
      - ## References
      - ## Related ops
    """
    layer_disp = _layer_display(layer_id)
    axis_slug = _slug(axis)
    func_name = doc.op_func_name or option.value

    parts: list[str] = []
    # Header
    parts.append(f"# `{option.value}` -- {doc.summary}")
    parts.append("")
    parts.append(
        f"[Back to `{axis}` axis](../axes/{axis_slug}.md) | "
        f"[Back to {layer_disp}](../index.md) | "
        f"[Browse all options](../../browse_by_option.md)"
    )
    parts.append("")
    parts.append(
        f"> Operational op under axis `{axis}`, sub-layer `{doc.sublayer}`, "
        f"layer `{layer_id}`."
    )
    if func_name:
        parts.append(
            f"> Standalone callable: `mf.functions.{func_name}`."
        )
    parts.append("")

    # Function signature
    # ParameterDoc.default=None means "required" (no default in the table).
    # For standalone functions in mf.functions, positional data args (no
    # default) are rendered first; optional keyword-only args (with defaults)
    # are grouped after *.
    parts.append("## Function signature")
    parts.append("")
    if doc.parameters:
        sig_lines = [f"mf.functions.{func_name}("]
        required = [p for p in doc.parameters if p.default is None]
        optional = [p for p in doc.parameters if p.default is not None]
        for p in required:
            sig_lines.append(f"    {p.name}: {p.type},")
        if optional:
            sig_lines.append("    *,")
            for p in optional:
                sig_lines.append(f"    {p.name}: {p.type} = {p.default!r},")
        sig_lines.append(")")
        parts.append("```python")
        parts.extend(sig_lines)
        parts.append("```")
    else:
        parts.append("```python")
        parts.append(f"mf.functions.{func_name}(...)")
        parts.append("```")
    parts.append("")

    # Parameters table
    if doc.parameters:
        parts.append("## Parameters")
        parts.append("")
        parts.append("| name | type | default | constraint | description |")
        parts.append("|---|---|---|---|---|")
        for p in doc.parameters:
            default_cell = f"`{p.default!r}`" if p.default is not None else "—"
            constraint_cell = p.constraint if p.constraint else "—"
            parts.append(
                f"| `{p.name}` | `{p.type}` | {default_cell} | {constraint_cell} | {p.description} |"
            )
        parts.append("")

    # Behavior
    parts.append("## Behavior")
    parts.append("")
    if doc.description:
        for paragraph in doc.description.split("\n\n"):
            parts.append(paragraph.strip())
            parts.append("")
    if doc.when_to_use:
        parts.append("**When to use**")
        parts.append("")
        parts.append(doc.when_to_use)
        parts.append("")
    if doc.when_not_to_use:
        parts.append("**When NOT to use**")
        parts.append("")
        parts.append(doc.when_not_to_use)
        parts.append("")

    # In recipe context
    parts.append("## In recipe context")
    parts.append("")
    if doc.examples:
        for example in doc.examples:
            parts.append(f"*{example.title}*")
            parts.append("")
            parts.append(f"```{example.language}")
            parts.append(example.code)
            parts.append("```")
            parts.append("")
    else:
        # Auto-generate minimal YAML recipe snippet.
        parts.append(
            f"Set ``params.{axis} = \"{option.value}\"`` in the relevant layer "
            f"to activate this op within a recipe:"
        )
        parts.append("")
        parts.append("```yaml")
        parts.append(f"# Layer {layer_id.upper()} recipe fragment")
        parts.append(f"params:")
        parts.append(f"  {axis}: {option.value}")
        parts.append("```")
        parts.append("")

    # References
    if doc.references:
        parts.append("## References")
        parts.append("")
        for ref in doc.references:
            text = ref.citation
            if ref.doi:
                text += f" (doi:{ref.doi})"
            if ref.url:
                text += f" <{ref.url}>"
            parts.append(f"* {text}")
        parts.append("")

    # Related ops
    if doc.related_options:
        parts.append("## Related ops")
        parts.append("")
        rel = ", ".join(f"`{name}`" for name in doc.related_options)
        parts.append(f"See also: {rel} (on the same axis).")
        parts.append("")

    if doc.last_reviewed:
        suffix = f" by {doc.reviewer}" if doc.reviewer else ""
        parts.append(f"_Last reviewed {doc.last_reviewed}{suffix}._")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Per-option rendering (markdown body for one option on one axis page)
# ---------------------------------------------------------------------------


def _render_option_body(option: OptionInfo, doc: OptionDoc | None, *, layer_id: str, sublayer: str, axis: str) -> str:
    """Return the markdown ``### option`` block for one option."""

    badge = _status_badge(option.status)
    lines: list[str] = []
    lines.append(f"### `{option.value}`  --  {badge}")
    lines.append("")

    if doc is None:
        # Schema-only: show the schema description if we have one, plus the
        # explicit "TBD" marker so a reader knows OptionDoc authoring is
        # still pending. (The completeness test in v1.0 ensures this list
        # only shrinks over time.)
        if option.description:
            lines.append(option.description)
        else:
            lines.append(f"_(no schema description for `{option.value}`)_")
        lines.append("")
        lines.append(
            "> TBD: option doc not yet authored for this value."
            " The encyclopedia falls back to the bare schema description"
            " above. PRs adding a full ``OptionDoc`` entry under"
            f" ``macroforecast/scaffold/option_docs/{layer_id}.py`` are"
            " welcome."
        )
        lines.append("")
        if option.leaf_config_required:
            lines.append(
                "**Required leaf-config keys**: "
                + ", ".join(f"`{k}`" for k in option.leaf_config_required)
            )
            lines.append("")
        if option.leaf_config_optional:
            lines.append(
                "**Optional leaf-config keys**: "
                + ", ".join(f"`{k}`" for k in option.leaf_config_optional)
            )
            lines.append("")
        return "\n".join(lines)

    # Per-op page: emit a 1-2 line stub + link instead of the full body.
    if doc.op_page:
        op_dir = _op_page_rel_path(layer_id=layer_id, axis=axis, option=option.value)
        stub_link = f"[{option.value} function page]({op_dir})"
        func_note = ""
        if doc.op_func_name:
            func_note = f" Standalone: ``mf.functions.{doc.op_func_name}``."
        lines.append(doc.summary)
        lines.append("")
        lines.append(f"See {stub_link} for full documentation + parameters + standalone usage.{func_note}")
        lines.append("")
        return "\n".join(lines)

    # Full OptionDoc.
    lines.append(doc.summary)
    lines.append("")
    if doc.description:
        for paragraph in doc.description.split("\n\n"):
            lines.append(paragraph.strip())
            lines.append("")
    if doc.when_to_use:
        lines.append("**When to use**")
        lines.append("")
        lines.append(doc.when_to_use)
        lines.append("")
    if doc.when_not_to_use:
        lines.append("**When NOT to use**")
        lines.append("")
        lines.append(doc.when_not_to_use)
        lines.append("")
    if doc.references:
        lines.append("**References**")
        lines.append("")
        for ref in doc.references:
            text = ref.citation
            if ref.doi:
                text += f" (doi:{ref.doi})"
            if ref.url:
                text += f" <{ref.url}>"
            lines.append(f"* {text}")
        lines.append("")
    if doc.related_options:
        # Related options usually live on the *same* axis page (same family
        # axis, etc.). Use relative anchor links rather than separate pages.
        # MyST auto-generates ``#<heading>`` slugs; the heading "`<value>` --
        # operational" lowercases + de-spaces to the anchor below.
        def _anchor(value: str) -> str:
            return (
                value.lower()
                .replace("_", "-")
                .replace("+", "-plus-")
                .replace(".", "")
                .replace("/", "-")
                .replace(":", "")
            )

        rel = ", ".join(
            f"[`{name}`](#{_anchor(name)})" for name in doc.related_options
        )
        lines.append(f"**Related options**: {rel}")
        lines.append("")
    if doc.examples:
        lines.append("**Examples**")
        lines.append("")
        for example in doc.examples:
            lines.append(f"*{example.title}*")
            lines.append("")
            lines.append(f"```{example.language}")
            lines.append(example.code)
            lines.append("```")
            lines.append("")
    if doc.parameters:
        lines.append("**Parameters**")
        lines.append("")
        lines.append("| name | type | default | constraint | description |")
        lines.append("|---|---|---|---|---|")
        for p in doc.parameters:
            default_cell = f"`{p.default!r}`" if p.default is not None else "—"
            constraint_cell = p.constraint if p.constraint else "—"
            lines.append(
                f"| `{p.name}` | `{p.type}` | {default_cell} | {constraint_cell} | {p.description} |"
            )
        lines.append("")
    if option.leaf_config_required:
        lines.append(
            "**Required leaf-config keys**: "
            + ", ".join(f"`{k}`" for k in option.leaf_config_required)
        )
        lines.append("")
    if option.leaf_config_optional:
        lines.append(
            "**Optional leaf-config keys**: "
            + ", ".join(f"`{k}`" for k in option.leaf_config_optional)
        )
        lines.append("")
    if doc.last_reviewed:
        suffix = f" by {doc.reviewer}" if doc.reviewer else ""
        lines.append(f"_Last reviewed {doc.last_reviewed}{suffix}._")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-axis page
# ---------------------------------------------------------------------------


def _render_axis_page(layer_id: str, axis: AxisInfo) -> str:
    """Return the full markdown text for one axis page."""

    layer_disp = _layer_display(layer_id)
    sublayer_label = axis.sublayer
    n_op = sum(1 for o in axis.options if o.status == "operational")
    n_future = sum(1 for o in axis.options if o.status == "future")

    parts: list[str] = []
    parts.append(f"# `{axis.name}`")
    parts.append("")
    parts.append(
        f"[Back to {layer_disp}](../index.md) | "
        f"[Browse all axes](../../browse_by_axis.md) | "
        f"[Browse all options](../../browse_by_option.md)"
    )
    parts.append("")
    parts.append(
        f"> Axis ``{axis.name}`` on sub-layer ``{sublayer_label}`` "
        f"(layer ``{layer_id}``)."
    )
    parts.append("")
    parts.append("## Sub-layer")
    parts.append("")
    parts.append(f"**{sublayer_label}**" + (" (gated)" if axis.has_gate else ""))
    parts.append("")
    parts.append("## Axis metadata")
    parts.append("")
    parts.append(f"- Default: `{axis.default!r}`")
    parts.append(f"- Sweepable: {axis.sweepable}")
    parts.append(f"- Status: {axis.status}")
    if axis.leaf_config_keys:
        parts.append(
            "- Leaf-config keys: "
            + ", ".join(f"`{k}`" for k in axis.leaf_config_keys)
        )
    parts.append("")
    parts.append("## Operational status summary")
    parts.append("")
    parts.append(f"- Operational: {n_op} option(s)")
    parts.append(f"- Future: {n_future} option(s)")
    parts.append("")

    if not axis.options:
        # Numeric / continuous axis with no enumerated options (e.g. L6
        # ``mcs_alpha``, ``cpa_window_type``). Surface a stub note.
        parts.append("## Options")
        parts.append("")
        parts.append(
            "_This axis takes a free-form value (numeric, list, or"
            " other non-enumerated leaf-config datum); see the layer"
            " architecture page for the accepted shape._"
        )
        parts.append("")
    else:
        parts.append("## Options")
        parts.append("")
        for option in axis.options:
            doc = OPTION_DOCS.get((layer_id, sublayer_label, axis.name, option.value))
            parts.append(
                _render_option_body(
                    option, doc, layer_id=layer_id, sublayer=sublayer_label, axis=axis.name
                )
            )

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Per-layer index page
# ---------------------------------------------------------------------------


def _render_layer_index(layer_id: str) -> str:
    info = introspect.layer(layer_id)
    layer_axes = introspect.axes(layer_id)
    layer_disp = _layer_display(layer_id)

    # Group axes by their sublayer (preserving declaration order).
    axes_by_sublayer: dict[str, list[AxisInfo]] = defaultdict(list)
    for ax in layer_axes:
        axes_by_sublayer[ax.sublayer].append(ax)

    parts: list[str] = []
    parts.append(f"# Layer {layer_disp}  --  {info.name}")
    parts.append("")
    parts.append(
        f"[Back to encyclopedia](../index.md) | "
        f"[Browse layers](../browse_by_layer.md) | "
        f"[Browse all axes](../browse_by_axis.md)"
    )
    parts.append("")
    parts.append(f"- Layer ID: `{layer_id}`")
    parts.append(f"- Category: `{info.category}`")
    parts.append(f"- Sub-layers: {len(info.sub_layers)}")
    parts.append(f"- Axes: {len(layer_axes)}")
    parts.append(
        f"- Options across axes: "
        f"{sum(len(a.options) for a in layer_axes)}"
    )
    parts.append("")

    parts.append("## Sub-layers")
    parts.append("")
    parts.append("| Sub-layer | Name | Gate | Axes |")
    parts.append("|---|---|---|---|")
    for sub in info.sub_layers:
        sub_axes = axes_by_sublayer.get(sub.id, [])
        gate_marker = "gated" if sub.has_gate else "always"
        if sub_axes:
            axis_links = ", ".join(
                f"[{a.name}](axes/{_slug(a.name)}.md)" for a in sub_axes
            )
        else:
            axis_links = (
                "_no axis options at this sub-layer; see operational ops "
                "in `core/ops/`._"
            )
        parts.append(f"| `{sub.id}` | {sub.name} | {gate_marker} | {axis_links} |")
    parts.append("")

    if layer_axes:
        parts.append("```{toctree}")
        parts.append(":hidden:")
        parts.append(":maxdepth: 1")
        parts.append("")
        seen_axis_files: set[str] = set()
        for ax in layer_axes:
            slug = _slug(ax.name)
            if slug in seen_axis_files:
                continue
            seen_axis_files.add(slug)
            parts.append(f"axes/{slug}")
        parts.append("```")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Top-level browse pages
# ---------------------------------------------------------------------------


def _render_top_index(stats: dict[str, int]) -> str:
    parts: list[str] = []
    parts.append("# Encyclopedia")
    parts.append("")
    parts.append(
        "Encyclopedia-style browse for every macroforecast schema choice. "
        "Each layer, sub-layer, axis, and option value has its own page or "
        "anchor; the tree below is generated from the live "
        "`LayerImplementationSpec` registry plus the `OPTION_DOCS` "
        "documentation registry under `macroforecast/scaffold/option_docs/`."
    )
    parts.append("")
    parts.append(
        "> **Looking for the design narrative instead?** Use "
        "[Architecture](../architecture/index.md) -- that's where the "
        "prose \"why is L2 separated from L3\" / \"how does L7 read L4 "
        "sinks\" / cross-layer reference explanations live. Encyclopedia "
        "pages here are **auto-generated lookup** for individual option "
        "values (description, when to use, when NOT, references, related "
        "options); Architecture pages there are **hand-written narrative** "
        "for the design contracts. Both are sourced from the same "
        "`LayerImplementationSpec` registry -- encyclopedia is the "
        "machine-locked option dictionary, architecture is the "
        "human-edited design guide."
    )
    parts.append("")
    parts.append("## Counts")
    parts.append("")
    parts.append(f"- Layers: {stats['n_layers']}")
    parts.append(f"- Sub-layers: {stats['n_sublayers']}")
    parts.append(f"- Axes (operational + future): {stats['n_axes']}")
    parts.append(f"- Option values: {stats['n_options']}")
    parts.append(f"- OptionDoc entries (full prose): {stats['n_documented']}")
    parts.append("")
    parts.append("## Browse")
    parts.append("")
    parts.append("- [Browse by layer](browse_by_layer.md)  --  L0 to L8 + diagnostics, table form.")
    parts.append("- [Browse by axis](browse_by_axis.md)  --  every axis A-Z.")
    parts.append("- [Browse by option](browse_by_option.md)  --  every option *value* A-Z (e.g. `ridge`, `pca`, `ar_p`).")
    parts.append("- [Public Python API](public_api.md)  --  curated `macroforecast.run` / `macroforecast.replicate` surface.")
    parts.append("")
    parts.append("## Layer pages")
    parts.append("")
    parts.append("```{toctree}")
    parts.append(":maxdepth: 1")
    parts.append("")
    for layer_id in introspect.list_layers():
        parts.append(f"{layer_id}/index")
    parts.append("```")
    parts.append("")
    parts.append("```{toctree}")
    parts.append(":hidden:")
    parts.append(":maxdepth: 1")
    parts.append("")
    parts.append("browse_by_layer")
    parts.append("browse_by_axis")
    parts.append("browse_by_option")
    parts.append("public_api")
    parts.append("```")
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_browse_by_layer() -> str:
    parts: list[str] = []
    parts.append("# Browse by layer")
    parts.append("")
    parts.append("[Back to encyclopedia](index.md)")
    parts.append("")
    parts.append("| Layer | Name | Category | Sub-layers | Axes | Options |")
    parts.append("|---|---|---|---:|---:|---:|")
    for layer_id in introspect.list_layers():
        info = introspect.layer(layer_id)
        layer_axes = introspect.axes(layer_id)
        n_options = sum(len(a.options) for a in layer_axes)
        parts.append(
            f"| [`{layer_id}`]({layer_id}/index.md) "
            f"| {info.name} "
            f"| `{info.category}` "
            f"| {len(info.sub_layers)} "
            f"| {len(layer_axes)} "
            f"| {n_options} |"
        )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _all_axes() -> list[tuple[str, AxisInfo]]:
    """Return every (layer_id, AxisInfo) pair across all layers."""

    out: list[tuple[str, AxisInfo]] = []
    for layer_id in introspect.list_layers():
        for ax in introspect.axes(layer_id):
            out.append((layer_id, ax))
    return out


def _render_browse_by_axis() -> str:
    parts: list[str] = []
    parts.append("# Browse by axis")
    parts.append("")
    parts.append("[Back to encyclopedia](index.md)")
    parts.append("")
    parts.append("Every axis across every layer, sorted A-Z by axis name. Same axis name can appear on more than one layer (e.g. ``selection_method`` shows up on multiple diagnostic layers); each row links to the page for that specific (layer, axis) pair.")
    parts.append("")
    parts.append("| Axis | Layer | Sub-layer | Status | Sweepable | Options |")
    parts.append("|---|---|---|---|---|---:|")
    rows = sorted(
        _all_axes(),
        key=lambda pair: (pair[1].name, pair[0], pair[1].sublayer),
    )
    for layer_id, ax in rows:
        slug = _slug(ax.name)
        layer_disp = _layer_display(layer_id)
        sweep = "yes" if ax.sweepable else "no"
        parts.append(
            f"| [`{ax.name}`]({layer_id}/axes/{slug}.md) "
            f"| {layer_disp} "
            f"| `{ax.sublayer}` "
            f"| {ax.status} "
            f"| {sweep} "
            f"| {len(ax.options)} |"
        )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_browse_by_option() -> str:
    """Index every option *value* across every (layer, sublayer, axis)."""

    rows: list[tuple[str, str, AxisInfo, OptionInfo]] = []
    for layer_id in introspect.list_layers():
        for ax in introspect.axes(layer_id):
            for option in ax.options:
                rows.append((layer_id, option.value, ax, option))
    rows.sort(key=lambda row: (row[1].lower(), row[0], row[2].name))

    parts: list[str] = []
    parts.append("# Browse by option")
    parts.append("")
    parts.append("[Back to encyclopedia](index.md)")
    parts.append("")
    parts.append(
        "Every option *value* across every axis, sorted A-Z. Same value can "
        "appear on more than one axis (e.g. `multi`, `none`, `auto`); each "
        "row links to the option's section on the relevant axis page."
    )
    parts.append("")
    parts.append("| Option | Layer | Axis | Status |")
    parts.append("|---|---|---|---|")
    for layer_id, value, ax, option in rows:
        slug_axis = _slug(ax.name)
        # Markdown anchor: GitHub/MyST lowercase + replace special chars.
        anchor = (
            f"{value}"
            .lower()
            .replace("_", "-")
            .replace("+", "-plus-")
            .replace(".", "")
            .replace("/", "-")
            .replace(":", "")
        )
        # Heading is "`value`  --  status", so anchor is roughly "value----status";
        # keep it simple and link to the page (myst will surface the anchor list).
        link = f"{layer_id}/axes/{slug_axis}.md"
        parts.append(
            f"| [`{value}`]({link}) "
            f"| {_layer_display(layer_id)} "
            f"| `{ax.name}` "
            f"| {option.status} |"
        )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_public_api() -> str:
    """Preserve the curated public-API table from the legacy
    ``docs/reference/public_api.md`` under encyclopedia/. Pure curated
    content -- not generated from the schema."""

    return (
        "# Public Python API\n"
        "\n"
        "[Back to encyclopedia](index.md)\n"
        "\n"
        "Curated reference for the public surface of the macroforecast"
        " package. The encyclopedia tree above documents the YAML *recipe*"
        " surface (axes / options); this page documents the Python"
        " *package* surface (importable symbols).\n"
        "\n"
        "## Top-level API\n"
        "\n"
        "| Symbol | Description |\n"
        "|--------|-------------|\n"
        "| `macroforecast.run(recipe, output_directory=...)` | Execute a recipe end-to-end (L1->L8). Iterates every `{sweep: [...]}` cell, applies L0 failure_policy + seed, returns a `ManifestExecutionResult`. |\n"
        "| `macroforecast.replicate(manifest_path)` | Re-execute a stored recipe and verify per-cell sink hashes match bit-for-bit. |\n"
        "| `macroforecast.ManifestExecutionResult` | Per-cell `RuntimeResult` + `sink_hashes`; serializes to `manifest.json`. |\n"
        "| `macroforecast.ReplicationResult` | `recipe_match`, `sink_hashes_match`, `per_cell_match`. |\n"
        "\n"
        "## Submodule surfaces\n"
        "\n"
        "| Module | Purpose |\n"
        "|--------|---------|\n"
        "| `macroforecast.core` | 12-layer DAG runtime (foundation, layers, ops, runtime, execution, figures) |\n"
        "| `macroforecast.raw` | FRED-MD/QD/SD adapters, vintage manager, manifest |\n"
        "| `macroforecast.preprocessing` | Preprocessing contract helpers |\n"
        "| `macroforecast.tuning` | Hyperparameter search engines |\n"
        "| `macroforecast.custom` | User-defined model/preprocessor/feature registration |\n"
        "| `macroforecast.defaults` | Default profile dict template |\n"
        "\n"
        "## Layer modules\n"
        "\n"
        "`macroforecast.core.layers.l{0..8}` (plus `l{1,2,3,4}_5` diagnostics) hold the\n"
        "canonical schema (`LayerImplementationSpec`) for each layer. Runtime\n"
        "materialization helpers live in `macroforecast.core.runtime`:\n"
        "\n"
        "- `materialize_l1`, `materialize_l2`, `materialize_l3_minimal`,\n"
        "  `materialize_l4_minimal`, `materialize_l5_minimal`,\n"
        "  `materialize_l6_runtime`, `materialize_l7_runtime`,\n"
        "  `materialize_l8_runtime`\n"
        "- `materialize_l1_5_diagnostic` ... `materialize_l4_5_diagnostic`\n"
        "- `execute_minimal_forecast(recipe)` -- single-cell convenience wrapper\n"
        "\n"
        "Sweep loop + bit-exact replicate are in `macroforecast.core.execution`:\n"
        "`execute_recipe`, `replicate_recipe`, `CellExecutionResult`,\n"
        "`ManifestExecutionResult`, `ReplicationResult`.\n"
        "\n"
        "Figure rendering (matplotlib + stylized US state choropleth) is in\n"
        "`macroforecast.core.figures`: `render_bar_global`, `render_heatmap`,\n"
        "`render_pdp_line`, `render_us_state_choropleth`,\n"
        "`render_default_for_op`.\n"
        "\n"
        "## Operational coverage\n"
        "\n"
        "See [`CLAUDE.md`](https://github.com/NanyeonK/macroforecast/blob/main/CLAUDE.md) at the repo root for the operational\n"
        "matrix: 30+ model families, 37 L3 ops, 7 L6 sub-layers, 29 L7 importance\n"
        "ops, FRED-SD US state choropleth, parquet/latex/markdown export.\n"
    )


# ---------------------------------------------------------------------------
# write_all entry point
# ---------------------------------------------------------------------------


def _gather_stats() -> dict[str, int]:
    n_layers = len(introspect.list_layers())
    n_sublayers = 0
    n_axes = 0
    n_options = 0
    documented_keys = set(OPTION_DOCS.keys())
    n_documented = 0
    for layer_id in introspect.list_layers():
        info = introspect.layer(layer_id)
        n_sublayers += len(info.sub_layers)
        for ax in introspect.axes(layer_id):
            n_axes += 1
            for option in ax.options:
                n_options += 1
                if (layer_id, ax.sublayer, ax.name, option.value) in documented_keys:
                    n_documented += 1
    return {
        "n_layers": n_layers,
        "n_sublayers": n_sublayers,
        "n_axes": n_axes,
        "n_options": n_options,
        "n_documented": n_documented,
    }


def write_all(out_dir: str | Path) -> list[Path]:
    """Render the entire encyclopedia tree under ``out_dir`` and return
    the list of files written.

    Idempotent: any existing files at the target paths are overwritten,
    but no unrelated files are removed (the caller is responsible for
    starting from a clean directory if exact diff is desired -- which is
    what the CI sync gate does)."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Root index + browse pages.
    stats = _gather_stats()
    (out / "index.md").write_text(_render_top_index(stats), encoding="utf-8")
    written.append(out / "index.md")
    (out / "browse_by_layer.md").write_text(_render_browse_by_layer(), encoding="utf-8")
    written.append(out / "browse_by_layer.md")
    (out / "browse_by_axis.md").write_text(_render_browse_by_axis(), encoding="utf-8")
    written.append(out / "browse_by_axis.md")
    (out / "browse_by_option.md").write_text(_render_browse_by_option(), encoding="utf-8")
    written.append(out / "browse_by_option.md")
    (out / "public_api.md").write_text(_render_public_api(), encoding="utf-8")
    written.append(out / "public_api.md")

    # Per-layer pages.
    for layer_id in introspect.list_layers():
        layer_dir = out / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        axes_dir = layer_dir / "axes"
        axes_dir.mkdir(parents=True, exist_ok=True)

        index_path = layer_dir / "index.md"
        index_path.write_text(_render_layer_index(layer_id), encoding="utf-8")
        written.append(index_path)

        seen_files: set[Path] = set()
        for ax in introspect.axes(layer_id):
            slug = _slug(ax.name)
            axis_path = axes_dir / f"{slug}.md"
            if axis_path in seen_files:
                # Two axes can share a slug after sanitisation -- guard.
                continue
            seen_files.add(axis_path)
            axis_path.write_text(_render_axis_page(layer_id, ax), encoding="utf-8")
            written.append(axis_path)

            # Cycle 22 POC: emit per-op page for options with op_page=True.
            for option in ax.options:
                doc = OPTION_DOCS.get(
                    (layer_id, ax.sublayer, ax.name, option.value)
                )
                if doc is None or not doc.op_page:
                    continue
                op_dir = layer_dir / slug
                op_dir.mkdir(parents=True, exist_ok=True)
                op_slug = _slug(option.value)
                op_page_path = op_dir / f"{op_slug}.md"
                op_page_path.write_text(
                    _render_op_page(layer_id, ax.name, option, doc),
                    encoding="utf-8",
                )
                written.append(op_page_path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m macroforecast.scaffold.render_encyclopedia",
        description="Emit the source-committed Encyclopedia tree (markdown).",
    )
    parser.add_argument(
        "-o", "--output",
        default="docs/encyclopedia",
        help="Output directory (default: docs/encyclopedia/).",
    )
    args = parser.parse_args(argv)
    written = write_all(args.output)
    print(f"wrote {len(written)} encyclopedia pages to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
