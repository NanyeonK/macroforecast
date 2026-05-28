"""Emit the source-committed generated reference tree under ``docs/reference/``.

Maintains the generated option-reference pages with an encyclopedia-style markdown tree where every
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
push and diffs the generated output under ``docs/reference/`` to enforce that
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
from .option_docs.types import REQUIRED


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
# Per-op page helpers (v0.8.x)
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
    # data_args = positional data inputs (X/y, y_true/y_pred) -- always before *.
    # parameters = keyword-only tuning args -- default=REQUIRED means no default
    #              (positional required kwarg); default=None means actual None.
    parts.append("## Function signature")
    parts.append("")
    has_data_args = bool(doc.data_args)
    has_params = bool(doc.parameters)
    if has_data_args or has_params:
        callable_name = f"mf.functions.{func_name}"
        sig_lines = [f"{callable_name}("]
        # Positional data args -- always first, never show a default.
        for p in doc.data_args:
            sig_lines.append(f"    {p.name}: {p.type},")
        # Keyword-only separator only when there are keyword params.
        if has_params:
            sig_lines.append("    *,")
            for p in doc.parameters:
                if p.default is REQUIRED:
                    sig_lines.append(f"    {p.name}: {p.type},")
                else:
                    sig_lines.append(f"    {p.name}: {p.type} = {p.default!r},")
        # Close paren, optionally with return type.
        if doc.return_type:
            sig_lines.append(f") -> {doc.return_type}")
        else:
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
    all_params = list(doc.data_args) + list(doc.parameters)
    if all_params:
        parts.append("## Parameters")
        parts.append("")
        parts.append("| name | type | default | constraint | description |")
        parts.append("|---|---|---|---|---|")
        for p in all_params:
            default_cell = "—" if p.default is REQUIRED else f"`{p.default!r}`"
            constraint_cell = p.constraint if p.constraint else "—"
            parts.append(
                f"| `{p.name}` | `{p.type}` | {default_cell} | {constraint_cell} | {p.description} |"
            )
        parts.append("")

    # Returns section
    if doc.return_type:
        parts.append("## Returns")
        parts.append("")
        if doc.returns_attrs:
            parts.append(f"`{doc.return_type}` — frozen dataclass with fit results.")
            parts.append("")
            parts.append("| Attribute | Type | Description |")
            parts.append("|-----------|------|-------------|")
            for attr_name, attr_type, attr_desc in doc.returns_attrs:
                parts.append(f"| `{attr_name}` | `{attr_type}` | {attr_desc} |")
            parts.append("")
        else:
            # Scalar return (e.g., float) -- one-line description.
            parts.append(f"`{doc.return_type}` — scalar result.")
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
            f" ``tools/docgen/option_docs/{layer_id}.py`` are"
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
            default_cell = "—" if p.default is REQUIRED else f"`{p.default!r}`"
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
        f"[Back to reference](../index.md) | "
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
    parts.append("# Reference")
    parts.append("")
    parts.append(
        "Lookup pages for the public Python surface, recipe YAML "
        "contract, and generated layer option dictionary. These pages are "
        "for exact lookup; tutorials and design narrative live elsewhere."
    )
    parts.append("")
    parts.append(
        "> **Looking for design rationale?** Use "
        "[Architecture](../explanation/architecture/index.md). Reference pages "
        "define names, keys, options, and import surfaces; architecture pages "
        "explain why the layers are separated and how they interact."
    )
    parts.append("")
    parts.append("## Recipe API")
    parts.append("")
    parts.append("- [Recipe gallery](gallery.md): runnable examples.")
    parts.append("- [Layer contract](layer_contract.md): layer keys, graph node shape, and complete recipe form.")
    parts.append("- [Data](data.md): source, target, horizon, and geography choices.")
    parts.append("- [Data policies](data_policies.md): missingness, outliers, release lags, and same-period predictors.")
    parts.append("- [Defaults](defaults.md): package-level default profiles.")
    parts.append("- [Runtime support](runtime_support.md): what executes today.")
    parts.append("- [Output](output.md): artifact directory and manifest layout.")
    parts.append("- [FRED datasets](fred_datasets.md): FRED-MD, FRED-QD, and FRED-SD reference status.")
    parts.append("")
    parts.append("## Python Surface")
    parts.append("")
    parts.append("- [Public Python API](public_api.md): top-level imports and semantic package map.")
    parts.append("- [Standalone functions](standalone_functions/index.md): direct `mf.functions.*` callables.")
    parts.append("- [Navigator](navigator/tree_navigator.md): layer-topology navigator details.")
    parts.append("")
    parts.append("## Option Encyclopedia")
    parts.append("")
    parts.append(
        "Generated from the live `LayerImplementationSpec` registry plus the "
        "`OPTION_DOCS` registry under `tools/docgen/option_docs/`."
    )
    parts.append("")
    parts.append(f"- Layers: {stats['n_layers']}")
    parts.append(f"- Sub-layers: {stats['n_sublayers']}")
    parts.append(f"- Axes (operational + future): {stats['n_axes']}")
    parts.append(f"- Option values: {stats['n_options']}")
    parts.append(f"- OptionDoc entries: {stats['n_documented']}")
    parts.append("")
    parts.append("Browse indexes:")
    parts.append("")
    parts.append("- [Browse by layer](browse_by_layer.md)")
    parts.append("- [Browse by axis](browse_by_axis.md)")
    parts.append("- [Browse by option](browse_by_option.md)")
    parts.append("")
    parts.append("## Layer Pages")
    parts.append("")
    parts.append("```{toctree}")
    parts.append(":maxdepth: 1")
    parts.append("")
    parts.append("gallery")
    parts.append("layer_contract")
    parts.append("data")
    parts.append("data_policies")
    parts.append("defaults")
    parts.append("runtime_support")
    parts.append("output")
    parts.append("fred_datasets")
    parts.append("public_api")
    parts.append("standalone_functions/index")
    parts.append("navigator/tree_navigator")
    parts.append("browse_by_layer")
    parts.append("browse_by_axis")
    parts.append("browse_by_option")
    for layer_id in introspect.list_layers():
        parts.append(f"{layer_id}/index")
    parts.append("```")
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_browse_by_layer() -> str:
    parts: list[str] = []
    parts.append("# Browse by layer")
    parts.append("")
    parts.append("[Back to reference](index.md)")
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
    parts.append("[Back to reference](index.md)")
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
    parts.append("[Back to reference](index.md)")
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
    """Render the curated public Python surface page."""

    return """# Public Python API

[Back to reference](index.md)

Curated reference for the importable macroforecast surface. The generated layer pages document recipe axes and options; this page documents Python imports and semantic package ownership.

## Top-level API

| Symbol | Description |
|--------|-------------|
| `macroforecast.forecast(...)` | One-shot helper that assembles a default recipe and runs it. |
| `macroforecast.Experiment(...)` | Builder with `compare_models`, `compare`, `sweep`, `run`, `replicate`, `to_yaml`, and `validate`. |
| `macroforecast.ForecastResult` | Thin facade over a recipe execution result. |
| `macroforecast.run(recipe, output_directory=...)` | Execute a recipe end-to-end and return `ManifestExecutionResult`. |
| `macroforecast.run_file(path, output_directory=...)` | Execute a recipe file. |
| `macroforecast.replicate(manifest_path)` | Re-execute a stored manifest and verify sink hashes. |
| `macroforecast.ManifestExecutionResult` | Per-cell `RuntimeResult` plus sink hashes. |
| `macroforecast.ReplicationResult` | Bit-exact replication comparison result. |
| `macroforecast.l0(...)` | Callable L0/meta recipe block builder; equivalent to authoring `0_meta` in YAML. |

## Callable Recipe Blocks

| Symbol | Description |
|--------|-------------|
| `macroforecast.meta.configure(...)` | Build and validate a canonical `0_meta` block. |
| `macroforecast.meta.l0(...)` | Alias for `configure(...)`; also exported as `macroforecast.l0(...)`. |
| `macroforecast.meta.build_layer_block(...)` | Internal helper for the body under `0_meta`. |

## Submodule Surfaces

| Module | Purpose |
|--------|---------|
| `macroforecast.recipes` | Recipe orchestration namespace; top-level `run`, `replicate`, `Experiment`, and `forecast` route here. |
| `macroforecast.meta` | L0 study setup, failure policy, reproducibility, and compute policy. |
| `macroforecast.data` | FRED-MD/QD/SD adapters, vintage manager, manifests, cache helpers. |
| `macroforecast.preprocessing` | L2 preprocessing schemas, transformations, and contract helpers. |
| `macroforecast.features` | L3 feature engineering ops, transforms, and selectors. |
| `macroforecast.models` | L4 model classes, model ops, paper helpers, and tuning. |
| `macroforecast.evaluation` | L5 metrics and evaluation ops. |
| `macroforecast.stat_tests` | L6 forecast-comparison statistical tests. |
| `macroforecast.interpretation` | L7 interpretation schemas, ops, and methods. |
| `macroforecast.output` | L8 artifact, provenance, and export ops. |
| `macroforecast.diagnostics` | L1.5/L2.5/L3.5/L4.5 diagnostic packages. |
| `macroforecast.core` | Cross-layer runtime, registry, manifest, cache, validation, execution, and figures. |
| `macroforecast.api.functions` | Canonical standalone callable namespace; also available as `macroforecast.functions`. |
| `macroforecast.api.defaults` | Canonical default profile helpers; also available through top-level lazy exports and `macroforecast.defaults`. |
| `macroforecast.api.custom` | Custom model, preprocessor, feature, and target-transform registration. |
| `macroforecast.feature_selection` | Promoted compatibility namespace for selector classes. |
| `macroforecast.transforms` | Promoted compatibility namespace for transform callables. |

## Layer Module Ownership

Canonical implementation now lives in semantic packages: `meta`, `data`, `preprocessing`, `features`, `models`, `evaluation`, `stat_tests`, `interpretation`, `output`, and `diagnostics.*`.

`macroforecast.layers.*` is compatibility-only. `macroforecast.core.layers` owns registry-facing compatibility modules and runtime glue, not the primary public implementation surface.

## Runtime Helpers

Runtime materialization helpers live in `macroforecast.core.runtime`:

- `materialize_l1`, `materialize_l2`, `materialize_l3_minimal`, `materialize_l4_minimal`, `materialize_l5_minimal`, `materialize_l6_runtime`, `materialize_l7_runtime`, `materialize_l8_runtime`
- `materialize_l1_5_diagnostic` through `materialize_l4_5_diagnostic`
- `execute_minimal_forecast(recipe)`

Sweep execution and bit-exact replication live in `macroforecast.core.execution`: `execute_recipe`, `replicate_recipe`, `CellExecutionResult`, `ManifestExecutionResult`, and `ReplicationResult`.
"""




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

            # emit per-op page for options with op_page=True (v0.8.x).
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
        default="docs/reference",
        help="Output directory (default: docs/reference/).",
    )
    args = parser.parse_args(argv)
    written = write_all(args.output)
    print(f"wrote {len(written)} encyclopedia pages to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
