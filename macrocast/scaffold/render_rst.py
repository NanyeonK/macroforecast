"""Emit reference RST pages from the OptionDoc registry.

Used by ``docs/source/reference/*.rst`` (auto-generated at sphinx
build time) and as a stand-alone ``python -m macrocast.scaffold.render_rst``
entry for CI verification that the docs are in sync with the schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import introspect
from .option_docs import OPTION_DOCS, OptionDoc


def _section(title: str, char: str = "-") -> str:
    return f"{title}\n{char * len(title)}\n"


def _format_option(doc: OptionDoc) -> str:
    lines: list[str] = []
    lines.append(_section(f"``{doc.option}``", char="^"))
    lines.append("")
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
            lines.append(f"* {ref.to_rst()}")
        lines.append("")
    if doc.related_options:
        lines.append(f"**Related options**: {', '.join(f'``{name}``' for name in doc.related_options)}")
        lines.append("")
    if doc.examples:
        lines.append("**Examples**")
        lines.append("")
        for example in doc.examples:
            lines.append(f"*{example.title}*::")
            lines.append("")
            for code_line in example.code.splitlines():
                lines.append(f"    {code_line}")
            lines.append("")
    if doc.last_reviewed:
        lines.append(f".. note:: Last reviewed {doc.last_reviewed} by {doc.reviewer or 'macrocast author'}.")
        lines.append("")
    return "\n".join(lines)


def _format_option_placeholder(layer: str, sublayer: str, axis: str, option: str, schema_desc: str) -> str:
    lines = [
        _section(f"``{option}``", char="^"),
        "",
        schema_desc or "(schema description not available)",
        "",
        ".. warning:: Detailed OptionDoc not yet registered for this option.",
        "",
    ]
    return "\n".join(lines)


def render_layer(layer_id: str) -> str:
    """Return the full RST page for one layer."""

    info = introspect.layer(layer_id)
    parts: list[str] = []
    parts.append(_section(f"L{layer_id.upper().replace('L', '').replace('_', '.')}  --  {info.name}", char="="))
    parts.append("")
    parts.append(f":Layer ID: ``{layer_id}``")
    parts.append(f":Category: ``{info.category}``")
    parts.append(f":Sub-layers: {', '.join(s.id for s in info.sub_layers)}")
    parts.append("")
    for axis in introspect.axes(layer_id):
        if axis.status != "operational":
            continue
        parts.append(_section(f"axis ``{axis.name}``  ({axis.sublayer})", char="-"))
        parts.append("")
        parts.append(f":Default: ``{axis.default!r}``")
        parts.append(f":Sweepable: {axis.sweepable}")
        if axis.has_gate:
            parts.append(":Gate: this axis is gated by an earlier-layer choice")
        if axis.leaf_config_keys:
            parts.append(f":Leaf config keys: {', '.join(f'``{k}``' for k in axis.leaf_config_keys)}")
        parts.append("")
        for option in axis.options:
            if option.status != "operational":
                continue
            doc = OPTION_DOCS.get((layer_id, axis.sublayer, axis.name, option.value))
            if doc is not None:
                parts.append(_format_option(doc))
            else:
                parts.append(_format_option_placeholder(layer_id, axis.sublayer, axis.name, option.value, option.description))
    return "\n".join(parts)


def render_index() -> str:
    """Return the reference index page."""

    parts: list[str] = [
        _section("Reference", char="="),
        "",
        "Per-layer × per-sub-layer × per-axis × per-option documentation.",
        "Auto-generated from the ``OptionDoc`` registry; every operational",
        "option has at least a Tier-1 entry (summary + description + when-to-use",
        "+ at least one reference) by v1.0.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    for layer_id in introspect.list_layers():
        parts.append(f"   {layer_id}")
    parts.append("")
    return "\n".join(parts)


def write_all(output_dir: str | Path) -> list[Path]:
    """Render every layer + the index into ``output_dir``. Returns the
    list of files written."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index_path = out / "index.rst"
    index_path.write_text(render_index(), encoding="utf-8")
    written.append(index_path)
    for layer_id in introspect.list_layers():
        layer_path = out / f"{layer_id}.rst"
        layer_path.write_text(render_layer(layer_id), encoding="utf-8")
        written.append(layer_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m macrocast.scaffold.render_rst",
        description="Emit reference RST pages from the OptionDoc registry.",
    )
    parser.add_argument(
        "-o", "--output",
        default="docs/source/reference",
        help="Output directory (default: docs/source/reference).",
    )
    args = parser.parse_args(argv)
    written = write_all(args.output)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
