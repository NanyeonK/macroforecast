"""CLI wizard -- gate-following recipe authoring.

Prompts the user layer-by-layer for fixed-axis values, expands the
``OptionDoc`` registry on ``?``, validates after each layer, and writes
a recipe YAML at the end. Intentionally stdlib-only -- no rich /
textual dependency.

Entry point: ``python -m macrocast scaffold`` (see ``cli.py``).
"""
from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import introspect
from .builder import RecipeBuilder
from .option_docs import OPTION_DOCS, OptionDoc


# ---------------------------------------------------------------------------
# Output helpers (no third-party deps)
# ---------------------------------------------------------------------------

_RULE = "─" * 72


def _print(*args: Any) -> None:
    print(*args, flush=True)


def _heading(text: str) -> None:
    _print()
    _print(text)
    _print(_RULE)


def _wrap(text: str, *, indent: int = 2, width: int = 70) -> str:
    return textwrap.indent(textwrap.fill(text, width=width), " " * indent)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

@dataclass
class _PromptResult:
    value: str
    help_shown: bool


def _prompt_axis(
    axis: introspect.AxisInfo,
    *,
    input_fn: Callable[[str], str] = input,
) -> _PromptResult:
    """Show axis options, accept a value or ``?`` for full doc.

    Loops until the user supplies a valid choice. The default value is
    accepted on empty input.
    """

    options = list(axis.options)
    default = axis.default
    summary_lines = [
        f"  [{idx + 1}] {opt.value:30s} {opt.label or opt.description[:60]}"
        for idx, opt in enumerate(options)
    ]
    while True:
        _print()
        _print(f"{axis.sublayer}  {axis.name}  [default = {default!r}]")
        if summary_lines:
            for line in summary_lines:
                _print(line)
        prompt = "  Choose by number or value, ? for details, [enter] for default: "
        raw = input_fn(prompt).strip()
        if raw == "":
            return _PromptResult(value=str(default), help_shown=False)
        if raw == "?":
            _show_axis_help(axis)
            continue
        if raw.startswith("?"):
            target = raw[1:].strip()
            _show_option_help(axis, target, options)
            continue
        # Numeric choice.
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return _PromptResult(value=options[idx].value, help_shown=False)
            _print(f"  ! invalid number {raw!r}; try 1..{len(options)} or a value name.")
            continue
        # Match by value name.
        match = next((opt for opt in options if opt.value == raw), None)
        if match is not None:
            return _PromptResult(value=match.value, help_shown=False)
        # Free-form input (axis with non-enumerated values, e.g., target).
        if not options:
            return _PromptResult(value=raw, help_shown=False)
        _print(
            f"  ! {raw!r} is not one of the offered options; "
            f"type ? for help or pick from {[o.value for o in options]}."
        )


def _show_axis_help(axis: introspect.AxisInfo) -> None:
    _print()
    _print(_RULE)
    _print(f"axis: {axis.layer}.{axis.sublayer}.{axis.name}")
    _print(f"default: {axis.default!r}")
    _print(f"sweepable: {axis.sweepable}")
    if axis.has_gate:
        _print("(this axis has a gate predicate -- earlier choices may make it inactive)")
    _print()
    if not axis.options:
        _print("(this axis accepts free-form input; no enumerated options)")
        _print(_RULE)
        return
    _print("options:")
    for opt in axis.options:
        _print(f"  - {opt.value:30s} {opt.label or opt.description}")
    _print()
    _print(f"Type '? <option_value>' to see the full documentation for that option.")
    _print(_RULE)


def _show_option_help(axis: introspect.AxisInfo, target: str, options: Iterable[introspect.OptionInfo]) -> None:
    options_list = list(options)
    match = next((o for o in options_list if o.value == target), None)
    if match is None and target.isdigit():
        idx = int(target) - 1
        if 0 <= idx < len(options_list):
            match = options_list[idx]
    if match is None:
        _print(f"  ! no option named {target!r}; try one of {[o.value for o in options_list]}.")
        return
    doc = OPTION_DOCS.get((axis.layer, axis.sublayer, axis.name, match.value))
    _print()
    _print(_RULE)
    _print(f"option [{match.value}] -- {axis.layer}.{axis.sublayer}.{axis.name}")
    _print()
    if doc is None:
        _print(_wrap(match.description or "(no detailed documentation registered yet)"))
        _print()
        _print("  ! Tier-1 OptionDoc not yet registered for this option. The wizard")
        _print("    will surface the schema's short description instead.")
    else:
        _print(_wrap(doc.summary))
        _print()
        _print("Description:")
        for paragraph in doc.description.split("\n\n"):
            _print(_wrap(paragraph))
            _print()
        _print("When to use:")
        _print(_wrap(doc.when_to_use))
        if doc.when_not_to_use:
            _print()
            _print("When NOT to use:")
            _print(_wrap(doc.when_not_to_use))
        if doc.references:
            _print()
            _print("References:")
            for ref in doc.references:
                _print(f"  - {ref.to_rst()}")
        if doc.related_options:
            _print()
            _print(f"Related options: {', '.join(doc.related_options)}")
        if doc.last_reviewed:
            _print()
            _print(f"Last reviewed: {doc.last_reviewed} by {doc.reviewer or 'macrocast author'}")
    _print(_RULE)


# ---------------------------------------------------------------------------
# Wizard flow
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "1": "single_model_forecast",
    "2": "model_horse_race",
    "3": "regime_conditional",
    "4": "blank",
}


def _intro(input_fn: Callable[[str], str]) -> str:
    _heading("macrocast — recipe scaffold wizard")
    _print()
    _print("What kind of study?")
    _print("  [1] Single-model forecast (default)")
    _print("  [2] Model horse race")
    _print("  [3] Regime-conditional forecast")
    _print("  [4] Blank (advanced -- start from scratch)")
    raw = input_fn("Choice [1]: ").strip() or "1"
    return _TEMPLATES.get(raw, "single_model_forecast")


_LAYER_KEYS: dict[str, str] = {
    "l0": "0_meta",
    "l1": "1_data",
    "l1_5": "1_5_data_diagnostics",
    "l2": "2_preprocessing",
    "l2_5": "2_5_preprocessing_diagnostics",
    "l3": "3_feature_engineering",
    "l3_5": "3_5_feature_diagnostics",
    "l4": "4_forecasting_model",
    "l4_5": "4_5_model_diagnostics",
    "l5": "5_evaluation",
    "l6": "6_statistical_tests",
    "l7": "7_interpretation",
    "l8": "8_output",
}


def _layer_block(builder: RecipeBuilder, layer_id: str) -> dict[str, Any]:
    """Return the recipe block dict for ``layer_id``.

    Main layers (L0-L8) use the builder's per-layer namespace so its
    presets stay consistent. Diagnostic layers (L1.5/L2.5/L3.5/L4.5)
    have no builder namespace; we write directly into the underlying
    recipe dict at the canonical key.
    """

    namespace = getattr(builder, layer_id, None)
    if namespace is not None:
        return namespace.block
    key = _LAYER_KEYS[layer_id]
    return builder._recipe.setdefault(key, {})


def _walk_layer(builder: RecipeBuilder, layer_id: str, *, input_fn: Callable[[str], str]) -> None:
    layer = introspect.layer(layer_id)
    _heading(f"{layer.id.upper()}  {layer.name}")
    block = _layer_block(builder, layer_id)
    fixed_axes = block.setdefault("fixed_axes", {})
    leaf_config = block.setdefault("leaf_config", {})
    for axis in introspect.axes(layer_id):
        if axis.status != "operational":
            continue
        result = _prompt_axis(axis, input_fn=input_fn)
        fixed_axes[axis.name] = _coerce_value(result.value, axis.default)
        for leaf_key in axis.leaf_config_keys:
            existing = leaf_config.get(leaf_key)
            prompt = f"  leaf_config.{leaf_key} [default = {existing!r}]: "
            raw = input_fn(prompt).strip()
            if raw:
                leaf_config[leaf_key] = raw


def _coerce_value(raw: str, default: Any) -> Any:
    """Match the type of the default when possible -- keeps int defaults
    int after a numeric input, etc."""

    if isinstance(default, bool):
        return raw.lower() in {"true", "yes", "y", "1"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _post_layer_validate(builder: RecipeBuilder) -> None:
    errors = builder.validate()
    if not errors:
        _print()
        _print("  ✓ partial recipe validates OK so far.")
        return
    _print()
    _print("  ! Validation errors:")
    for err in errors:
        _print(f"    - {err}")


_DEFAULT_INTERACTIVE_LAYERS: tuple[str, ...] = (
    "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8",
)
"""Main layers walked by default. Diagnostic layers (L1.5 / L2.5 / L3.5 /
L4.5) are opt-in via ``include_diagnostics=True``."""


def run_wizard(
    output_path: str | Path = "recipe.yaml",
    *,
    input_fn: Callable[[str], str] = input,
    interactive_layers: tuple[str, ...] | None = None,
    include_diagnostics: bool = False,
) -> RecipeBuilder:
    """Walk the user through the gate-following recipe authoring flow.

    By default v1.0 steps through every main layer (L0..L8). Pass
    ``include_diagnostics=True`` to additionally walk the four
    diagnostic layers (L1.5, L2.5, L3.5, L4.5). For backwards-compat
    callers can still pass an explicit ``interactive_layers`` tuple to
    restrict the walk (e.g. ``("l0",)`` for the smoke test).
    """

    if interactive_layers is None:
        interactive_layers = _DEFAULT_INTERACTIVE_LAYERS
        if include_diagnostics:
            interactive_layers = (
                "l0", "l1", "l1_5",
                "l2", "l2_5",
                "l3", "l3_5",
                "l4", "l4_5",
                "l5", "l6", "l7", "l8",
            )
    template = _intro(input_fn)
    builder = RecipeBuilder()
    if template != "blank":
        # Pre-fill defaults so the user can skip layers without rebuilding
        # from scratch.
        builder.l0()
        builder.l2.no_op()
        builder.l3.lag_only(n_lag=1)
        builder.l5.standard()
    for layer_id in interactive_layers:
        _walk_layer(builder, layer_id, input_fn=input_fn)
    _post_layer_validate(builder)
    output_path = Path(output_path)
    builder.to_yaml(output_path)
    _print()
    _print(f"  ✓ Wrote {output_path} ({output_path.stat().st_size} bytes).")
    return builder


__all__ = ["run_wizard"]
