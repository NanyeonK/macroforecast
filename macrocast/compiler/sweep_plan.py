"""Horse-race sweep plan compiler.

Expands a recipe dict's ``sweep_axes`` entries (across layers) into a
Cartesian product of concrete single-path variant recipe dicts. Each
variant is fully-specified and compilable through ``compile_recipe_dict``.

Part of Phase 1 (horse-race sweep executor - IDENTITY UNLOCK).
See plans/phases/phase_01_sweep_executor.md section 4.1.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_VARIANTS = 1000
_LAYER2_REPRESENTATION_SWEEP_AXES = {
    "target_transform",
    "target_normalization",
    "horizon_target_construction",
    "target_lag_block",
    "target_lag_selection",
    "x_lag_feature_block",
    "factor_feature_block",
    "feature_selection_policy",
    "feature_selection_semantics",
    "level_feature_block",
    "temporal_feature_block",
    "rotation_feature_block",
    "feature_block_combination",
    "feature_block_set",
}


class SweepPlanError(ValueError):
    """Raised when a recipe dict cannot be expanded into a valid sweep plan."""


@dataclass(frozen=True)
class SweepVariant:
    """One fully-specified variant recipe derived from a parent sweep recipe.

    Field contract: ``variant_id`` is a stable ``v-<8-hex>`` identifier
    derived from the axis values; ``axis_values`` stores layer-qualified
    fixed values such as ``{"3_training.model_family": "ridge"}``;
    ``parent_recipe_id`` preserves the parent recipe identity; and
    ``variant_recipe_dict`` is a standalone recipe dict with ``sweep_axes``
    merged into ``fixed_axes`` for single-path compilation.
    """

    variant_id: str
    axis_values: dict[str, str]
    parent_recipe_id: str
    variant_recipe_dict: dict[str, Any]


@dataclass(frozen=True)
class SweepPlan:
    """A compiled sweep plan: parent + Cartesian-expanded variants."""

    study_id: str
    parent_recipe_id: str
    parent_recipe_dict: dict[str, Any]
    axes_swept: tuple[str, ...]
    variants: tuple[SweepVariant, ...]
    governance: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.variants)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _variant_id(axis_values: dict[str, str]) -> str:
    digest = hashlib.sha256(_canonical_json(axis_values).encode("utf-8")).hexdigest()
    return f"v-{digest[:8]}"


def _study_id(
    parent_recipe_id: str,
    axes_swept: tuple[str, ...],
    variant_axis_values: list[dict[str, str]],
) -> str:
    payload = {
        "parent_recipe_id": parent_recipe_id,
        "axes_swept": list(axes_swept),
        "variants": variant_axis_values,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256-{digest[:16]}"


def _layer_keys(recipe_dict: dict[str, Any]) -> list[str]:
    path = recipe_dict.get("path")
    if not isinstance(path, dict):
        raise SweepPlanError("recipe dict missing 'path' object")
    return sorted(path.keys())


def _collect_sweep_axes(
    recipe_dict: dict[str, Any],
) -> list[tuple[str, str, list[Any]]]:
    triples: list[tuple[str, str, list[Any]]] = []
    for layer in _layer_keys(recipe_dict):
        layer_block = recipe_dict["path"][layer]
        if not isinstance(layer_block, dict):
            continue
        sweep_axes = layer_block.get("sweep_axes") or {}
        if not isinstance(sweep_axes, dict):
            raise SweepPlanError(
                f"layer '{layer}': sweep_axes must be a mapping, got "
                f"{type(sweep_axes).__name__}"
            )
        fixed_axes = layer_block.get("fixed_axes") or {}
        if not isinstance(fixed_axes, dict):
            raise SweepPlanError(
                f"layer '{layer}': fixed_axes must be a mapping, got "
                f"{type(fixed_axes).__name__}"
            )
        for axis_name, values in sorted(sweep_axes.items()):
            if axis_name in fixed_axes:
                raise SweepPlanError(
                    f"layer '{layer}': axis '{axis_name}' appears in both "
                    f"fixed_axes and sweep_axes - pick one"
                )
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise SweepPlanError(
                    f"layer '{layer}': sweep_axes['{axis_name}'] must be a "
                    f"non-empty list of values"
                )
            triples.append((layer, axis_name, list(values)))
    return triples


def _collect_nested_sweep_groups(
    recipe_dict: dict[str, Any],
) -> list[tuple[str, str, list[dict[tuple[str, str], Any]]]]:
    """Collect nested_sweep groups across layers.

    Shape returned: [(layer, parent_axis, variant_list)] where each
    variant_list entry is a dict mapping (layer, axis_name) -> value
    that fully specifies both the parent axis value and the chosen
    child axis value for one nested-sweep fragment.
    """
    groups: list[tuple[str, str, list[dict[tuple[str, str], Any]]]] = []
    for layer in _layer_keys(recipe_dict):
        layer_block = recipe_dict['path'][layer]
        if not isinstance(layer_block, dict):
            continue
        nested = layer_block.get('nested_sweep_axes') or {}
        if not isinstance(nested, dict):
            raise SweepPlanError(
                f"layer '{layer}': nested_sweep_axes must be a mapping, got "
                f"{type(nested).__name__}"
            )
        fixed_axes = layer_block.get('fixed_axes') or {}
        sweep_axes = layer_block.get('sweep_axes') or {}
        for parent_axis, children in sorted(nested.items()):
            if parent_axis in fixed_axes or parent_axis in sweep_axes:
                raise SweepPlanError(
                    f"layer '{layer}': nested_sweep parent axis "
                    f"'{parent_axis}' also appears in fixed_axes or sweep_axes"
                )
            if not isinstance(children, dict) or not children:
                raise SweepPlanError(
                    f"layer '{layer}': nested_sweep_axes['{parent_axis}'] "
                    "must be a non-empty mapping "
                    "{parent_value: {child_axis: [values]}}"
                )
            variant_list: list[dict[tuple[str, str], Any]] = []
            for parent_value, child_spec in sorted(children.items()):
                if not isinstance(child_spec, dict) or len(child_spec) != 1:
                    raise SweepPlanError(
                        f"layer '{layer}': nested_sweep "
                        f"'{parent_axis}.{parent_value}' child_spec must be "
                        "a single-key mapping {child_axis: [values]}"
                    )
                child_axis, child_values = next(iter(child_spec.items()))
                if not isinstance(child_values, (list, tuple)) or not child_values:
                    raise SweepPlanError(
                        f"layer '{layer}': nested_sweep "
                        f"'{parent_axis}.{parent_value}.{child_axis}' values "
                        "must be a non-empty list"
                    )
                for cv in child_values:
                    variant_list.append(
                        {
                            (layer, parent_axis): parent_value,
                            (layer, child_axis): cv,
                        }
                    )
            groups.append((layer, parent_axis, variant_list))
    return groups


def _materialise_variant(
    parent_recipe_dict: dict[str, Any],
    picks: dict[tuple[str, str], Any],
    variant_id: str,
) -> dict[str, Any]:
    variant = copy.deepcopy(parent_recipe_dict)
    parent_id = variant.get("recipe_id", "recipe")
    variant["recipe_id"] = f"{parent_id}#{variant_id}"

    for (layer, axis_name), value in picks.items():
        layer_block = variant["path"][layer]
        sweep_axes = layer_block.get("sweep_axes") or {}
        if axis_name in sweep_axes:
            new_sweep_axes = {k: v for k, v in sweep_axes.items() if k != axis_name}
            if new_sweep_axes:
                layer_block["sweep_axes"] = new_sweep_axes
            else:
                layer_block.pop("sweep_axes", None)
        fixed_axes = dict(layer_block.get("fixed_axes") or {})
        fixed_axes[axis_name] = value
        layer_block["fixed_axes"] = fixed_axes

    # Nested sweep blocks have been fully resolved into picks; clear them.
    for layer_block in variant.get("path", {}).values():
        if isinstance(layer_block, dict):
            layer_block.pop("nested_sweep_axes", None)

    return variant


def _sweep_governance(
    axes_swept: tuple[str, ...],
    *,
    max_variants: int | None,
    total_variants: int,
) -> dict[str, Any]:
    layer2_axes = tuple(axis for axis in axes_swept if axis.startswith("2_preprocessing."))
    layer2_representation_axes = tuple(
        axis
        for axis in layer2_axes
        if axis.rsplit(".", 1)[-1] in _LAYER2_REPRESENTATION_SWEEP_AXES
    )
    model_axes = tuple(axis for axis in axes_swept if axis.endswith(".model_family"))
    return {
        "schema_version": "sweep_governance_v1",
        "expansion_policy": "cartesian_expand_all_then_compile_each_variant",
        "invalid_combination_policy": "materialize_then_gate_at_variant_compile_or_execute",
        "axes_swept": list(axes_swept),
        "layer2_axes": list(layer2_axes),
        "layer2_representation_axes": list(layer2_representation_axes),
        "model_axes": list(model_axes),
        "co_sweeps_model_and_layer2": bool(model_axes and layer2_representation_axes),
        "max_variants": max_variants,
        "variant_count": total_variants,
        "public_api_note": (
            "The sweep planner may expand all requested Layer 2 representation combinations; "
            "public callers should still surface per-variant compile status because some "
            "combinations are intentionally gated by Layer 2/3 runtime contracts."
        ),
    }


def compile_sweep_plan(
    recipe_dict: dict[str, Any],
    *,
    max_variants: int | None = DEFAULT_MAX_VARIANTS,
) -> SweepPlan:
    """Expand ``sweep_axes`` across layers into a Cartesian SweepPlan.

    Args:
        recipe_dict: Parent recipe dict containing one or more
            ``path.<layer>.sweep_axes`` entries.
        max_variants: Upper bound on generated variants, default 1000. Pass
            ``None`` to disable (not recommended for user-supplied recipes).

    Returns:
        A :class:`SweepPlan` whose ``variants`` can each be handed to
        :func:`macrocast.compile_recipe_dict` as a standalone single-path
        recipe.

    Raises:
        SweepPlanError: if the recipe dict has no sweep_axes, if a sweep
            axis duplicates a fixed axis on the same layer, or if the
            Cartesian size exceeds ``max_variants``.
    """

    if not isinstance(recipe_dict, dict):
        raise SweepPlanError("recipe_dict must be a mapping")
    parent_recipe_id = recipe_dict.get("recipe_id", "recipe")

    sweep_triples = _collect_sweep_axes(recipe_dict)
    nested_groups = _collect_nested_sweep_groups(recipe_dict)
    if not sweep_triples and not nested_groups:
        raise SweepPlanError(
            "recipe dict has no sweep_axes or nested_sweep_axes - use "
            "compile_recipe_dict for single-path recipes"
        )

    axes_swept_parts: list[str] = [f"{layer}.{axis}" for layer, axis, _ in sweep_triples]
    for layer, parent_axis, variant_list in nested_groups:
        axes_swept_parts.append(f"{layer}.{parent_axis}")
        child_axes_seen: set[tuple[str, str]] = set()
        for frag in variant_list:
            for key in frag:
                if key != (layer, parent_axis):
                    child_axes_seen.add(key)
        for (l, a) in sorted(child_axes_seen):
            axes_swept_parts.append(f"{l}.{a}")
    axes_swept = tuple(axes_swept_parts)

    total = 1
    for _, _, values in sweep_triples:
        total *= len(values)
    for _, _, variant_list in nested_groups:
        total *= len(variant_list)
    if max_variants is not None and total > max_variants:
        raise SweepPlanError(
            f"sweep would produce {total} variants, exceeds max_variants="
            f"{max_variants}. Narrow sweep_axes or raise max_variants."
        )

    variants: list[SweepVariant] = []
    variant_axis_values: list[dict[str, str]] = []
    seen_variant_ids: set[str] = set()

    sweep_combos_iter = (
        itertools.product(*[values for _, _, values in sweep_triples])
        if sweep_triples
        else [()]
    )
    nested_combos_list = (
        list(itertools.product(*[variant_list for _, _, variant_list in nested_groups]))
        if nested_groups
        else [()]
    )
    for sweep_combo in sweep_combos_iter:
        for nested_combo in nested_combos_list:
            picks: dict[tuple[str, str], Any] = {}
            axis_values: dict[str, str] = {}
            for (layer, axis_name, _values), value in zip(sweep_triples, sweep_combo):
                picks[(layer, axis_name)] = value
                axis_values[f"{layer}.{axis_name}"] = value
            for frag in nested_combo:
                for (layer, axis_name), value in frag.items():
                    picks[(layer, axis_name)] = value
                    axis_values[f"{layer}.{axis_name}"] = value

            vid = _variant_id(axis_values)
            if vid in seen_variant_ids:
                raise SweepPlanError(
                    f"variant_id collision for axis_values={axis_values}; this "
                    "should be impossible for distinct sweep combinations"
                )
            seen_variant_ids.add(vid)

            variant_dict = _materialise_variant(recipe_dict, picks, vid)
            variants.append(
                SweepVariant(
                    variant_id=vid,
                    axis_values=dict(axis_values),
                    parent_recipe_id=parent_recipe_id,
                    variant_recipe_dict=variant_dict,
                )
            )
            variant_axis_values.append(dict(axis_values))

    study_id = _study_id(parent_recipe_id, axes_swept, variant_axis_values)

    return SweepPlan(
        study_id=study_id,
        parent_recipe_id=parent_recipe_id,
        parent_recipe_dict=copy.deepcopy(recipe_dict),
        axes_swept=axes_swept,
        variants=tuple(variants),
        governance=_sweep_governance(
            axes_swept,
            max_variants=max_variants,
            total_variants=total,
        ),
    )


__all__ = [
    "SweepPlan",
    "SweepVariant",
    "SweepPlanError",
    "compile_sweep_plan",
    "DEFAULT_MAX_VARIANTS",
]
