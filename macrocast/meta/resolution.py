from __future__ import annotations

from copy import deepcopy
from typing import Any

from macrocast.meta.validators import check_override_legality


def resolve_preset(preset_registry: dict[str, Any], preset_id: str) -> dict[str, Any]:
    for preset in preset_registry.get('presets', []):
        if preset['id'] == preset_id:
            return deepcopy(preset)
    raise KeyError(f'unknown preset_id: {preset_id}')


def _merge_layer(base: dict[str, Any], defaults: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    if defaults:
        out.update(deepcopy(defaults))
    return out


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any], axes_registry: dict[str, Any], *, stage: str) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overrides.items():
        check_override_legality(key, axes_registry, stage=stage)
        out[key] = value
    return out


def resolve_meta_config(
    *,
    preset_registry: dict[str, Any],
    axes_registry: dict[str, Any],
    preset_id: str | None = None,
    global_defaults: dict[str, Any] | None = None,
    dataset_defaults: dict[str, Any] | None = None,
    target_defaults: dict[str, Any] | None = None,
    model_defaults: dict[str, Any] | None = None,
    experiment_overrides: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
    cell_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved: dict[str, Any] = deepcopy(global_defaults or {})
    resolved = _merge_layer(resolved, dataset_defaults)
    resolved = _merge_layer(resolved, target_defaults)
    resolved = _merge_layer(resolved, model_defaults)
    if preset_id is not None:
        preset = resolve_preset(preset_registry, preset_id)
        resolved = apply_overrides(resolved, preset.get('base_defaults', {}), axes_registry, stage='experiment')
        resolved['_preset_id'] = preset_id
    if experiment_overrides:
        resolved = apply_overrides(resolved, experiment_overrides, axes_registry, stage='experiment')
    if run_overrides:
        resolved = apply_overrides(resolved, run_overrides, axes_registry, stage='run')
    if cell_overrides:
        resolved = apply_overrides(resolved, cell_overrides, axes_registry, stage='cell')
    return resolved
