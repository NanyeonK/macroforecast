from __future__ import annotations

from typing import Any

from macrocast.registries.loaders import REGISTRY_LAYERS

REQUIRED_LAYER_FILES = {
    'meta': {'global_defaults'},
    'data': {'datasets'},
    'training': {'models'},
    'evaluation': {'metric_suites'},
    'output': {'output_registry'},
}


def validate_registry_layer(layer: str, bundle: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if layer not in REQUIRED_LAYER_FILES:
        raise ValueError(f'unknown registry layer: {layer}')
    missing = REQUIRED_LAYER_FILES[layer] - set(bundle)
    if missing:
        raise ValueError(f'{layer} missing registry files: {sorted(missing)}')
    for name, content in bundle.items():
        if not isinstance(content, dict) or not content:
            raise ValueError(f'{layer}/{name} must be non-empty dict root')
        reg = content.get('registry')
        if not isinstance(reg, dict):
            raise ValueError(f'{layer}/{name} missing registry metadata block')
        for key in ['id', 'source', 'role']:
            if key not in reg:
                raise ValueError(f'{layer}/{name} registry metadata missing key: {key}')
    return bundle


def validate_registry_bundle(bundle: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    missing_layers = set(REGISTRY_LAYERS) - set(bundle)
    if missing_layers:
        raise ValueError(f'missing registry layers: {sorted(missing_layers)}')
    for layer in REGISTRY_LAYERS:
        validate_registry_layer(layer, bundle[layer])
    return bundle
