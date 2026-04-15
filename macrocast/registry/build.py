from __future__ import annotations

from functools import lru_cache
import importlib
import pkgutil

from .base import AxisDefinition, axis_definition_to_legacy_entry
from .types import AxisRegistryEntry

CANONICAL_LAYER_ORDER: tuple[str, ...] = ('0_meta', '1_data_task', '2_preprocessing', '3_training', '4_evaluation', '5_output_provenance', '6_stat_tests', '7_importance')
_STAGE_PACKAGES: tuple[str, ...] = ('stage0', 'data', 'preprocessing', 'training', 'evaluation', 'output', 'tests', 'importance')


@lru_cache(maxsize=1)
def _discover_axis_definitions() -> dict[str, AxisDefinition]:
    discovered: dict[str, AxisDefinition] = {}
    for package_name in _STAGE_PACKAGES:
        package = importlib.import_module(f'{__package__}.{package_name}')
        module_infos = sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name)
        for module_info in module_infos:
            if module_info.name.startswith('_'):
                continue
            module = importlib.import_module(f'{package.__name__}.{module_info.name}')
            definition = getattr(module, 'AXIS_DEFINITION', None)
            if definition is None:
                continue
            if not isinstance(definition, AxisDefinition):
                raise TypeError(
                    f'{module.__name__}.AXIS_DEFINITION must be AxisDefinition, got {type(definition).__name__}'
                )
            if definition.axis_name in discovered:
                raise ValueError(f'duplicate axis definition for {definition.axis_name!r}')
            discovered[definition.axis_name] = definition
    return discovered


@lru_cache(maxsize=1)
def _axis_registry() -> dict[str, AxisRegistryEntry]:
    return {
        axis_name: axis_definition_to_legacy_entry(definition)
        for axis_name, definition in _discover_axis_definitions().items()
    }


def get_canonical_layer_order() -> tuple[str, ...]:
    return CANONICAL_LAYER_ORDER


def get_axis_registry() -> dict[str, AxisRegistryEntry]:
    return dict(_axis_registry())


def get_axis_registry_entry(axis_name: str) -> AxisRegistryEntry:
    return _axis_registry()[axis_name]


def axis_governance_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    registry = _axis_registry()
    for axis_name in sorted(registry):
        entry = registry[axis_name]
        rows.append(
            {
                'axis_name': entry.axis_name,
                'layer': entry.layer,
                'axis_type': entry.axis_type,
                'allowed_values': list(entry.allowed_values),
                'current_status': dict(entry.current_status),
                'default_policy': entry.default_policy,
                'compatible_with': {k: list(v) for k, v in entry.compatible_with.items()},
                'incompatible_with': {k: list(v) for k, v in entry.incompatible_with.items()},
            }
        )
    return rows
