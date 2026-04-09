from macrocast.meta import load_axes_registry, load_preset_registry, validate_preset_registry
from macrocast.meta.exceptions import PresetResolutionError


def _known_axes() -> set[str]:
    axes = load_axes_registry()['axis_classes']
    return set(axes['invariant']) | set(axes['experiment_fixed']) | set(axes['research_sweep']) | set(axes['conditional'])


def test_preset_registry_valid() -> None:
    axes = load_axes_registry()
    reg = load_preset_registry()
    validate_preset_registry(reg, invariant_axes=set(axes['invariant_axes']), known_axes=_known_axes())


def test_preset_registry_rejects_invariant_override() -> None:
    axes = load_axes_registry()
    reg = load_preset_registry()
    reg['presets'][0]['allowed_overrides'] = list(reg['presets'][0]['allowed_overrides']) + ['no_lookahead_rule']
    try:
        validate_preset_registry(reg, invariant_axes=set(axes['invariant_axes']), known_axes=_known_axes())
    except PresetResolutionError:
        return
    raise AssertionError('expected PresetResolutionError')


def test_preset_registry_rejects_unknown_override() -> None:
    axes = load_axes_registry()
    reg = load_preset_registry()
    reg['presets'][0]['allowed_overrides'] = list(reg['presets'][0]['allowed_overrides']) + ['dataset_id']
    try:
        validate_preset_registry(reg, invariant_axes=set(axes['invariant_axes']), known_axes=_known_axes())
    except PresetResolutionError:
        return
    raise AssertionError('expected PresetResolutionError')
