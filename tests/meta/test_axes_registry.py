from macrocast.meta import load_axes_registry, validate_axes_registry
from macrocast.meta.exceptions import AxisClassificationError


def test_axes_registry_valid() -> None:
    reg = load_axes_registry()
    validate_axes_registry(reg)
    assert reg['unit_of_run']['experiment'].startswith('one_target')
    assert 'dataset' in reg['axis_classes']['experiment_fixed']


def test_axes_registry_detects_duplicate_axis() -> None:
    reg = load_axes_registry()
    reg['axis_classes']['research_sweep'] = list(reg['axis_classes']['research_sweep']) + ['dataset']
    try:
        validate_axes_registry(reg)
    except AxisClassificationError:
        return
    raise AssertionError('expected AxisClassificationError')


def test_axes_registry_rejects_legacy_mirror_drift() -> None:
    reg = load_axes_registry()
    reg['invariant_axes'] = list(reg['invariant_axes']) + ['fake_axis']
    try:
        validate_axes_registry(reg)
    except AxisClassificationError:
        return
    raise AssertionError('expected AxisClassificationError')
