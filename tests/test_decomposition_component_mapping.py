"""Every AxisDefinition must carry a valid component (or None)."""
from __future__ import annotations

from macrocast.decomposition.components import is_valid_component
from macrocast.registry.build import _discover_axis_definitions


def test_every_axis_has_valid_component_or_none():
    definitions = _discover_axis_definitions()
    for axis_name, defn in definitions.items():
        assert is_valid_component(defn.component), (
            f"axis {axis_name!r} has invalid component {defn.component!r}"
        )


def test_phase_7_concrete_mappings_are_set():
    definitions = _discover_axis_definitions()
    expected = {
        "scaling_policy": "preprocessing",
        "dimensionality_reduction_policy": "preprocessing",
        "feature_selection_policy": "preprocessing",
        "target_transform_policy": "preprocessing",
        "x_transform_policy": "preprocessing",
        "tcode_policy": "preprocessing",
        "model_family": "nonlinearity",
        "feature_builder": "feature_representation",
        "predictor_family": "feature_representation",
        "data_richness_mode": "feature_representation",
        "factor_count": "feature_representation",
        "feature_block_set": "feature_representation",
        "target_lag_block": "feature_representation",
        "x_lag_feature_block": "feature_representation",
        "factor_feature_block": "feature_representation",
        "level_feature_block": "feature_representation",
        "temporal_feature_block": "feature_representation",
        "rotation_feature_block": "feature_representation",
        "benchmark_family": "benchmark",
        "importance_method": "importance",
    }
    for axis, comp in expected.items():
        assert axis in definitions, f"axis {axis!r} not registered"
        assert definitions[axis].component == comp, (
            f"axis {axis!r} component={definitions[axis].component!r} != {comp!r}"
        )


def test_unmapped_components_have_no_axes_yet():
    """Phase 7 reserves 3 component slots with no axes: regularization, cv_scheme, loss."""
    definitions = _discover_axis_definitions()
    for unmapped in ("regularization", "cv_scheme", "loss"):
        axes_for_comp = [n for n, d in definitions.items() if d.component == unmapped]
        assert axes_for_comp == [], (
            f"component {unmapped!r} was not supposed to have axes yet; found {axes_for_comp}"
        )
