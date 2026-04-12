from __future__ import annotations

from typing import Any

from macrocast.taxonomy.loaders import TAXONOMY_LAYERS


REQUIRED_LAYER_FILES = {
    '0_meta': {'path_grammar', 'experiment_unit', 'axis_type', 'registry_type', 'reproducibility_mode', 'failure_policy', 'compute_mode'},
    '1_data': {'source', 'frequency', 'info_set', 'alignment', 'vintage', 'release_lag', 'ragged_edge', 'variable_universe', 'sample_period'},
    '2_target_x': {'target_family', 'predictor_family', 'contemporaneous_x', 'own_target_lags', 'x_map_policy', 'target_transform', 'target_scale'},
    '3_preprocess': {'recipe_registry', 'target_missing', 'target_outlier', 'x_missing', 'x_outlier', 'x_scale', 'x_reduce', 'execution_order'},
    '4_training': {'framework', 'split', 'model_registry', 'tuning_registry', 'feature_registry'},
    '5_evaluation': {'metric_registry', 'benchmark_registry', 'regime_registry', 'decomposition_registry', 'aggregation_registry'},
    '6_stat_tests': {'test_registry', 'dependence_correction'},
    '7_importance': {'importance_registry', 'grouping_registry', 'plotting_registry'},
    '8_output_provenance': {'artifact_registry', 'provenance_registry', 'export_format'},
}


def validate_taxonomy_layer(layer: str, bundle: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if layer not in REQUIRED_LAYER_FILES:
        raise ValueError(f'unknown taxonomy layer: {layer}')
    missing = REQUIRED_LAYER_FILES[layer] - set(bundle)
    if missing:
        raise ValueError(f'{layer} missing taxonomy files: {sorted(missing)}')
    for name, content in bundle.items():
        if not isinstance(content, dict) or not content:
            raise ValueError(f'{layer}/{name} must be non-empty dict root')
    return bundle


def validate_taxonomy_bundle(bundle: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    missing_layers = set(TAXONOMY_LAYERS) - set(bundle)
    if missing_layers:
        raise ValueError(f'missing taxonomy layers: {sorted(missing_layers)}')
    for layer in TAXONOMY_LAYERS:
        validate_taxonomy_layer(layer, bundle[layer])
    return bundle
