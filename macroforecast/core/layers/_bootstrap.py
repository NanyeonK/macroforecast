"""Bootstrap: populate the layer registry with all layer schema classes.

This module is imported for its side effects by macroforecast/core/layers/__init__.py.
It must be imported AFTER macroforecast/core/layers/registry.py is fully loaded.
Do NOT import this module directly in user code or tests.
"""
from __future__ import annotations

from .registry import register_layer

from macroforecast.data import DataSpec
from macroforecast.preprocessing.schema import L2Preprocessing
from macroforecast.features.schema import L3FeatureEngineering
from macroforecast.models.schema import L4ForecastingModel
from macroforecast.evaluation.schema import L5Evaluation
from macroforecast.stat_tests.schema import L6StatisticalTests
from macroforecast.interpretation.schema import L7Interpretation
from macroforecast.output.schema import L8Output
from macroforecast.diagnostics.data_summary.schema import L1_5DataSummary
from macroforecast.diagnostics.preprocessing.schema import L2_5PrePostPreprocessing
from macroforecast.diagnostics.features.schema import L3_5FeatureDiagnostics
from macroforecast.diagnostics.generator.schema import L4_5GeneratorDiagnostics


register_layer(
    id="l1",
    name="Data",
    category="construction",
    produces=("l1_data_definition_v1", "l1_regime_metadata_v1"),
    ui_mode="list",
)(DataSpec)


register_layer(
    id="l2",
    name="Preprocessing",
    category="construction",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l2_clean_panel_v1",),
    ui_mode="list",
)(L2Preprocessing)


register_layer(
    id="l3",
    name="Feature engineering",
    category="construction",
    expected_inputs=("l2_clean_panel_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l3_features_v1", "l3_metadata_v1"),
    ui_mode="graph",
)(L3FeatureEngineering)


register_layer(
    id="l4",
    name="Forecasting model",
    category="construction",
    expected_inputs=("l3_features_v1", "l3_metadata_v1", "l1_regime_metadata_v1"),
    produces=("l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"),
    ui_mode="graph",
)(L4ForecastingModel)


register_layer(
    id="l1_5",
    name="Data summary",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l1_5_diagnostic_v1",),
    ui_mode="list",
)(L1_5DataSummary)


register_layer(
    id="l2_5",
    name="Pre vs post preprocessing",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1", "l2_clean_panel_v1"),
    produces=("l2_5_diagnostic_v1",),
    ui_mode="list",
)(L2_5PrePostPreprocessing)


register_layer(
    id="l3_5",
    name="Feature diagnostics",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1", "l2_clean_panel_v1", "l3_features_v1", "l3_metadata_v1"),
    produces=("l3_5_diagnostic_v1",),
    ui_mode="list",
)(L3_5FeatureDiagnostics)


register_layer(
    id="l4_5",
    name="Generator diagnostics",
    category="diagnostic",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1", "l3_features_v1"),
    produces=("l4_5_diagnostic_v1",),
    ui_mode="list",
)(L4_5GeneratorDiagnostics)


register_layer(
    id="l5",
    name="Evaluation",
    category="consumption",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l1_data_definition_v1", "l1_regime_metadata_v1", "l3_metadata_v1"),
    produces=("l5_evaluation_v1",),
    ui_mode="list",
)(L5Evaluation)


register_layer(
    id="l6",
    name="Statistical tests",
    category="consumption",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l5_evaluation_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l6_tests_v1",),
    ui_mode="list",
)(L6StatisticalTests)


register_layer(
    id="l7",
    name="Interpretation",
    category="consumption",
    expected_inputs=("l4_model_artifacts_v1", "l4_forecasts_v1", "l3_features_v1", "l3_metadata_v1", "l5_evaluation_v1", "l6_tests_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l7_importance_v1", "l7_transformation_attribution_v1"),
    ui_mode="graph",
)(L7Interpretation)


register_layer(
    id="l8",
    name="Output / provenance",
    category="consumption",
    expected_inputs=(
        "l1_data_definition_v1", "l1_regime_metadata_v1", "l2_clean_panel_v1", "l3_features_v1", "l3_metadata_v1",
        "l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1", "l5_evaluation_v1", "l6_tests_v1",
        "l7_importance_v1", "l7_transformation_attribution_v1",
    ),
    produces=("l8_artifacts_v1",),
    ui_mode="list",
)(L8Output)
