from __future__ import annotations

from .registry import register_op
from ..types import DiagnosticArtifact


def _stub(name: str):
    def run(inputs, params):
        raise NotImplementedError(f"Phase 1 runtime: {name} implementation in execution PR")

    return run


for _name, _scope in (
    ("diagnostic_collect_l1", ("l1_5",)),
    ("l1_5_sample_coverage", ("l1_5",)),
    ("l1_5_univariate_summary", ("l1_5",)),
    ("l1_5_stationarity_tests", ("l1_5",)),
    ("l1_5_missing_outlier_audit", ("l1_5",)),
    ("l1_5_correlation_pre_cleaning", ("l1_5",)),
    ("l1_5_diagnostic_export", ("l1_5",)),
    ("diagnostic_collect_l2", ("l2_5",)),
    ("l2_5_comparison_axis", ("l2_5",)),
    ("l2_5_distribution_shift", ("l2_5",)),
    ("l2_5_correlation_shift", ("l2_5",)),
    ("l2_5_cleaning_effect_summary", ("l2_5",)),
    ("l2_5_diagnostic_export", ("l2_5",)),
    ("diagnostic_collect_l3", ("l3_5",)),
    ("l3_5_comparison_axis", ("l3_5",)),
    ("l3_5_factor_block_inspection", ("l3_5",)),
    ("l3_5_feature_correlation", ("l3_5",)),
    ("l3_5_lag_block_inspection", ("l3_5",)),
    ("l3_5_selected_features", ("l3_5",)),
    ("l3_5_diagnostic_export", ("l3_5",)),
    ("diagnostic_collect_l4", ("l4_5",)),
    ("l4_5_in_sample_fit", ("l4_5",)),
    ("l4_5_forecast_scale_view", ("l4_5",)),
    ("l4_5_window_stability", ("l4_5",)),
    ("l4_5_tuning_history", ("l4_5",)),
    ("l4_5_ensemble_diagnostics", ("l4_5",)),
    ("l4_5_diagnostic_export", ("l4_5",)),
):
    register_op(name=_name, layer_scope=_scope, input_types={"default": object}, output_type=DiagnosticArtifact)(_stub(_name))
