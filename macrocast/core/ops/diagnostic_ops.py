from __future__ import annotations

from .registry import register_op
from ..types import DiagnosticArtifact


def _passthrough(name: str):
    """Diagnostic ops collect upstream payloads into a DiagnosticArtifact.

    The actual diagnostic computations live in
    :mod:`macrocast.core.runtime` (e.g. ``materialize_l1_5_diagnostic``); this
    op is the no-op DAG entry that records inputs + params so the cache can
    materialize the sink without raising.
    """

    def run(inputs, params):
        return DiagnosticArtifact(
            layer_hooked=name,
            artifact_type="json",
            metadata={"inputs": list(inputs) if isinstance(inputs, list) else [inputs], "params": dict(params)},
            enabled=True,
        )

    run.__name__ = name
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
    register_op(name=_name, layer_scope=_scope, input_types={"default": object}, output_type=DiagnosticArtifact)(_passthrough(_name))
