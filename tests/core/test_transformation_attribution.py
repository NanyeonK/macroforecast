"""Issue #218 -- Shapley over pipelines / marginal_addition / LOO."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import _l7_transformation_attribution
from macrocast.core.types import L5EvaluationArtifact


def _l5_with_pipelines(losses: list[tuple[str, float]]) -> L5EvaluationArtifact:
    rows = [
        {"model_id": pipeline, "target": "y", "horizon": 1, "mse": loss, "mae": loss * 0.8}
        for pipeline, loss in losses
    ]
    metrics = pd.DataFrame(rows)
    return L5EvaluationArtifact(
        metrics_table=metrics,
        ranking_table=metrics.copy(),
        l5_axis_resolved={},
    )


def test_shapley_over_pipelines_assigns_positive_contribution_to_lower_loss_pipeline():
    eval_artifact = _l5_with_pipelines([("a", 1.0), ("b", 2.0), ("c", 3.0)])
    result = _l7_transformation_attribution(
        eval_artifact, params={"decomposition_method": "shapley_over_pipelines", "loss_function": "mse"}
    )
    contribs = {row.pipeline: row.contribution for row in result.summary_table.itertuples()}
    # Pipeline "a" should rank highest (lowest loss). The Shapley values for
    # negative-mean-loss payoffs are negative (lower loss -> higher / less
    # negative contribution).
    assert contribs["a"] > contribs["b"] > contribs["c"]


def test_leave_one_out_drops_to_zero_when_only_one_pipeline():
    eval_artifact = _l5_with_pipelines([("only", 1.5)])
    result = _l7_transformation_attribution(
        eval_artifact, params={"decomposition_method": "leave_one_out_pipeline", "loss_function": "mse"}
    )
    assert len(result.summary_table) == 1
    # With one pipeline the "without" set is empty -> contribution = -full_loss.
    assert result.summary_table.iloc[0]["contribution"] == -1.5


def test_marginal_addition_uses_worst_pipeline_baseline():
    eval_artifact = _l5_with_pipelines([("a", 1.0), ("b", 2.0), ("c", 4.0)])
    result = _l7_transformation_attribution(
        eval_artifact, params={"decomposition_method": "marginal_addition", "loss_function": "mse"}
    )
    contribs = {row.pipeline: row.contribution for row in result.summary_table.itertuples()}
    # Worst loss = 4.0 -> a contribution = 4-1 = 3, c = 0.
    assert contribs["a"] == 3.0
    assert contribs["c"] == 0.0


def test_decomposition_method_recorded_on_artifact():
    eval_artifact = _l5_with_pipelines([("a", 1.0), ("b", 1.5)])
    result = _l7_transformation_attribution(
        eval_artifact, params={"decomposition_method": "leave_one_out_pipeline"}
    )
    assert result.decomposition_method == "leave_one_out_pipeline"


def test_empty_metrics_returns_empty_artifact():
    empty = L5EvaluationArtifact(
        metrics_table=pd.DataFrame(columns=["model_id", "target", "horizon", "mse"]),
        ranking_table=pd.DataFrame(),
        l5_axis_resolved={},
    )
    result = _l7_transformation_attribution(empty, params={})
    assert result.summary_table.empty
    assert result.pipeline_contributions == {}
