from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

pytestmark = pytest.mark.reference


def _loss_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260601)
    rows = []
    for origin in range(48):
        common = 0.02 * np.sin(origin / 4.0)
        for model, base in (("benchmark", 0.70), ("candidate", 0.42), ("weak", 0.55)):
            rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": model,
                    "squared_error": base + common + rng.normal(0.0, 0.02),
                }
            )
    return pd.DataFrame(rows)


def test_dm_test_is_antisymmetric_reference_anchor() -> None:
    loss_a = pd.Series([0.3, 0.5, 0.4, 0.6, 0.7, 0.4, 0.5, 0.3])
    loss_b = pd.Series([0.5, 0.7, 0.5, 0.8, 0.9, 0.6, 0.7, 0.5])

    ab = mf.tests.dm_test(loss_a, loss_b, correction="none")
    ba = mf.tests.dm_test(loss_b, loss_a, correction="none")

    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
    assert ab.n_obs == ba.n_obs == 8


def test_blocked_reality_check_reference_anchor() -> None:
    result = mf.tests.blocked_oob_reality_check(
        _loss_panel(),
        benchmark="benchmark",
        alpha=0.1,
        n_boot=40,
        block_length=4,
        random_state=123,
    )

    candidate = result.set_index("model").loc["candidate"]
    assert result.attrs["macroforecast_metadata_schema"]["kind"] == "blocked_oob_reality_check"
    assert candidate["mean_diff"] > 0.0
    assert 0.0 <= candidate["p_value"] <= 1.0
    assert bool(candidate["decision"]) is True


def test_iterative_mcs_reference_anchor() -> None:
    result = mf.tests.iterative_model_confidence_set(
        _loss_panel(),
        alpha=0.1,
        n_boot=40,
        block_length=4,
        random_state=123,
    )
    included = result["mcs_inclusion"][0]["models"]
    rejected = result["mcs_rejections"][0]["models"]

    assert result["metadata_schema"]["kind"] == "iterative_model_confidence_set"
    assert "candidate" in included
    assert "benchmark" in rejected
    assert result["iteration_path"][0]["eliminated_model"] == "benchmark"
    assert result["statistic"] == "max"
    assert result["r_reference"] == "MCS/R/MCSprocedure.R::MCSprocedure"
    json.dumps(result)


def test_reporting_output_metadata_reference_anchor(tmp_path) -> None:
    scores = pd.DataFrame(
        {
            "model": ["candidate", "benchmark"],
            "rmse": [0.42, 0.70],
            "r2_oos": [0.12, -0.05],
        }
    )
    table = mf.reporting.report_table(
        scores,
        columns=("model", "rmse", "r2_oos"),
        rename={"model": "Model", "rmse": "RMSE", "r2_oos": "R2 OOS"},
        percent_columns=("R2 OOS",),
        precision=2,
        caption="Reference accuracy table",
    )
    bundle = mf.output.bundle_outputs(
        metadata={"reference_suite": "core"},
        extra={"accuracy_table": table.data},
    )
    manifest = mf.output.write_artifacts(
        mf.output.select_outputs(bundle, objects=("accuracy_table", "metadata")),
        tmp_path,
        formats=("json",),
        include_provenance=False,
    )

    assert table.data.attrs["macroforecast_metadata_schema"]["kind"] == "report_table"
    assert "\\caption{Reference accuracy table}" in table.to_latex()
    assert (tmp_path / "accuracy_table.json").exists()
    assert manifest.records[0].metadata["path_exists"] is True
