"""Custom FRED-SD mixed-frequency model example.

Run from the repository root after installing macrocast in editable mode:

    python examples/custom_fred_sd_mixed_frequency_model.py

The registered name can also be selected from a YAML recipe, but the Python
module that registers the callable must be imported before the recipe is run.
"""

from __future__ import annotations

import numpy as np

import macrocast as mc


def _as_2d_float(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


@mc.custom_model(
    "my_fred_sd_mixed_frequency_model",
    description="Example custom model that consumes FRED-SD native-frequency payloads.",
)
def my_fred_sd_mixed_frequency_model(X_train, y_train, X_test, context):
    payloads = context.get("auxiliary_payloads", {})
    block_payload = payloads.get("fred_sd_native_frequency_block_payload", {})
    column_to_frequency = block_payload.get("column_to_native_frequency", {})
    monthly_columns = [
        column for column, frequency in column_to_frequency.items() if frequency == "monthly"
    ]
    quarterly_columns = [
        column for column, frequency in column_to_frequency.items() if frequency == "quarterly"
    ]

    X = _as_2d_float(X_train)
    x_next = _as_2d_float(X_test)
    y = np.asarray(y_train, dtype=float).reshape(-1)

    # Minimal ridge baseline. Researchers normally replace this block with
    # their own mixed-frequency likelihood, weighting scheme, or state update.
    penalty = 1.0 + 0.1 * len(quarterly_columns) + 0.0 * len(monthly_columns)
    design = np.column_stack([np.ones(X.shape[0]), X])
    pred_design = np.column_stack([np.ones(x_next.shape[0]), x_next])
    gram = design.T @ design
    gram += penalty * np.eye(gram.shape[0])
    gram[0, 0] -= penalty
    coef = np.linalg.solve(gram, design.T @ y)
    prediction = pred_design @ coef

    return float(prediction[0])


def build_experiment() -> mc.Experiment:
    return (
        mc.Experiment(
            dataset="fred_sd",
            target="UR_CA",
            start="2000-01",
            end="2020-12",
            horizons=[1],
            frequency="monthly",
            feature_builder="raw_feature_panel",
            model_family="my_fred_sd_mixed_frequency_model",
            benchmark_config={"minimum_train_size": 5},
        )
        .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
        .use_fred_sd_mixed_frequency_adapter()
    )


if __name__ == "__main__":
    result = build_experiment().run(
        local_raw_source="tests/fixtures/fred_sd_sample.csv",
        output_root="results/fred_sd_custom_mixed_frequency",
    )
    print(result.artifact_dir)
