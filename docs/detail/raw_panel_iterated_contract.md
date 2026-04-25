# Raw-Panel Iterated Forecasting Contract

Raw-panel iterated forecasting is a Layer 3 execution protocol over a Layer 2
tabular raw-panel representation. It is not opened by the existing
`Layer2Representation` handoff alone, because each step after the origin needs
a policy for future predictors.

## Boundary

Layer ownership:

- Layer 1 owns observed data timing, release lags, vintages, and raw-source
  availability.
- Layer 2 owns the representation builder that maps available data into `Z`.
- Layer 3 owns the iterated forecast protocol, recursive state updates, and
  typed multi-step forecast payload.

Required contracts:

- `exogenous_x_path_contract_v1`;
- `multi_step_raw_panel_payload_v1`.

## Exogenous-X Path Contract

`exogenous_x_path_contract_v1` must be explicit about what the generator is
allowed to know or assume after the forecast origin.

Allowed `path_kind` values:

- `observed_future_x`: future X is supplied as observed future information and
  must be marked as such in provenance.
- `scheduled_known_future_x`: future X is known by schedule or release rule at
  the origin.
- `hold_last_observed`: future X is deterministically held at the latest
  origin-available value.
- `recursive_x_model`: future X is forecast by a separate model.
- `unavailable`: no valid future-X path exists, so the cell must remain
  blocked.

Required fields:

- `path_kind`;
- `origin_index`;
- `horizon_steps`;
- `predictor_names`;
- `x_path_frame_or_assumption`;
- `availability_mask`;
- `vintage_cutoff`;
- `release_lag_policy`;
- `no_lookahead_evidence`.

The first operational slice is `hold_last_observed`. It is deterministic
and honest: it records the scenario assumption instead of silently using future
observed X. Built-in scalar tabular generators and registered
`custom_model_v1` models can consume this slice; the custom model receives the
same raw-X-plus-recursive-target-lag representation context at each recursive
step.

## Multi-Step Payload Contract

`multi_step_raw_panel_payload_v1` must make every recursive step auditable.

Required fields:

- `origin_index`;
- `horizon`;
- `step_predictions`;
- `final_horizon_prediction`;
- `target_history_updates`;
- `exogenous_x_path_ref`;
- `recursive_state_trace`;
- `payload_metrics`.

The artifact writer should project the final horizon prediction into
`predictions.csv` and write the step trace to a separate long-form artifact.
The JSONL payload should preserve the step-level trace.

## Opening Checklist

Before a raw-panel iterated forecasting slice can become operational:

- compiler accepts the explicit future-X scenario;
- compiler rejects raw-panel iterated cells without a future-X contract;
- runtime writes step-level trace artifacts;
- runtime records recursive target-history updates;
- runtime records the future-X path reference or assumption;
- payload JSONL includes the multi-step payload object;
- tests cover no-lookahead, origin alignment, release-lag masks, final
  prediction projection, and failed-cell behavior under `skip_failed_cell`.
