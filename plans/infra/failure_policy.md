# Infra: Failure Policy

## 1. What

Runtime semantics for the 8 `failure_policy` values that the recipe registry already
accepts but does not yet wire into execution. This is a **P0 gap**: today the registry
validates the string but `execute_recipe` and `SweepRunner` have no branching on it,
so a failed cell always bubbles as an uncaught exception. This infra file is the
single source of truth for the 8 semantics and the handler contract that every
execution layer (single-path, sweep, ablation, tuning) must call into. Once wired,
the same policy string produces identical behaviour regardless of which engine is
driving the run.

## 2. Used by phases

- phase-01 — SweepRunner error handling and per-variant isolation
- phase-06 — ablation and replication failure handling (same handler)
- all multi-variant runs from phase-01 onward
- phase-00 — registers the module stub but does not wire semantics yet

## 3. API spec

Eight values and their semantics:

- `fail_fast` — first failure raises immediately; default for development and CI
- `skip_failed_cell` — (variant, target, horizon) cell skipped and logged; phase-01 default
- `skip_failed_model` — whole variant skipped if one model family fails; horse-race default
- `retry_then_skip` — N retries (configurable) then skip; handles flaky deep training
- `fallback_to_default_hp` — retry with default HP after tuning failure, then skip
- `save_partial_results` — persist everything completed so far plus explicit failure log
- `warn_only` — emit warning and continue; exploratory only
- `hard_error` — error without warning; production default

```python
# macrocast/execution/failure_policy.py (created in Phase 1)
from typing import Literal

Decision = Literal["skip", "raise", "retry"]

def handle_failure(
    *,
    policy: str,
    exc: Exception,
    context: dict,
    retry_count: int = 0,
    max_retries: int = 3,
) -> Decision:
    """Decide what the caller should do after `exc` was raised in `context`.

    `context` must include at minimum `{recipe_id, variant_id, target, horizon,
    model_family, stage}`. Callers are responsible for executing the returned
    action (re-raise, continue loop, or retry the same unit of work).
    """
    ...
```

## 4. Implementation notes

Policy is applied at three points in the execution graph:

- Per-variant in `SweepRunner` — `for variant in plan.variants: try ... except`
- Per-target in multi-target `execute_recipe` — inner loop around fit/predict
- Per-trial in the tuning engine — wraps a single HP trial

The handler is a pure decision function; side effects (logging, writing
`failures.json`, incrementing counters) are the caller's responsibility so the
handler stays testable. `retry_then_skip` and `fallback_to_default_hp` are the only
modes that return `"retry"`; the caller tracks `retry_count`.

Decision boundary — failure policy is a **runtime kwarg**, not a registry-only
declarative field. The registry validates the string; the runtime layer chooses when
to consult the handler. A recipe cannot change policy mid-run.

## 5. Test requirements

`tests/test_failure_policy.py` must cover all 8 × typical error scenarios:

- ValueError, RuntimeError, KeyboardInterrupt (always re-raises regardless of policy)
- `fail_fast` re-raises on first exception
- `skip_failed_cell` returns `"skip"` and does not mutate global state
- `skip_failed_model` returns `"skip"` with `context["stage"] == "fit"` family-wide
- `retry_then_skip` returns `"retry"` while `retry_count < max_retries`, then `"skip"`
- `fallback_to_default_hp` returns `"retry"` exactly once after tuning failure
- `save_partial_results` returns `"skip"` and signals caller to checkpoint
- `warn_only` emits a `UserWarning` and returns `"skip"`
- `hard_error` re-raises without warning

## 6. Owner / ADR references

Owned by phase-01 (first real wiring into SweepRunner). No dedicated ADR yet —
this infra doc serves as the reference until an ADR is promoted. The 8-value
enumeration aligns with recipe registry schema v1.
