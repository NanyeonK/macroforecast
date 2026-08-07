# Fix-lane spec: CV/grid selection silently routed to IC for IC-owning models

Status: SPEC ONLY — do NOT implement in the b4 worktree. This goes to the macroforecast
main fix-lane (like the all-NaN and DM-None fixes). Discovered during B4 (GCLS-2022) G3.

## The defect (package)

File: `macroforecast/forecasting/policies/base.py`, the `uses_ic` computation (line 156):

```python
explicit_ic = selected_method in {"information_criterion", "ic"}
model_owned_ic = str(getattr(model_spec, "selection_method", "cv")).lower() in ("bic", "aic", "aicc")
uses_ic = explicit_ic or (ic_selection_enabled and model_owned_ic)          # <-- BUG
if should_select and uses_ic:
    ...  # select_by_information_criterion(...)   (IC path; no validation split)
elif should_select:
    ...  # select_params(...)                     (grid/CV path; uses validation_splitter)
```

`model_owned_ic` is True for any model whose registered `selection_method` is `bic`/`aic`/
`aicc` — i.e. `ar` and `far` (`get_model('ar').selection_method == 'bic'`; same for `far`).
When such a model is used, `uses_ic` is True **even when the arm explicitly sets a
non-IC `SearchSpec`** (e.g. `method="grid"` with a `validation_splitter`). So the arm is
routed to `select_by_information_criterion` (BIC on the full training sample) and
`select_params` is **never called** — the requested POOS-CV / K-fold validation never runs.

### Evidence (re-derived from disk, B4 G3)
- Registry arms `AR,POOS`, `AR,KF`, `ARDI,POOS`, `ARDI,KF` set
  `SearchSpec(method="grid", validation_splitter=poos|random_kfold)`.
- Instrumented pipeline run (n_jobs=1): call counts `select_by_information_criterion=3,
  select_params=0, _resolve_degraded_selection=0` for the 3-arm {AR,BIC/POOS/KF} slice.
- In the G2 store, `AR,POOS ≡ AR,BIC` and `AR,KF ≡ AR,BIC` forecasts at **0/455 origins
  differ** (max |Δ| = 0.00e+00, 1980–2017); same for the ARDI trio. `AIC` differs.
- Isolated `select_params("ar", …, search=SearchSpec(grid, validation_splitter=poos),
  fixed_params={"direct": True})` correctly selects **bigger** models (n_lag=3 @1980,
  **n_lag=12 @2010**) — matching the paper's "POOS-CV selects bigger/worse models".
- So the defect is the routing/dispatch, not the CV machinery (which works when reached).

## The minimal fix

```python
explicit_non_ic = selected is not None and selected_method not in ("", "information_criterion", "ic")
uses_ic = explicit_ic or (ic_selection_enabled and model_owned_ic and not explicit_non_ic)
```

Rationale: an EXPLICIT non-IC `SearchSpec` (grid/random/bayesian/genetic/cv_path with a
splitter) must suppress the model's built-in IC fallback. The `model_owned_ic` default is
retained so AR/FM with **no** explicit SearchSpec still default to IC (then
`selected is None` or `selected_method == ""` → `explicit_non_ic = False`), preserving the
"AR runs with a window that carries no validation block" behaviour the code comment cites.

## Golden-identity gate (must pass before merge)

1. `tests/forecasting` + `tests/model_selection` + `tests/pipeline` green.
2. **Bit-identical IC path** for arms that legitimately use IC: `AR,BIC`, `AR,AIC`,
   `ARDI,BIC`, `ARDI,AIC` (explicit IC), and any model-default-IC arm with no explicit
   SearchSpec — same selected order + same forecasts as pre-fix.
3. The 42 non-`ar`/`far` arms (ridge/lasso/EN/kernel_ridge/svr/random_forest, all
   `selection_method='cv'`) are unchanged — they already reached `select_params`
   (`model_owned_ic=False`); the new clause is a no-op for them.
4. Add a regression test: an `ar`/`far` arm with `SearchSpec(method="grid",
   validation_splitter=poos)` calls `select_params` (not `select_by_information_criterion`).

## Behavioral assertion (post-fix, on the B4 registry)

- `AR,POOS` and `AR,KF` now call `select_params` and select an order **different from
  BIC** at later origins (e.g. n_lag ≈ 12 around 2010), and `POOS ≠ KF` in general
  (different validation splitters). Same for `ARDI,POOS`/`ARDI,KF` via `far`.
- In the G3 CV regression (Table 2), the `CV-POOS` coefficient turns **negative** (paper
  direction: −1.351), i.e. POOS-CV selects bigger/worse models than BIC.

## Blast radius + re-run plan

- Affected arms: exactly **4** — `AR,POOS`, `AR,KF`, `ARDI,POOS`, `ARDI,KF` (the only
  `ar`/`far` arms carrying an explicit CV SearchSpec). Confirmed: only `ar`,`far` are
  IC-owning; ridge/lasso/EN/kernel_ridge/svr/RF are `cv` (unaffected — parity shows their
  POOS ≠ KF already).
- Cells to recompute: 4 arms × 5 targets × 5 horizons = **100 cells** (of 1150), plus
  their G3/Table-2 rows. The Table A1 headline (best-model ML/factor > AR, 25/25) is
  INDEPENDENT of these arms and does not change.
- Re-run is NOT auto-invalidated: the result_store cell digest is **config-based**
  (arm SearchSpec + data), which the package fix does not change. So the 4 arms' stale
  (BIC-identical) cells must be **explicitly purged** from BOTH stores
  (`_result_store_indpro`, `_result_store_g2rest`) and recomputed; the other 42 arms'
  1050 cells are correct and are reused. Est. cost ≈ 1–3h (AR/ARDI are fast arms).
