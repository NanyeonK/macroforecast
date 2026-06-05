# ML-Useful (Goulet Coulombe, Leroux, Stevanovic, Surprenant 2022, JAE 10.1002/jae.2910)

"How is Machine Learning Useful for Macroeconomic Forecasting?" — paper-faithful
**reconstruction** (the official JAE/ZBW package is data-only; no runnable code).

The paper's contribution is a meta-analysis projecting forecasts into ML-**characteristic
space** (nonlinearity, regularization, hyperparameter-CV, loss) and estimating each
feature's marginal pseudo-R2 (Eq. 11). This harness expresses that design on the
`macroforecast` pipeline.

## Layout
- `registry.py`  — Table-1 model grid as pipeline `Arm`s carrying the 4 ML-feature flags
  + data-poor/data-rich regime; plus the 5 targets with paper target rules
  (INDPRO/CPI/HOUST → I(1) log-avg-growth; UNRATE → avg change; T10YFFM → level).
- `treatment.py` — Eq. 11 treatment-effect regression: per-forecast pseudo-R2 regressed on
  the ML-feature dummies with (target,horizon,origin) fixed effects and Newey-West HAC,
  reusing `mf.data_analysis.newey_west`.
- `run_bounded.py` — bounded end-to-end validation on the official `2018-01.csv` vintage.

## Status
- Bounded validation PASSES end-to-end on the real vintage (data -> preprocess -> arms ->
  pipeline forecasts -> relative RMSE -> alpha_F via HAC). Example bounded output (INDPRO,
  h=1, 11 origins): ARDI relRMSE 0.94, RFAR 0.94; alpha[nonlinear] estimated with HAC.
- The bounded run is NOT expected to reproduce Finding 1 (nonlinearity gains are strongest at
  LONGER horizons / data-rich / stress periods; h=1 on 11 calm origins shows none). The full
  design (5 targets x h in {1,3,9,12,24} x 1980M1-2017M12 expanding x full Table-1 grid) is a
  large compute job that recovers the rankings.

## Key reconstruction choices (see review "Replication Reconstruction Notes")
- Data: official `MainAnalysis/2018-01.csv` (glss-files.zip), McCracken-Ng t-codes.
- Target convention: stationarise the panel, then forecast the h-period AVERAGE of the
  stationary target (`direct_average` + `average_value`) — the package-proven, no-double-
  transform convention (avoids the GCLS double-difference pitfall).
- Treatment regression: within-(target,horizon,origin) demeaning + Newey-West HAC.

## Open gaps (block exact numeric matching, not ranking-level replication)
- No author forecasting code; continuous HP grids (KRR/SVR/EN penalties), seeds, exact CV
  split construction, HAC kernel/bandwidth, and MCS variant are not pinned in the source.
- Per-arm CV-type wiring (BIC/AIC/POOS-CV/K-fold) is a label here; actual per-arm
  model_selection is the next milestone (needed for the CV-feature treatment effect).
- Demo preprocesses upfront; the faithful run moves preprocessing into the pipeline arms
  with an origin-available policy (leak-aware), per the package's stage policies.
