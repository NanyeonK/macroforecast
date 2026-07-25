# Goulet Coulombe (2024) — MRF Table 4 replication

Reproduces Table 4 ("Main Quarterly Results") of *The Macroeconomy as a Random Forest*
(JAE 2024; arXiv 2006.12724): pseudo-out-of-sample relative RMSEs vs AR(4), for all 14
models x 6 targets x 5 horizons. See `docs/replication/mrf_2024_replication.md` for the
full write-up, deviation attribution, and the package fixes this replication surfaced
(PRs #468/#469/#470/#472).

## Run

    python scripts/replication/mrf_2024_pipeline/run_table4.py <TARGET>   # GDP|UR|INF|IR|SPREAD|HOUST

Data = `mf.data.load_fred_qd("2020-01")`; UR target = log-then-diff; S_t (Table 2) built
with expanding pca_step/maf_step; models fit per 2-year estimation point with
`random_state=42, parallelise=True`. Each run prints a mine/paper table and writes
`results/g2_<TARGET>.json`. Reproducible (seed=42); the MRF is bit-identical serial vs
parallel after PR #469.

## Results
Per-target overall mean|Δ| to the paper: GDP 0.058, HOUST 0.068, INF 0.083, UR 0.105,
IR 0.135, SPREAD 0.210 (with standardized penalized models). Core MRF family within
0.02-0.10 on every target. Raw cells in `results/`.
