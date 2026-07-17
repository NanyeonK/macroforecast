# macroforecast replication trust notes

`macroforecast` reproduces the published results of several external forecasting papers
directly through its callable API. Each replication is written up as a trust note built on a
common four-purpose spine:

- **P1 — trust via faithful replication.** A parity table of reproduced-vs-paper numbers
  (metric and tolerance named) plus a one-line headline verdict.
- **P2 — bugs caught during replication.** Defects in the package or runner that the
  replication surfaced, and how they were resolved.
- **P3 — technical efficiency.** The lever that made the run feasible (caching, parallel
  evaluation), with a wall-clock figure where one applies.
- **P4 — statistically-identical speedups.** An identity gate showing a fast/parallel path is
  bit- or statistically identical to the exact path (or the exact equivalence check the
  replication does provide).

**Common page skeleton.** Each note carries: a one-line headline verdict; a labeled "four
purposes" map (P1-P4, each STRONG / PARTIAL / ABSENT / N/A-with-reason); a KEY FINDING where
one exists; the parity/evidence body (every parity table names its metric and tolerance band);
and provenance/caveats. Where a purpose does not apply to a given replication it is marked
**N/A with a stated reason**, so an absence is always a scope decision, not a silent gap.
`hounyo_li_2026.md` is the exemplar page.

## Index

| Paper | Target exhibit | Headline verdict | P1 | P2 | P3 | P4 | Doc |
|---|---|---|---|---|---|---|---|
| Medeiros et al. (2021), inflation forecasting, IJF 37(2) | Table 5 — RMSE ratio vs RW; h = 1/3/6/12 | RW/AR/RF reproduce Table 5 (RF `CLOSE` at all four horizons); UCSV is published-spec faithful, its residual gap is the paper's unpublished MCMC/inverse-gamma calibration, not a package defect. | STRONG | STRONG | STRONG | PARTIAL | `medeiros_2021.md` |
| Hounyo & Li (2026), supervised scaled PCA, IJF 42 | Table 2 (factor-method comparison) + Tables D.11-D.22 (robustness grid) | On the author-oracle (leak-emulating) surface the package reproduces Table 2 and the D.11-D.22 grid; the honest leak-free output differs because the paper's target standardization carries a look-ahead leak. | STRONG | STRONG | STRONG | STRONG | `hounyo_li_2026.md` |
| Zhang, Wahab & Wang (2023), oil-volatility forecasting, IJF 39(2) | Table 3 (futures, main) + Table 4 (spot, robustness); R²_OS % | Reproduces the headline (PCA-VS beats AR; R²_OS positive and rising with horizon, to ~1-2 pp) on both exhibits once ZWW's implicit covariance-PCA choice is matched. | STRONG | STRONG | STRONG | STRONG | `zww_2023_replication.md` |
| Goulet Coulombe et al. (2021), data transformations, IJF 37(4) | Appendix B Tables 3-14 — direct + path-average relative RMSE; 10 targets x 6 horizons | Leak-free and configuration-faithful; four critical bugs fixed; the residual long-horizon gap is the expected R `randomForest`-vs-scikit-learn engine difference. | STRONG | STRONG | N/A¹ | N/A² | `gcls_2021_replication.md` |

¹ P3 N/A (GCLS): a leak-free faithfulness/correctness replication whose efficiency lever was per-origin preprocessing/factor caching, not a measured speedup or a supervised-scale bottleneck.
² P4 N/A (GCLS): the "identical" claims are correctness invariants (e.g. plain `ols` reproducing the direct/path-average object exactly), not a speedup-identity gate proving an approximate/parallel path equal to an exact path.

### Status vocabulary

- **STRONG** — the purpose is fully delivered and evidenced on the page.
- **PARTIAL** — delivered, but narrower than the exemplar (e.g. an exact equivalence check that is not a labeled speedup-identity gate).
- **ABSENT** — the purpose is expected but not present (no page is currently ABSENT on any purpose).
- **N/A (with reason)** — the purpose does not apply to this replication; the reason is stated so the absence is a scope decision, not a silent gap.
