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

## Where the code is

**The trust notes live on `main`; most of the runners do not.** A note here may print a
command like `python3 scripts/replication/medeiros_2021_pipeline/run_block.py` while that
script exists only on its replication branch — so reading the note on `main` does not tell
you where to run it. This table closes that gap.

Branches move. **Pin the commit**, not the branch name, when you cite a result.

| Paper | Code | Pinned commit | Manifest | Runnable from a fresh clone? |
|---|---|---|---|---|
| Medeiros et al. (2021) | `repro/medeiros-2021` | `31e5b451` | `scripts/replication/medeiros_2021_pipeline/replication.yaml` | **yes** — `prepare_data.py --archive …` |
| Zhang, Wahab & Wang (2023) | `repro/zww-2023` | `d2fc450e` | `scripts/replication/zww_2023_pipeline/replication.yaml` | **no** — raw-source acquisition not scripted |
| Hounyo & Li (2026) | `repro/hounyo-li-2026` | `c5b46ec3` | *(not yet written)* | **no** — assumes the author package is pre-extracted |
| Goulet Coulombe et al. (2022) | `repro/gcls-2022` | `0de239ac` | `scripts/replication/gcls_2022_pipeline/replication.yaml` | **yes** — `MF_GCLS_ARCHIVE=… ` |
| Goulet Coulombe (2024), MRF | on `main` | `1e6a0a33` | *(not yet written)* | — |

Each manifest follows `macroforecast-replication-v1`: paper and exhibit, the branch and
both commits, every data source with its hash and whether it is redistributable, the
prepare/smoke/full commands, expected artifacts, and a decomposed status. Fields that would
have to be invented are marked `[FILL]` rather than guessed — a manifest with made-up
hashes looks checkable while being unverifiable.

```{admonition} What these replications validate
:class: warning

Each replication branch pins its own package commit and imports the `macroforecast` in
that checkout — **not** current `main`. A result on `repro/medeiros-2021` is a statement
about the package at that commit. Later `main` fixes (target-only fit samples, the custom
preprocessing state contract, input provenance, the k-fold seed, the interpretation
audit) are not automatically covered by it.

`repro/gcls-2022` is the deliberate exception and says so in its manifest: it is its base
commit plus exactly one cherry-pick, so its `,KF` re-run differs from the earlier run in
the seed and nothing else.
```

## Status, decomposed

A single label hides disagreements. Medeiros was carried as `STRONG` while its own note
records UCSV outside tolerance at h = 3, 6 and 12 — both true, of different things. Each
manifest therefore reports four axes:

| Axis | Question |
|---|---|
| `headline_replication` | does the paper's main qualitative result reproduce? |
| `protocol_fidelity` | is the published specification implemented as written? |
| `table_cell_parity` | what fraction of cells fall inside the stated tolerance? |
| `unresolved` | what numerical gap remains, and why? |

| Paper | headline | protocol | cell parity | note |
|---|---|---|---|---|
| Medeiros | STRONG | STRONG | **PARTIAL** | UCSV outside tolerance at h = 3, 6, 12 — unpublished sampler calibration |
| ZWW | STRONG | STRONG | **PARTIAL** | PCA-VS to 1–2 pp; some competing magnitudes direction-only (~2×) |
| Hounyo–Li | STRONG (author-oracle) | STRONG | 55/60 within \|Δ\| ≤ 0.03 | leak-free output differs for identified reasons |
| GCLS 2022 | STRONG | STRONG | **PARTIAL** | **IN PROGRESS** — 1 of 5 targets re-run; 104/225 beat AR,BIC vs the paper's 128/225 |

## Index

| Paper | Target exhibit | Headline verdict | P1 | P2 | P3 | P4 | Doc |
|---|---|---|---|---|---|---|---|
| Medeiros et al. (2021), inflation forecasting, IJF 37(2) | Table 5 — RMSE ratio vs RW; h = 1/3/6/12 | RW/AR/RF reproduce Table 5 (RF `CLOSE` at all four horizons); UCSV is published-spec faithful, its residual gap is the paper's unpublished MCMC/inverse-gamma calibration, not a package defect. | STRONG | STRONG | STRONG | PARTIAL | `medeiros_2021.md` |
| Hounyo & Li (2026), supervised scaled PCA, IJF 42 | Table 2 (factor-method comparison) + Tables D.11-D.22 (robustness grid) | On the author-oracle (leak-emulating) surface the package reproduces Table 2 and the D.11-D.22 grid; the honest leak-free output differs because the paper's target standardization carries a look-ahead leak. | STRONG | STRONG | STRONG | STRONG | `hounyo_li_2026.md` |
| Zhang, Wahab & Wang (2023), oil-volatility forecasting, IJF 39(2) | Table 3 (futures, main) + Table 4 (spot, robustness); R²_OS % | Reproduces the headline (PCA-VS beats AR; R²_OS positive and rising with horizon, to ~1-2 pp) on both exhibits once ZWW's implicit covariance-PCA choice is matched. | STRONG | STRONG | STRONG | STRONG | `zww_2023_replication.md` |
| Goulet Coulombe et al. (2021), data transformations, IJF 37(4) | Appendix B Tables 3-14 — direct + path-average relative RMSE; 10 targets x 6 horizons | Leak-free and configuration-faithful; four critical bugs fixed; the residual long-horizon gap is the expected R `randomForest`-vs-scikit-learn engine difference. | STRONG | STRONG | N/A¹ | N/A² | `gcls_2021_replication.md` |
| Goulet Coulombe (2024), the macroeconomy as a random forest, JAE 39 | Table 4 — main quarterly results; 6 targets x 14 models x 5 horizons | Full Table 4 reproduced; the MRF family lands inside mean\|Δ\| 0.02-0.10 on every target and the paper's rankings hold. | STRONG | STRONG | STRONG | STRONG | `mrf_2024_replication.md` |
| Goulet Coulombe (2024), monthly companion, JAE 39 App. A.6 | Table 5 — monthly results; 5 targets x 11 models x 5 horizons | All 275 model-cells reproduced (overall mean\|Δ\| 0.098); both A.6 qualitative claims hold, and the benchmark ambiguity in the paper turns out immaterial. | STRONG | STRONG | STRONG | STRONG | `mrf_2024_table5_monthly.md` |
| Han, Lu & Zhou, macro financial trends and market expected returns, RAPS 16(2) | Tables 2-4 — forecast combinations, dense trend ladders, factor methods | The package's first verified **bring-your-own-data** example: driven entirely through `custom_dataset` with the FRED loaders unused, it reproduces the authors' 696-month forecast paths to 1e-15 and their printed `R²_OS` exactly. | STRONG | STRONG | STRONG | STRONG | `han_lu_zhou_2025.md` |

¹ P3 N/A (GCLS): a leak-free faithfulness/correctness replication whose efficiency lever was per-origin preprocessing/factor caching, not a measured speedup or a supervised-scale bottleneck.
² P4 N/A (GCLS): the "identical" claims are correctness invariants (e.g. plain `ols` reproducing the direct/path-average object exactly), not a speedup-identity gate proving an approximate/parallel path equal to an exact path.

### Status vocabulary

- **STRONG** — the purpose is fully delivered and evidenced on the page.
- **PARTIAL** — delivered, but narrower than the exemplar (e.g. an exact equivalence check that is not a labeled speedup-identity gate).
- **ABSENT** — the purpose is expected but not present (no page is currently ABSENT on any purpose).
- **N/A (with reason)** — the purpose does not apply to this replication; the reason is stated so the absence is a scope decision, not a silent gap.

```{toctree}
:hidden:
:glob:

*
```
