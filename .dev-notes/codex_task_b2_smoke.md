# B2 (Hounyo & Li 2026, IJF 42:414-433, SsPCA) — SETUP + G1 SMOKE only

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`, off main
c3cd17e5). Read FIRST, in order:
1. `.dev-notes/REPLICATION_OBJECTIVES.md` — the four governing purposes (P1 docs trust, P2 bug
   hunt, P3 efficiency, P4 statistically-identical speedups). Your STOP report must cover all four.
2. `.dev-notes/phaseB_design_b2_hounyo_li.md` — the BINDING design (paper-verified + author-MATLAB-
   verified). Follow it. It lists every package mapping, the exact fold scheme, grids, preselection
   order, and the staged parity gates.
3. `qa/hounyo_li_fulltext.txt` — the paper fulltext (oracle for numbers/spec).

Paper note (design §intro): the journal could NOT reproduce this paper's numbers due to compute
cost. So parity is tolerance-based + STAGED. YOUR SCOPE IS **G1 SMOKE ONLY** — do NOT run G2
(table2) or G3 (full). Stop at the smoke gate and report.

## Materials / oracle
- Author MATLAB reproducibility package (ORACLE + data source):
  `~/second_brain/00_wiki/raw/paper_support/downloads_related_20260530/code/Reproducibility package_Hounyo and Li (2024).zip`
  (754MB). Extract the RELEVANT scripts + macro data to `qa/hounyo_li_matlab/` — at minimum
  `SsPCA_tune.m`, `inflation_linear_tune.m`, `dmtest_withp.m`, the AR benchmark script, and the
  macro dataset(s) (the author panel: FRED-MD 1971:4-2023:3 pre-transformed, n=126, + the inflation
  target series). Do NOT extract the whole 754MB blindly if it contains large finance XLS sets —
  list with `unzip -l` first and extract selectively. These author scripts are the tuning oracle:
  read them to CONFIRM the exact fold boundaries, (K,qN) grid, preselection order, and standardization.

## SETUP (do this first)
1. Extract + read the author scripts; CONFIRM against the design doc: fold scheme
   {train 1-80|val 81-130},{1-130|131-190},{1-190|191-240} with within-fold per-observation
   expanding refit (SsPCA_tune.m); (K,qN) grid K=1..10, qN=18:6:108 for n=126; preselection on RAW
   window X regressed on y_{t+h} THEN standardize selected X THEN impute missing=0; target also
   standardized in-window; AR benchmark = BIC lag select (max 12), direct, target standardized.
2. Build the data: convert the author macro panel + inflation target to CSV under `qa/` (a
   conversion script committed to `scripts/replication/hounyo_li_2026_pipeline/`). Author panel is
   the PRIMARY source (design §3).
3. VERIFY current API on main c3cd17e5 (the design was written vs post-#463 main; main has since
   merged 5 fixes — NOTE especially #4 changed the window API to horizon-dependent rules and #1
   made explicit Arm.params pin the search space). Confirm the exact call signatures for:
   `supervised_scaled_pca`/`scaled_pca`/`supervised_pca`/`far`/`pls` params (scale, preselect,
   t_threshold, elastic_net_alpha/l1_ratio, quadratic_factors, n_components), the `ar` BIC route,
   `explicit_folds` splitter (does it take explicit fold boundaries + within-fold expanding refit?
   — design §1(b)1 is the requirement; if the splitter cannot express within-fold expanding refit,
   RECORD it as a package gap and use the closest expressible approximation, clearly labeled), the
   per-arm search-space override for the (K,qN) grid, `impute="zero"`, and the window rolling spec.
   Adapt the design's API calls to the ACTUAL current signatures — do not assume.

## G1 SMOKE (the only run)
Per design §3 gate G1: **1 target = inflation, h = 1, full sample, threshold = none, the 5 methods**
{PCA=`far`, sPCA=`scaled_pca`, SPCA=`supervised_pca`, SsPCA=`supervised_scaled_pca`, PLS=`pls`} with
the AR_BIC benchmark, rolling window size 240, the A2 explicit-fold splitter, (K,qN) grid tuning,
`forecast_policy="direct"`, `run_pipeline(n_jobs="auto")`, `result_store="runs/hl2026_store"`.
Configure every arm to the AUTHOR spec (never package defaults). Score the 5 methods' RMSFE ratios
vs AR (relative_mse) and compare to **Table 2 left column** (inflation, h=1) in the paper.

Gate G1 pass = the 5 methods' RMSFE ratios match Table 2's inflation/h=1 column in SIGN + RANK, and
|Δ ratio| ≤ 0.03. Report per-method ratio / paper / Δ / verdict.

## Constraints
- Configure to AUTHOR spec (from the MATLAB scripts + paper), NEVER package defaults.
- Do NOT patch `macroforecast/**` from this worktree — record any bug/gap/missing-knob in
  `.dev-notes/replication_findings_hounyo_li.md` (BUGS/GAPS section, file:line + severity) as a
  fix-lane input. If a needed capability is missing (e.g., within-fold expanding refit in
  explicit_folds, or per-arm (K,qN) grid override), record it and use the closest labeled
  approximation for smoke; do NOT fake numbers.
- P4: no stats-changing shortcuts (no reduced grid to "make it fast" beyond what smoke scope
  defines; smoke is a legitimately reduced SCOPE = 1 target/1 horizon, not reduced fidelity within
  that cell). Use result_store + n_jobs for speed.
- Do NOT run G2/G3. Do NOT run the finance scope. No git push/gh.

## Deliverables
- `scripts/replication/hounyo_li_2026_pipeline/` — data-build + smoke runner.
- `.dev-notes/replication_findings_hounyo_li.md` — setup notes + BUGS/GAPS (API verification results,
  any missing capability), EFFICIENCY notes.
- `docs/replication/hounyo_li_2026.md` — start the trust page: paper+venue, the journal-not-reproduced
  note, arm→author-param map (via package params, not defaults), and the G1 smoke parity table.

## STOP report → write `qa/codex_last_msg_b2smoke.txt` AND print (4 purposes):
- P1: G1 smoke parity table (5 methods vs Table 2 inflation/h=1: ratio/paper/Δ/verdict); does smoke PASS?
- P2: API verification results; every package gap/bug found (esp. explicit_folds within-fold refit,
  (K,qN) grid override, preselect order); confirm no package patch.
- P3/P4: runtime, result_store use, n_jobs; confirm no fidelity-reducing shortcut.
Then STOP. Do NOT proceed to G2/G3.
