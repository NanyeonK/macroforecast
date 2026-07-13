# B2 (Hounyo-Li) — write the DEFINITIVE final trust doc (honest framing, prominent leak finding)

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. All B2 runs are complete. Write the FINAL,
docs-site-quality `docs/replication/hounyo_li_2026.md` + a 1-page `.dev-notes/b2_replication_summary.md`.
DOCS-ONLY: no model runs, no `macroforecast/**` patch, no number changes. Assemble the grids from the
existing result files; get the FRAMING exactly as specified below (this is the binding structure).

Data sources for the tables (read, do not recompute):
- Author-oracle Table 2 grid (55/60): `qa/codex_last_msg_b2table2.txt` + `runs/hl2026_table2_author_oracle/`.
- Our-package leak-free grid: `runs/hl2026_table2_leakfree_load_fred_md/leakfree_table2_parity.csv`.
- Leak decomposition + proofs: `qa/codex_last_msg_b2decomp.txt`, `b2kdiag.txt`, `b2finaldocs.txt`.

## Required structure (in this order)

### 1. Title + venue + the journal "not reproduced, owing to computational cost" note.

### 2. ⚠️ KEY FINDING — Look-ahead bias in the paper's target standardization  (PROMINENT, up front)
State it as the headline finding:
- WHAT: the paper's OOS evaluation standardizes the target block INCLUDING the realized future target
  `y_{T+h}` (`inflation_linear.m:196-197`, `ytplush(:,1+h:T+h)`) — a look-ahead not available in
  real-time forecasting (a data leak).
- REPRODUCTION EVIDENCE (make it airtight): emulating the leak reproduces Table 2 (PCA inflation h=1 =
  0.9706 ≈ paper 0.970); leak-free gives 1.081; the gap decomposes to ~19% direct leak + ~81%
  K-selection tuned on the leaky surface; and `macroforecast.pcr` fed the author surface is
  BIT-IDENTICAL to the author (max diff 3.8e-14) and reproduces the author K exactly. So the paper's
  factor-method advantage depends on this look-ahead.
- IMPLICATION: the paper's OOS results are optimistically biased; macroforecast is leak-free by design,
  so its honest numbers differ. State carefully (based on reading the author's published MATLAB,
  cross-validated; a common subtle form of leakage), not as an accusation.

### 3. Verdict / Bottom line
macroforecast is VERIFIED CORRECT (pcr bit-identical to the author; A2 splitter not defective; K-prefix
speedup bitwise-identical, max diff 0.0). The difference from Table 2 is NOT a package defect — it is the
paper's look-ahead leak (+ data-pipeline + COVID interaction, below). The package REPRODUCES Table 2 on
the author's methodology (inflation 20/20 exact, 55/60), and its honest leak-free output differs for
identified, documented reasons.

### 4. Demonstration A — the package reproduces Table 2 on the author methodology
The author-oracle Table 2 grid (3×4×5), 55/60 within |Δ|≤0.03, inflation 20/20 exact. Full grid table.
Runtime 4:19:30 on 48 cores via the verified K-prefix speedup (vs ~19.6 h/cell naively → weeks).

### 5. Demonstration B — the package's HONEST leak-free output (our data via load_fred_md)
The full leak-free Table 2 grid from `leakfree_table2_parity.csv` (our ratio / paper / Δ). It DIFFERS
from the paper. Then the HONEST caveats (state plainly):
- The difference confounds TWO things: (i) the leak (methodology), and (ii) the data pipeline (our
  `load_fred_md` vintage/transforms vs the paper's exact dataset).
- COVID caveat (verified): the unemployment h=1 leak-free ratios (3.5–4.9 vs paper 1.4–1.7) are **97.7%
  driven by the 2020 COVID point** — the leak-free factor methods overshoot the COVID unemployment shock
  (SsPCA predicts a +20.1 change in May 2020 vs actual −1.5; that single point is ~80% of the SsPCA SSE),
  while the paper's leaky standardization masks it. The unemployment target is verified correctly built
  as the CHANGE in unemployment (not a package/data bug). So this is itself a finding — the leaky
  standardization dampens COVID-period instability — NOT a defect in macroforecast.
- Therefore the CLEAN isolation of "the difference = the leak" is the MATCHED-DATA comparison (inflation,
  controlled data): leak-free ≠ paper by ~0.1–0.2 = the leak; the load_fred_md grid additionally carries
  data + COVID effects.

### 6. What the replication delivered (the 4 purposes)
- P1 (trust): the package reproduces Table 2 on the author methodology (Demonstration A) + transparently
  documents why its honest output differs.
- P2 (bugs/findings): the paper's look-ahead leak; plus package bugs/gaps found (model_selection silent
  override, parallel_cell_timeout, the 6 missing methods now added, etc.).
- P3 (efficiency): the K-prefix grouped evaluator (made the supervised full Table 2 feasible: weeks → 4.3h).
- P4 (statistically identical): K-prefix verified bitwise-identical (max diff 0.0); no reduced grids/folds/origins.
- Package additions on main: pcr, ar_bic, pls score_projection, standardize_scope, nan_policy,
  score_aggregation, preselect_stage, K-prefix evaluator.

### 7. Provenance / caveats
The author IP-growth/unemployment source exists in the reproducibility ZIP; using it would close the 5
author-oracle misses, but the replication deliberately uses macroforecast (not the author's MATLAB) to
compute forecasts. Author's MATLAB is used only as a documented oracle for the leak proof.

## Also write `.dev-notes/b2_replication_summary.md` — 1-page executive summary of the above.

## Constraints: DOCS-ONLY, honest, docs-site quality, no number changes, no `macroforecast/**` patch,
no runs. No push.

## STOP report → `qa/codex_last_msg_b2finaldoc2.txt`: files written + a 5-line verdict summary. Then STOP.
