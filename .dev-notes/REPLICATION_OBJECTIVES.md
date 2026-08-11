# Replication program — governing objectives (ALL papers B1–B5)

The replication program is NOT just "get the parity numbers." It has FOUR co-equal purposes.
Every replication lane must deliver on all four and the monitor tracks all four.

## Purpose 1 — DOCUMENTED faithful replication that demonstrates package TRUST
The deliverable is a polished `docs/replication/<paper>.md` page (docs-site quality, not a scratch
worksheet) that a skeptical reader can follow to see macroforecast faithfully reproduce a published
paper. It must contain: the paper + venue, exactly which exhibits are replicated, the EXACT
macroforecast configuration for every arm (arm→author-param map, expressed through package
parameters — never package defaults), the parity table (MATCH/CLOSE/DIVERGENT vs the paper's
numbers with tolerances), a runnable reproduction recipe (`scripts/replication/<paper>_pipeline/`),
and an honest gaps/caveats section. This page is the package's public trust artifact — write it to
that standard.

## Purpose 2 — actively HUNT macroforecast bugs/gaps during replication
Replicating a real paper exercises the package harder than unit tests. Do NOT wait to be blocked —
PROACTIVELY record every defect, gap, silent-wrong-number risk, missing parameter, confusing API,
or friction encountered, in `.dev-notes/replication_findings_<paper>.md` under a BUGS/GAPS section
(file:line + repro + severity). These become fix-lane inputs for the orchestrator (Fable). Never
patch package code from a replication worktree — record and report. A replication that finds zero
package issues has under-looked; assume there are some and go find them.

## Purpose 3 — improve TECHNICAL efficiency of the package (via real workloads)
The replication's realistic workload (FRED-MD × many models × horizons × origins) is the venue to
find and fix performance problems. Record every slow/redundant/wasteful computation you hit in an
EFFICIENCY section of the findings file (what's slow, file:line, why, estimated cost on this
workload). Cross-reference the standing efficiency review (`.dev-notes/pipeline_efficiency_review.md`
— feature-fit sharing, per-target cache, evaluate-pivot reuse, checkpoint chunking, etc.).

## Purpose 4 — speedups must be STATISTICALLY IDENTICAL (same numbers, just faster)
This is the hard constraint on Purpose 3. Any efficiency change must produce BITWISE/near-bitwise
IDENTICAL results — never trade statistical fidelity for speed. Forbidden: reducing MCMC draws,
loosening tolerances, subsampling, fewer trees, coarser grids — those change the statistics and are
NOT acceptable speedups. Allowed: n_jobs parallelism (with seeded determinism), caching/reuse
(ResultStore, shared fits), vectorization, avoiding redundant recomputation, algebraic shortcuts
that are exact. Every proposed speedup must be VALIDATED by an identical-numbers before/after check
(run the same cell both ways, assert equality to tolerance) and that check recorded. The
replication itself is the equivalence oracle: a paper whose numbers reproduce identically before and
after a speedup proves the speedup is safe.

## How the four compose per lane
- Get the parity (configure package → author spec) — the vehicle.
- While doing it: fill the docs trust page (P1), the bugs/gaps log (P2), the efficiency log (P3)
  with statistically-identical-speedup proposals + validation (P4).
- At each STOP/report, summarize ALL FOUR: parity status, docs page state, new bugs/gaps, new
  efficiency findings (+ any validated speedups). Not just the numbers.
