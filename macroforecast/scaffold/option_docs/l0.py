"""L0 study setup -- per-option documentation.

The L0 layer fixes three execution-policy axes that govern how
``execute_recipe`` runs the cell loop. These choices are study-wide:
they interact with every downstream layer (random_seed propagation,
parallel-unit scheduling, failure-policy on cell errors), so getting
them right at recipe-authoring time matters.
"""
from __future__ import annotations

from . import register
from .types import CodeExample, OptionDoc, Reference


_REVIEWED = "2026-05-04"
_REVIEWER = "macroforecast author"


# ---------------------------------------------------------------------------
# L0.A failure_policy
# ---------------------------------------------------------------------------

_FAILURE_POLICY_FAIL_FAST = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="failure_policy",
    option="fail_fast",
    summary="Stop the entire study on the first cell that errors.",
    description=(
        "When the cell-loop catches an exception in any sweep cell, ``fail_fast`` "
        "raises immediately and the manifest is **not** written. The remaining "
        "cells are skipped.\n\n"
        "This is the default because the typical authoring failure mode is a "
        "schema or data error that affects every cell -- catching it after the "
        "first cell saves wall-clock and surfaces the problem with a single "
        "traceback rather than a wall of identical errors. For sweeps where "
        "cells *can* fail independently (e.g., one model family throws on a "
        "particular target while others succeed), use ``continue_on_failure`` "
        "instead so partial results survive."
    ),
    when_to_use=(
        "Default for every authoring iteration. Pick this while the recipe is "
        "still being tuned; the first failure tells you exactly what to fix "
        "without waiting for a full sweep to finish."
    ),
    references=(
        Reference(
            citation=(
                "macroforecast design Part 1, L0 §A: 'fail_fast vs continue_on_failure "
                "is the canonical execution-policy choice for any cell-loop study.'"
            ),
        ),
    ),
    when_not_to_use=(
        "Long-running production sweeps where a transient failure on one cell "
        "(e.g., a memory hiccup on one bootstrap iteration) should not abort "
        "the whole study."
    ),
    related_options=("continue_on_failure",),
    examples=(
        CodeExample(
            title="Author-time recipe (default)",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    failure_policy: fail_fast\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


_FAILURE_POLICY_CONTINUE = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="failure_policy",
    option="continue_on_failure",
    summary="Record failed cells in the manifest and keep the sweep running.",
    description=(
        "Per-cell exceptions are caught by the cell loop, the cell's "
        "``CellExecutionResult.error`` and ``traceback`` fields are populated, "
        "and the loop moves on to the next cell. The manifest's "
        "``cells_summary`` distinguishes succeeded from failed cells; the "
        "failed-cell entries carry the captured traceback for post-hoc "
        "diagnosis.\n\n"
        "Replication still runs end-to-end on a manifest with failed cells: "
        "``replicate()`` re-executes every cell and verifies the failure "
        "occurs in the same place with the same exception class."
    ),
    when_to_use=(
        "Production horse-race sweeps where partial coverage is more useful "
        "than no coverage. Common examples: a 50-cell model-family sweep where "
        "one optional family (xgboost without the extra) fails to import, or a "
        "long bootstrap where a single iteration trips a numerical edge case."
    ),
    references=(
        Reference(
            citation=(
                "macroforecast design Part 1, L0 §A: 'continue_on_failure preserves "
                "partial coverage; the manifest carries enough context to "
                "diagnose each failed cell after the run.'"
            ),
        ),
    ),
    when_not_to_use=(
        "Authoring iteration -- failures are usually configuration problems "
        "that affect every cell, and ``fail_fast`` shortens the feedback loop."
    ),
    related_options=("fail_fast",),
    examples=(
        CodeExample(
            title="Production sweep over many model families",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    failure_policy: continue_on_failure\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


# ---------------------------------------------------------------------------
# L0.A reproducibility_mode
# ---------------------------------------------------------------------------

_REPRODUCIBILITY_SEEDED = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="reproducibility_mode",
    option="seeded_reproducible",
    summary="Fix a deterministic seed and propagate it through every RNG.",
    description=(
        "The cell-loop reads ``leaf_config.random_seed`` (default ``0``) and "
        "applies it to ``random.seed``, ``numpy.random.seed``, "
        "``torch.manual_seed`` (when torch is installed), and the "
        "``PYTHONHASHSEED`` environment variable for the current process. "
        "Each L4 estimator inherits the seed via its ``params.random_state`` "
        "key (issue #215); per-fit-node ``random_state`` overrides the L0 "
        "seed when present.\n\n"
        "This is the only mode under which ``macroforecast.replicate(manifest)`` "
        "can verify bit-exact sink hashes. Use it for every study you intend "
        "to publish, share, or compare against later."
    ),
    when_to_use=(
        "Default. Required for any study where bit-exact replication matters "
        "(papers, internal benchmarks, regression tests, comparisons across "
        "package versions)."
    ),
    references=(
        Reference(
            citation=(
                "Stodden, McNutt, Bailey et al. (2016) 'Enhancing reproducibility "
                "for computational methods', Science 354(6317)."
            ),
            doi="10.1126/science.aah6168",
        ),
        Reference(
            citation="macroforecast PR #215 -- L0 random_seed propagation into L4 random_state.",
        ),
    ),
    when_not_to_use=(
        "Stochastic exploration where the explicit goal is to characterise "
        "the variability of a procedure across seeds. Pick ``exploratory`` "
        "and re-run; the manifest will record that the seed was unfixed."
    ),
    related_options=("exploratory",),
    examples=(
        CodeExample(
            title="Reproducible study (default)",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    reproducibility_mode: seeded_reproducible\n"
                "  leaf_config:\n"
                "    random_seed: 42\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


_REPRODUCIBILITY_EXPLORATORY = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="reproducibility_mode",
    option="exploratory",
    summary="Do not fix stochastic seeds; each run draws fresh randomness.",
    description=(
        "Skips the global RNG seeding that ``seeded_reproducible`` performs. "
        "Each cell pulls its own randomness from the OS entropy pool; "
        "downstream estimators that take an explicit ``random_state`` still "
        "use whatever the recipe sets per node, but the L0 default of ``0`` "
        "is *not* propagated.\n\n"
        "``replicate()`` cannot guarantee bit-exact sink hashes under this "
        "mode -- the recipe still re-runs and produces structurally identical "
        "manifests, but the numeric forecasts will differ run-over-run."
    ),
    when_to_use=(
        "Sensitivity studies where you want to measure how much variability "
        "the random components introduce. Wrap the run in a sweep over "
        "several executions and compare the spread."
    ),
    references=(
        Reference(
            citation=(
                "macroforecast design Part 1, L0 §A: 'exploratory mode trades "
                "reproducibility for unbiased sampling of stochastic variability.'"
            ),
        ),
    ),
    when_not_to_use=(
        "Anything you intend to publish or share. The reviewer will not be "
        "able to reproduce your manifest."
    ),
    related_options=("seeded_reproducible",),
    examples=(
        CodeExample(
            title="Sensitivity sweep over fresh seeds",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    reproducibility_mode: exploratory\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


# ---------------------------------------------------------------------------
# L0.A compute_mode
# ---------------------------------------------------------------------------

_COMPUTE_SERIAL = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="compute_mode",
    option="serial",
    summary="Run every sweep cell sequentially in the calling process.",
    description=(
        "The cell loop iterates expanded cells one at a time. No worker "
        "pool, no extra processes, no thread pool. ``parallel_unit`` is "
        "ignored.\n\n"
        "This is the default because (a) most authoring sweeps are small "
        "enough that the cell loop dwarfs scheduling overhead, and (b) "
        "every parallel mode introduces an additional surface for "
        "scheduling-induced non-determinism that has to be ruled out "
        "explicitly. Pick ``parallel`` when wall-clock matters and the "
        "study has been validated under ``serial`` first."
    ),
    when_to_use=(
        "Default. Authoring iteration, small sweeps, debugging, any case "
        "where a stack trace from a particular cell needs to be readable "
        "without thread / process noise."
    ),
    references=(
        Reference(
            citation=(
                "macroforecast design Part 1, L0 §A: 'compute_mode = serial is the "
                "deterministic default; parallel modes are opt-in for "
                "wall-clock-sensitive sweeps.'"
            ),
        ),
    ),
    when_not_to_use=(
        "Multi-cell sweeps where each cell takes more than a minute and the "
        "machine has multiple CPU cores. Switch to ``parallel`` and record "
        "the speed-up in the manifest."
    ),
    related_options=("parallel",),
    examples=(
        CodeExample(
            title="Default serial study",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    compute_mode: serial\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


_COMPUTE_PARALLEL = OptionDoc(
    layer="l0",
    sublayer="l0_a",
    axis="compute_mode",
    option="parallel",
    summary="Distribute work over multiple workers; pick the unit via parallel_unit.",
    description=(
        "Activates the parallel cell loop. The granularity is controlled by "
        "the ``parallel_unit`` sub-axis:\n\n"
        "* ``cells`` -- one process per sweep cell (``ProcessPoolExecutor``). "
        "  Cell-level parallelism is the safest path because cells are by "
        "  construction independent.\n"
        "* ``models`` -- threads over ``fit_model`` nodes inside a single "
        "  cell (issue #204). Sklearn-family estimators release the GIL; the "
        "  thread pool avoids the pickling overhead of processes.\n"
        "* ``oos_dates`` -- threads over walk-forward origins inside a fit "
        "  node (issue #250). Per-origin RNG state is derived deterministically "
        "  from ``base_seed + position`` (issue #279) so thread scheduling "
        "  cannot affect the forecasts.\n"
        "* ``horizons`` / ``targets`` -- map to the same fan-out when L4 "
        "  produces single-horizon / single-target output per fit node.\n\n"
        "``leaf_config.n_workers`` (cell-level) and ``n_workers_inner`` "
        "(sub-cell) cap the pool sizes."
    ),
    when_to_use=(
        "Long sweeps on multi-core machines. Validate the manifest under "
        "``compute_mode = serial`` first to confirm the recipe is "
        "deterministic, then flip to ``parallel`` and verify "
        "``replicate()`` still passes."
    ),
    references=(
        Reference(
            citation="macroforecast PR #173 -- cell-level ProcessPoolExecutor.",
        ),
        Reference(
            citation="macroforecast PR #204 / #250 -- sub-cell ThreadPoolExecutor.",
        ),
        Reference(
            citation="macroforecast PR #279 -- deterministic per-origin seed propagation.",
        ),
    ),
    when_not_to_use=(
        "Recipes that mutate global state (e.g., a custom L3 op that writes "
        "to a shared file or a base estimator with a non-thread-safe "
        "C extension). Audit thread / process safety before flipping the "
        "switch."
    ),
    related_options=("serial",),
    examples=(
        CodeExample(
            title="Cell-level parallel sweep",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    compute_mode: parallel\n"
                "    parallel_unit: cells\n"
                "  leaf_config:\n"
                "    n_workers: 4\n"
            ),
        ),
        CodeExample(
            title="Sub-cell parallel over fit_model nodes",
            code=(
                "0_meta:\n"
                "  fixed_axes:\n"
                "    compute_mode: parallel\n"
                "    parallel_unit: models\n"
                "  leaf_config:\n"
                "    n_workers_inner: 8\n"
            ),
        ),
    ),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


register(
    _FAILURE_POLICY_FAIL_FAST,
    _FAILURE_POLICY_CONTINUE,
    _REPRODUCIBILITY_SEEDED,
    _REPRODUCIBILITY_EXPLORATORY,
    _COMPUTE_SERIAL,
    _COMPUTE_PARALLEL,
)
