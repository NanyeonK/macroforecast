"""L8 output / provenance -- per-option documentation.

L8 governs how a run is materialised on disk and how the manifest is
populated for bit-exact replication. Sub-layers:

* L8.A export_format / compression -- file formats and compression policy,
* L8.B saved_objects -- which artifacts to persist + model serialisation,
* L8.C provenance_fields / manifest_format -- manifest content,
* L8.D artifact_granularity / naming_convention -- per-cell sink layout.

Every option ships hand-written description + when-to-use guidance with
references to the design document and (where applicable) the underlying
file-format / serialization standard.
"""
from __future__ import annotations

from . import register
from .types import OptionDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L8 = Reference(
    citation="macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'",
)
_REF_PARQUET = Reference(
    citation="Apache Parquet specification (apache/parquet-format).",
    url="https://parquet.apache.org/docs/file-format/",
)
_REF_PANDAS_LATEX = Reference(
    citation="pandas DataFrame.to_latex documentation.",
    url="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html",
)


def _e(sublayer: str, axis: str, option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "",
       references: tuple[Reference, ...] = (_REF_DESIGN_L8,),
       related: tuple[str, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l8", sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# ---------------------------------------------------------------------------
# L8.A export_format -- 9 formats
# ---------------------------------------------------------------------------

_FORMAT_DOCS: dict[str, tuple[str, str, str, tuple[Reference, ...]]] = {
    "json": (
        "JSON dump of every artifact (default).",
        (
            "Default round-trip-safe format; native Python / JS / R "
            "support; preserves nested structure (dicts of dicts of "
            "DataFrames). All numeric values rendered as floats with "
            "full precision; date-like values rendered as ISO 8601 "
            "strings."
        ),
        "Default; round-trips cleanly into Python / JS / R.",
        (_REF_DESIGN_L8,),
    ),
    "csv": (
        "CSV tables for tabular artifacts (forecasts, metrics, importance).",
        (
            "Standard comma-separated values, UTF-8 encoded. The "
            "lowest-common-denominator format for spreadsheet / R "
            "workflows. Loses dtype information (everything becomes "
            "string on round-trip); for analytics workloads prefer "
            "``parquet``."
        ),
        "Spreadsheet / R workflows; collaborators who avoid JSON.",
        (_REF_DESIGN_L8,),
    ),
    "parquet": (
        "Apache Parquet (pyarrow); columnar binary tabular format.",
        (
            "Columnar binary format with full dtype preservation, "
            "automatic dictionary encoding for low-cardinality "
            "columns, and per-column compression. 5-10× smaller than "
            "CSV for typical macro panels; an order of magnitude "
            "faster to read for column-subset queries. Requires "
            "``pyarrow`` (already a transitive dependency)."
        ),
        "Large-scale analytics; preserving dtypes; cross-language workflows (Spark, DuckDB, R arrow).",
        (_REF_DESIGN_L8, _REF_PARQUET),
    ),
    "json_csv": (
        "Both JSON and CSV for every applicable artifact.",
        (
            "Convenience option emitting both formats. Used when "
            "downstream consumers vary -- Python users want JSON "
            "round-trip, R / Excel users want CSV. Doubles the "
            "artifact-directory size."
        ),
        "When downstream consumers vary across both Python and Excel / R.",
        (_REF_DESIGN_L8,),
    ),
    "json_parquet": (
        "Both JSON and Parquet for every applicable artifact.",
        (
            "Hybrid option for runs that combine reproducibility "
            "(JSON for the manifest / small artifacts) with analytics "
            "(Parquet for large forecast tables). Recommended for "
            "production sweeps."
        ),
        "Hybrid analytics + reproducibility setups.",
        (_REF_DESIGN_L8, _REF_PARQUET),
    ),
    "latex_tables": (
        "LaTeX ``tabular`` snippets ready to ``\\input`` into a paper.",
        (
            "Emits one ``.tex`` file per tabular artifact (forecasts, "
            "metrics, ranking). Booktabs-friendly column alignment "
            "and column-name escaping; uses pandas' ``to_latex`` "
            "backend."
        ),
        "Paper-draft pipelines.",
        (_REF_DESIGN_L8, _REF_PANDAS_LATEX),
    ),
    "markdown_report": (
        "Single Markdown report bundling tables and figure references.",
        (
            "Renders a self-contained ``.md`` document with pipe-"
            "aligned tables and embedded image references. Intended "
            "as the human-readable summary for stakeholder reports "
            "and GitHub / wiki documentation."
        ),
        "Lightweight Markdown / GitHub-rendered reports.",
        (_REF_DESIGN_L8,),
    ),
    "html_report": (
        "Self-contained HTML report with embedded plots and tables.",
        (
            "Renders a single ``.html`` file combining tables (via "
            "pandas' ``to_html``) and base64-embedded matplotlib "
            "figures. Opens in any browser without a server; ideal "
            "for stakeholder-shareable reports without LaTeX tooling."
        ),
        "Stakeholder-shareable reports without LaTeX tooling.",
        (_REF_DESIGN_L8,),
    ),
    "all": (
        "Emit every supported export format together.",
        (
            "Comprehensive option emitting JSON + CSV + Parquet + "
            "LaTeX + Markdown + HTML for every applicable artifact. "
            "Largest disk footprint but covers every downstream "
            "consumer in one run."
        ),
        "Comprehensive reproducibility / sharing -- single run that covers every audience.",
        (_REF_DESIGN_L8,),
    ),
}
for option, (summary, desc, when, refs) in _FORMAT_DOCS.items():
    register(_e("L8_A_export_format", "export_format", option, summary, desc, when,
        references=refs,
        related=tuple(k for k in _FORMAT_DOCS if k != option)[:5]))


# ---------------------------------------------------------------------------
# L8.A compression -- none / gzip / zip
# ---------------------------------------------------------------------------

register(
    _e("L8_A_export_format", "compression", "none",
       "No compression (default).",
       (
           "Default. Files are written uncompressed -- cheapest at "
           "write time and most convenient for direct browsing / "
           "spot-checking. Recommended for development."
       ),
       "Default; cheapest write-time option.",
       related=("gzip", "zip")),
    _e("L8_A_export_format", "compression", "gzip",
       "Gzip-compress every output file individually.",
       (
           "Each ``.json`` / ``.csv`` becomes ``.json.gz`` / ``.csv.gz``. "
           "Reduces artifact size by 60-80% for typical macro panels "
           "with marginal write-time overhead. Read-side: pandas / "
           "pyarrow auto-detect the gzip extension."
       ),
       "Reducing artifact size for archival; production sweeps.",
       related=("none", "zip")),
    _e("L8_A_export_format", "compression", "zip",
       "Zip-archive the entire run output directory.",
       (
           "Wraps the run output directory in a single ``.zip`` "
           "archive after writing. Convenient for transferring an "
           "entire run via email / web upload as a single file. "
           "Slightly less efficient than per-file gzip but shipping "
           "a single archive matters for some workflows."
       ),
       "Packaging the run for transfer over email / file-sharing services.",
       related=("none", "gzip")),
)


# ---------------------------------------------------------------------------
# L8.B saved_objects -- 20 saveable artifacts
# ---------------------------------------------------------------------------

_SAVED_DOCS: dict[str, tuple[str, str, str]] = {
    "raw_panel": (
        "Raw L1 panel before any L2 cleaning.",
        (
            "The original raw FRED-MD / -QD / -SD / custom panel. "
            "Default-off because raw FRED panels are large; enabling "
            "this makes the run fully self-contained -- a downstream "
            "user can re-run the entire pipeline from the manifest "
            "alone without internet access."
        ),
        "Default-off for size; enable for fully self-contained runs.",
    ),
    "clean_panel": (
        "Cleaned L2 panel (post tcode / outlier / imputation / frame edge).",
        (
            "The output of the L2 pipeline. Useful when downstream "
            "re-runs need to skip the (potentially expensive) "
            "cleaning stages."
        ),
        "When downstream re-runs without re-cleaning are needed.",
    ),
    "feature_metadata": (
        "L3 column lineage + pipeline definitions.",
        (
            "The L3 metadata sink containing per-feature lineage, "
            "transformation chain, and pipeline ID. Default-on when "
            "L7 ``lineage_attribution`` or ``transformation_attribution`` "
            "is active -- those ops require this metadata to function."
        ),
        "Default-on when L7 lineage / transformation_attribution is active.",
    ),
    "forecasts": (
        "Per-cell point forecasts.",
        (
            "The headline output: per (cell, target, horizon, origin) "
            "forecast. Default-on; required for replication and for "
            "every downstream L5 / L6 / L7 op."
        ),
        "Default-on; required for replication.",
    ),
    "forecast_intervals": (
        "Per-cell prediction intervals (when forecast_object = quantile / density).",
        (
            "Quantile forecasts at the user-specified ``α`` levels "
            "(default 5% / 50% / 95%). Default-on when L4 emits "
            "``forecast_object = quantile`` or ``density``."
        ),
        "Default-on when forecast_object = quantile / density.",
    ),
    "metrics": (
        "L5 metric tables.",
        (
            "Per-cell per-metric scores aggregated by the L5.C "
            "configuration. Default-on; the standard headline output "
            "for every horse-race study."
        ),
        "Default-on.",
    ),
    "ranking": (
        "L5.E ranking tables.",
        (
            "Models ranked by primary metric / MCS inclusion / "
            "Borda count / etc. Default-on when L5.E ranking is "
            "active."
        ),
        "Default-on when ranking is active.",
    ),
    "decomposition": (
        "L5.D decomposition tables (per-period / per-block / Shapley).",
        (
            "Variance / loss decomposition outputs. Default-on when "
            "L5.D decomposition is active."
        ),
        "Default-on when decomposition is active.",
    ),
    "tests": (
        "L6 test outputs (DM / GW / MCS / SPA / RC / StepM / PT / residual / density).",
        (
            "Test statistics, p-values, kernel choices, and lag-truncation "
            "parameters for every L6 sub-layer that is enabled. "
            "Default-on when any L6 sub-layer is active."
        ),
        "Default-on when L6 sub-layers are active.",
    ),
    "importance": (
        "L7 importance outputs.",
        (
            "Tables and figures from every L7.A op in the recipe's "
            "interpretation DAG. Default-on when L7 is enabled."
        ),
        "Default-on when L7 is active.",
    ),
    "transformation_attribution": (
        "L7 transformation_attribution Shapley table.",
        (
            "Per-pipeline Shapley contributions to forecast skill. "
            "Active when ``transformation_attribution`` is in the L7 "
            "DAG (typically alongside multi-cell sweeps over alternative "
            "L3 transforms)."
        ),
        "Active when transformation_attribution op is in the L7 DAG.",
    ),
    "regime_metrics": (
        "Regime-conditional metrics.",
        (
            "Metric breakdowns by L1.G regime classification. "
            "Default-on when L1.G regime is non-pooled (i.e. "
            "regime-conditional analysis is intended)."
        ),
        "Active when L1.G regime is non-pooled.",
    ),
    "state_metrics": (
        "State-level metrics for FRED-SD geographic studies.",
        (
            "Per-state metric breakdowns. Default-on when L1.D "
            "geography is state-level (FRED-SD pipelines)."
        ),
        "Active when L1.D geography is state-level (FRED-SD).",
    ),
    "combination_weights": (
        "Ensemble weights from L4 combine ops.",
        (
            "Per-origin per-member weights produced by L4 combine ops "
            "(equal_weighted / dmsfe / inverse_msfe / mallows_cp / etc.). "
            "Active when ensemble combine ops are in the L4 DAG."
        ),
        "Active when ensemble combine ops are in the L4 DAG.",
    ),
    "diagnostics_l1_5": (
        "L1.5 diagnostic outputs (sample coverage / stationarity / outlier audit).",
        (
            "JSON + figures from the L1.5 sub-layer. Active when L1.5 "
            "is enabled in the recipe."
        ),
        "Active when L1.5 is enabled.",
    ),
    "diagnostics_l2_5": (
        "L2.5 diagnostic outputs (cleaning effect summaries).",
        (
            "JSON + figures from the L2.5 sub-layer. Active when L2.5 "
            "is enabled."
        ),
        "Active when L2.5 is enabled.",
    ),
    "diagnostics_l3_5": (
        "L3.5 diagnostic outputs (factor / lag / selection inspection).",
        (
            "JSON + figures from the L3.5 sub-layer. Active when L3.5 "
            "is enabled."
        ),
        "Active when L3.5 is enabled.",
    ),
    "diagnostics_l4_5": (
        "L4.5 diagnostic outputs (in-sample fit / window stability / tuning history).",
        (
            "JSON + figures from the L4.5 sub-layer. Active when L4.5 "
            "is enabled."
        ),
        "Active when L4.5 is enabled.",
    ),
    "diagnostics_all": (
        "Every active diagnostic layer's output (convenience option).",
        (
            "Convenience flag: enables ``diagnostics_l{1..4}_5`` "
            "simultaneously when the corresponding diagnostic layer "
            "is active. Recommended default for first-time runs."
        ),
        "Default convenience option for full-diagnostic runs.",
    ),
    "model_artifacts": (
        "Pickled / joblib model objects.",
        (
            "Serialised fitted estimators (one per (model, origin) "
            "pair). Default-off because model objects can be large; "
            "enable for downstream prediction without re-fitting."
        ),
        "When downstream prediction without re-fitting is needed.",
    ),
}
for option, (summary, desc, when) in _SAVED_DOCS.items():
    register(_e("L8_B_saved_objects", "saved_objects", option, summary, desc, when,
        related=tuple(k for k in _SAVED_DOCS if k != option)[:4]))


# ---------------------------------------------------------------------------
# L8.B model_artifacts_format -- joblib / pickle / onnx / pmml
# ---------------------------------------------------------------------------

register(
    _e("L8_B_saved_objects", "model_artifacts_format", "joblib",
       "Default sklearn / xgboost serialisation via joblib.",
       (
           "Optimised for numpy-array-heavy estimators (sklearn / "
           "xgboost / lightgbm). Smaller and faster than plain pickle "
           "for typical sklearn fitted-model graphs."
       ),
       "Default; broad compatibility across sklearn / xgboost / lightgbm.",
       related=("pickle", "onnx", "pmml")),
    _e("L8_B_saved_objects", "model_artifacts_format", "pickle",
       "Plain Python pickle (less efficient than joblib).",
       (
           "Compatibility option for older toolchains or non-sklearn "
           "estimators that don't benefit from joblib's array "
           "optimisation. Larger files but maximally portable across "
           "Python versions."
       ),
       "Compatibility with older toolchains.",
       related=("joblib", "onnx", "pmml")),
    _e("L8_B_saved_objects", "model_artifacts_format", "onnx",
       "ONNX export (where supported by the family).",
       (
           "Open Neural Network Exchange format. Cross-language "
           "deployment (C++ / C# / Java / JS runtimes) and faster "
           "inference than the native sklearn pickle. Supported for "
           "sklearn / xgboost / lightgbm / pytorch families; raises "
           "if the active L4 family lacks an ONNX exporter."
       ),
       "Cross-language deployment; production inference servers.",
       when_not_to_use="Models without ONNX support (BVAR, DFM, MRF, custom callables).",
       references=(_REF_DESIGN_L8, Reference(citation="ONNX specification.", url="https://onnx.ai/")),
       related=("joblib", "pmml")),
    _e("L8_B_saved_objects", "model_artifacts_format", "pmml",
       "PMML export (PMML-compatible families only).",
       (
           "Predictive Model Markup Language; XML-based exchange "
           "format primarily used in enterprise / Java deployments. "
           "Supported for linear / tree-family models via "
           "``sklearn2pmml``."
       ),
       "Enterprise / Java deployment.",
       when_not_to_use="Modern ML deployment -- ONNX is more widely supported.",
       references=(_REF_DESIGN_L8, Reference(citation="PMML 4.4 specification.", url="https://dmg.org/pmml/v4-4/GeneralStructure.html")),
       related=("joblib", "onnx")),
)


# ---------------------------------------------------------------------------
# L8.C provenance_fields -- 14 fields default-all
# ---------------------------------------------------------------------------

_PROV_DOCS: dict[str, tuple[str, str, str]] = {
    "recipe_yaml_full": (
        "Full recipe YAML embedded in the manifest.",
        (
            "Verbatim copy of the recipe as supplied by the user "
            "(post-canonicalisation). Required for ``replicate()`` "
            "to reconstruct the exact run; without it the manifest "
            "is descriptive but not replayable."
        ),
        "Default-on; required for replication.",
    ),
    "recipe_hash": (
        "SHA-256 hash of the canonicalised recipe.",
        (
            "Cheap consistency check. Compares against the recipe "
            "hash from the original run during ``replicate()``; "
            "mismatch triggers a hard error before any compute is "
            "wasted."
        ),
        "Default-on; cheap consistency check.",
    ),
    "package_version": (
        "macroforecast version string.",
        (
            "From ``macroforecast.__version__``. Lets ``replicate()`` "
            "warn the user when the manifest was produced by a "
            "different package version."
        ),
        "Default-on.",
    ),
    "python_version": (
        "Python interpreter version (3-tuple major.minor.patch).",
        (
            "From ``sys.version_info``. Lets ``replicate()`` warn "
            "when running on a different interpreter."
        ),
        "Default-on.",
    ),
    "r_version": (
        "R version (when R-backed steps are active).",
        (
            "Captured via ``rpy2`` when any L3 / L4 / L6 / L7 op "
            "calls into R. Optional -- only emitted when R is "
            "actually used."
        ),
        "Recipes that call R (e.g. for arima or robust statistics).",
    ),
    "julia_version": (
        "Julia version (when Julia-backed steps are active).",
        (
            "Captured via ``julia`` Python bridge when any pipeline "
            "step calls into Julia. Optional."
        ),
        "Recipes that call Julia.",
    ),
    "git_commit_sha": (
        "Git commit SHA of the active checkout.",
        (
            "From ``git rev-parse HEAD``. Default-on when the run "
            "executes inside a git working tree; provides exact "
            "code traceability."
        ),
        "Default-on when the run executes inside a git tree.",
    ),
    "git_branch_name": (
        "Git branch name.",
        (
            "From ``git rev-parse --abbrev-ref HEAD``. Default-on "
            "with git_commit_sha; documents which feature branch "
            "produced the run."
        ),
        "Default-on with git_commit_sha.",
    ),
    "runtime_environment": (
        "Hostname / OS / CPU summary string.",
        (
            "Captured at run start; useful for diagnosing performance "
            "regressions across machines (laptop vs cluster)."
        ),
        "Default-on.",
    ),
    "runtime_duration": (
        "Wall-clock duration per cell.",
        (
            "Per-cell timings; useful for cost-tracking and detecting "
            "slow cells in a sweep."
        ),
        "Default-on; useful for cost-tracking.",
    ),
    "random_seed_used": (
        "Resolved random seeds (L0 + per-cell propagation).",
        (
            "The exact seed values used by every numpy / sklearn / "
            "torch RNG. Required for bit-exact replication; the "
            "seed-propagation system in v0.2 ensures every "
            "non-deterministic op receives a deterministic seed."
        ),
        "Default-on; required for bit-exact replication.",
    ),
    "data_revision_tag": (
        "FRED vintage / data revision tag.",
        (
            "When the L1 raw is FRED-MD / -QD / -SD, captures the "
            "vintage tag (e.g. ``2024-09``) so future re-runs against "
            "an updated FRED snapshot can detect that the input data "
            "has revised."
        ),
        "Default-on when raw data is FRED-MD / -QD / -SD.",
    ),
    "dependency_lockfile": (
        "Lockfile contents (pip freeze / poetry.lock / conda env).",
        (
            "Verbatim contents of the active environment's lockfile. "
            "Critical for reproducing the same package versions on a "
            "different machine."
        ),
        "Default-on; needed for environment replication.",
    ),
    "cell_resolved_axes": (
        "Per-cell resolved axis values from sweep expansion.",
        (
            "For sweep-expanded cells, records the (axis → value) "
            "mapping that produced each cell. Without this field, "
            "interpreting which cell ran which configuration "
            "requires re-expanding the sweep."
        ),
        "Default-on when sweeps are active.",
    ),
}
for option, (summary, desc, when) in _PROV_DOCS.items():
    register(_e("L8_C_provenance", "provenance_fields", option, summary, desc, when,
        related=tuple(k for k in _PROV_DOCS if k != option)[:4]))


# ---------------------------------------------------------------------------
# L8.C manifest_format -- json / yaml / json_lines
# ---------------------------------------------------------------------------

register(
    _e("L8_C_provenance", "manifest_format", "json",
       "Manifest written as a single JSON document (default).",
       (
           "Default. Round-trips cleanly into Python / JS / R; "
           "preserves nested structure. The natural choice for "
           "every consumer that uses ``macroforecast.replicate``."
       ),
       "Default; round-trips cleanly.",
       related=("yaml", "json_lines")),
    _e("L8_C_provenance", "manifest_format", "yaml",
       "Manifest written as YAML.",
       (
           "Hand-readable alternative to JSON. Commits cleaner in "
           "git diffs; useful when manifests are expected to be "
           "reviewed by humans (paper supplementary materials, "
           "code reviews of recipe changes)."
       ),
       "Hand-readable manifests; paper supplementary materials.",
       related=("json", "json_lines")),
    _e("L8_C_provenance", "manifest_format", "json_lines",
       "Manifest written as JSONL (one cell per line).",
       (
           "Streaming-friendly format: each cell becomes one JSON "
           "object on its own line. Sweep manifests with thousands "
           "of cells stay parseable line-by-line without loading "
           "the entire manifest into memory."
       ),
       "Sweep manifests with thousands of cells; streaming consumers.",
       related=("json", "yaml")),
)


# ---------------------------------------------------------------------------
# L8.D artifact_granularity -- 5 layouts
# ---------------------------------------------------------------------------

_GRAN_DOCS: dict[str, tuple[str, str, str]] = {
    "per_cell": (
        "One sub-directory per sweep cell (default).",
        (
            "Default. Each cell gets its own ``cell_NNN/`` directory "
            "containing the cell's full set of artifacts. Isolates "
            "every cell's output for clean per-cell inspection."
        ),
        "Default; isolates each cell's artifacts.",
    ),
    "per_target": (
        "Group cells by target variable (one directory per target).",
        (
            "When the sweep varies over multiple targets, this "
            "groups the artifacts by target rather than by cell. "
            "Useful when downstream analysis is per-target."
        ),
        "Multi-target studies where target-grouping aids browsing.",
    ),
    "per_horizon": (
        "Group cells by forecast horizon (one directory per horizon).",
        (
            "When the sweep varies over multiple horizons, groups "
            "by horizon. Useful for horizon-by-horizon analysis."
        ),
        "Multi-horizon sweeps.",
    ),
    "per_target_horizon": (
        "Group cells by (target, horizon) pair.",
        (
            "Combines ``per_target`` and ``per_horizon`` for sweeps "
            "that vary across both axes."
        ),
        "Studies sweeping across both target and horizon.",
    ),
    "flat": (
        "Single flat directory; cells distinguished by filename suffix.",
        (
            "All cells write into one directory; cell IDs become "
            "filename suffixes. Useful for sweeps with thousands of "
            "small artifacts where per-cell directory creation is "
            "wasteful."
        ),
        "Sweeps with thousands of small artifacts.",
    ),
}
for option, (summary, desc, when) in _GRAN_DOCS.items():
    register(_e("L8_D_artifact_granularity", "artifact_granularity", option, summary, desc, when,
        related=tuple(k for k in _GRAN_DOCS if k != option)))


# ---------------------------------------------------------------------------
# L8.D naming_convention -- cell_id / descriptive / recipe_hash / custom
# ---------------------------------------------------------------------------

register(
    _e("L8_D_artifact_granularity", "naming_convention", "cell_id",
       "Use the cell numeric id (cell_001/, cell_002/, ...).",
       (
           "Default. Stable and short; sorts naturally in directory "
           "listings. Recommended for production sweeps where the "
           "exact axis-value mapping is captured separately by the "
           "manifest's ``cell_resolved_axes`` field."
       ),
       "Default; stable, short.",
       related=("descriptive", "recipe_hash", "custom")),
    _e("L8_D_artifact_granularity", "naming_convention", "descriptive",
       "Use a descriptive template combining the cell's resolved axes.",
       (
           "Generates names like "
           "``ridge_log_diff_h1_seed42/`` from the cell's resolved "
           "axes. Human-readable; useful when humans browse the "
           "output directory directly. Long names can hit filesystem "
           "limits for wide sweeps."
       ),
       "When humans browse the output directory.",
       when_not_to_use="Wide sweeps with many axes -- names exceed filesystem limits.",
       related=("cell_id", "custom")),
    _e("L8_D_artifact_granularity", "naming_convention", "recipe_hash",
       "Use the per-cell recipe hash as the directory name.",
       (
           "Reproducibility-first naming: directory names are the "
           "first 8 chars of the cell's recipe hash. Deterministic "
           "across runs (a re-run with the same recipe produces "
           "the same directory names), but unreadable to humans."
       ),
       "Reproducibility-first naming; deterministic across runs.",
       related=("cell_id", "descriptive")),
    _e("L8_D_artifact_granularity", "naming_convention", "custom",
       "User-supplied template via ``leaf_config.naming_template``.",
       (
           "Bespoke directory layouts: the template string can "
           "interpolate any cell-resolved axis (e.g. "
           "``{model_family}_{horizon}h_{seed}``) plus the cell's "
           "numeric id."
       ),
       "Bespoke directory layouts.",
       related=("descriptive", "cell_id")),
)
