"""Pipeline specification: schema, t-code -> target resolution, and validation.

Stage 0 of the comprehensive POOS pipeline. Defines the declarative configuration
objects (frozen dataclasses) and the validating generator ``pipeline_spec``. The
execution (``run_pipeline``), interpretation, and reporting layers build on these.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd


# t-code (FRED-MD/QD integration order) -> (forecast_policy, target_transform).
# The forecast object is the h-period CUMULATION (direct_average), not the raw
# single-period transform. I(2) price/level series (6, 3) are reduced to the
# first-difference object averaged over the horizon (the standard convention:
# forecast average inflation, not the twice-differenced object).
TCODE_TARGET_MAP: dict[int, tuple[str, str]] = {
    1: ("direct", "level"),
    2: ("direct_average", "change"),
    3: ("direct_average", "change"),
    4: ("direct", "level"),  # log(x): forecast the level directly (log is a monotone
                            # reparametrisation; no distinct log-level target_transform exists)
    5: ("direct_average", "log_growth"),
    6: ("direct_average", "log_growth"),
    7: ("direct_average", "growth"),
}


@dataclass(frozen=True)
class TargetSpec:
    """A forecast target and how its forecast object is defined.

    ``name`` is the panel column to forecast. ``transform`` and ``policy`` may
    be left as ``None`` so FRED transformation-code metadata chooses the
    conventional forecast object. For example, a FRED-MD growth-rate target
    resolves to a direct-average growth forecast rather than a raw level
    forecast. ``annualize`` affects reporting scale only, while ``reduce_i2``
    keeps the package's convention for I(2) series by forecasting the
    first-difference object.

    Returns
    -------
    TargetSpec
        Immutable target declaration consumed by ``pipeline_spec(...)`` and
        resolved to ``ResolvedTarget`` during execution.

    Example
    -------
    >>> from macroforecast.pipeline import TargetSpec
    >>> target = TargetSpec(name="INDPRO", annualize=True)
    """

    name: str
    transform: str | None = None  # None -> derive from the t-code
    policy: str | None = None     # None -> derive from the transform/mapping
    annualize: bool = False       # reporting scale (e.g. monthly growth x12)
    reduce_i2: bool = True        # I(2) -> first-difference object (convention)


@dataclass(frozen=True)
class ResolvedTarget:
    """A target with its forecast policy and transform resolved."""

    name: str
    policy: str
    transform: str
    tcode: int | None
    annualize: bool


@dataclass(frozen=True)
class InterpretSpec:
    """ML interpretation request for an arm (deferred, multi-method)."""

    methods: tuple[str, ...] = ()
    which_fit: str = "auto"       # "auto"|"final"|"origin_mean"
    background: int | None = None
    top_k: int | None = 20
    on_targets: tuple[str, ...] | None = None
    shap_kind: str = "auto"


@dataclass(frozen=True)
class Arm:
    """A target-agnostic configuration: preprocessing + features + a single model.

    An arm is NOT itself a cell. Applied to a target and a horizon it forms one
    cell (executed by one ``run()`` call); in the evaluation it appears as exactly
    one contender (one arm = one contender).
    """

    name: str
    model: Any
    preprocessing: Any | None = None
    preprocessing_policy: Any | None = None
    features: Any | None = None
    feature_policy: Any | None = None
    params: Mapping[str, Any] | None = None
    model_selection: Any | None = None
    model_selection_metric: str = "mse"
    # Optional per-arm window override. When set, this arm runs on its own window
    # (e.g. the autoregression on a no-validation window) instead of the shared
    # spec window, so each cell of the pipeline is managed independently.
    window: Any | None = None
    interpret: InterpretSpec | tuple[str, ...] | None = None
    is_benchmark: bool = False
    # Clark-West is only valid when the benchmark is NESTED within this arm's model
    # (the benchmark is a parameter restriction of the larger model, e.g. AR nested
    # in ARDI). Declare it so the evaluator emits CW only where it is licensed;
    # otherwise CW is silently invalid. Diebold-Mariano is reported regardless.
    nested_in_benchmark: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CombinationContender:
    """A forecast combination that becomes an additional contender."""

    name: str
    method: str
    over: tuple[str, ...] | str = "all"
    by: tuple[str, ...] = ("target", "horizon")
    params: Mapping[str, Any] | None = None
    weight_window: int | None = None
    shrink_to_equal: float | None = None


#: Test names ``EvalSpec.tests`` currently wires into the evaluation. The pairwise
#: benchmark-comparison names include the historical ``dm``/``cw`` plus GW,
#: nested encompassing, directional-accuracy, and Giacomini-Rossi fluctuation
#: tests.
#: ``berkowitz``/``pit_autocorr``/``coverage`` are the PIT-based calibration tests
#: (Phase 1 density pipeline): they populate ``report.calibration`` instead of
#: ``report.significance`` and are never on by default (see ``EvalSpec.tests``'s
#: docstring below).
SUPPORTED_EVAL_TESTS: frozenset[str] = frozenset(
    {
        "dm", "cw", "gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr", "mz", "mcs",
        "spa", "rc", "stepm", "uspa", "aspa", "berkowitz", "pit_autocorr",
        "coverage",
    }
)
_ARCH_BACKED_EVAL_TESTS: frozenset[str] = frozenset({"spa", "rc", "stepm"})

#: The subset of :data:`SUPPORTED_EVAL_TESTS` that are PIT-based calibration
#: diagnostics: they run in ``pipeline/evaluate.py::calibration_table`` and land
#: in ``PipelineReport.calibration``, not ``significance``/``mcs``.
CALIBRATION_EVAL_TESTS: frozenset[str] = frozenset({"berkowitz", "pit_autocorr", "coverage"})


def _arch_available() -> bool:
    return importlib.util.find_spec("arch") is not None


_EVAL_TEST_OPTION_TARGETS: Mapping[str, tuple[str, str]] = {
    "dm": ("macroforecast.tests", "dm_test"),
    "cw": ("macroforecast.tests", "clark_west_test"),
    "gw": ("macroforecast.tests", "giacomini_white_test"),
    "enc_new": ("macroforecast.tests", "enc_new_test"),
    "enc_t": ("macroforecast.tests", "enc_t_test"),
    "pt": ("macroforecast.tests", "directional_accuracy_test"),
    "hm": ("macroforecast.tests", "directional_accuracy_test"),
    "ag": ("macroforecast.tests", "directional_accuracy_test"),
    "gr": ("macroforecast.tests", "conditional_predictive_ability_test"),
    "mz": ("macroforecast.tests", "mincer_zarnowitz_test"),
    "mcs": ("macroforecast.tests", "model_confidence_set"),
    "spa": ("macroforecast.tests", "superior_predictive_ability_test"),
    "rc": ("macroforecast.tests", "reality_check_test"),
    "stepm": ("macroforecast.tests", "stepm_test"),
    "uspa": ("macroforecast.tests", "multi_horizon_spa_test"),
    "aspa": ("macroforecast.tests", "multi_horizon_spa_test"),
    "berkowitz": ("macroforecast.tests", "density_interval_tests"),
    "pit_autocorr": ("macroforecast.tests", "density_interval_tests"),
    "coverage": ("macroforecast.tests", "interval_coverage_test"),
}


def _accepted_eval_test_options(test_name: str) -> frozenset[str]:
    """Keyword option names accepted by the public callable backing a pipeline test."""

    import importlib

    try:
        module_name, function_name = _EVAL_TEST_OPTION_TARGETS[test_name]
    except KeyError:
        return frozenset()
    function = getattr(importlib.import_module(module_name), function_name)
    signature = inspect.signature(function)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return frozenset()
    return frozenset(
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )


def _validate_eval_test_options(evaluation: "EvalSpec") -> None:
    """Fail fast when ``EvalSpec.test_options`` cannot be consumed safely."""

    requested = set(evaluation.tests)
    option_tests = set(evaluation.test_options)
    missing = option_tests - requested
    if missing:
        raise ValueError(
            f"evaluation.test_options contains option block(s) for test(s) "
            f"{sorted(missing)} that are not present in evaluation.tests "
            f"{sorted(requested)}"
        )
    for test_name, options in evaluation.test_options.items():
        accepted = _accepted_eval_test_options(str(test_name))
        unknown = set(options) - accepted
        if unknown:
            raise ValueError(
                f"evaluation.test_options[{test_name!r}] contains unsupported "
                f"option name(s) {sorted(unknown)}; accepted options for "
                f"{test_name!r} are {sorted(accepted)}"
            )


_DEFAULT_EVAL_BY = ("target", "horizon")
_DEFAULT_PRIMARY_AXIS = "contender"


def _validate_unimplemented_eval_fields(evaluation: "EvalSpec") -> None:
    """Reject declared EvalSpec fields that are not wired into evaluation yet."""

    if tuple(evaluation.by) != _DEFAULT_EVAL_BY:
        raise ValueError(
            "evaluation.by is not implemented; use test_options / file an issue"
        )
    if str(evaluation.primary_axis) != _DEFAULT_PRIMARY_AXIS:
        raise ValueError(
            "evaluation.primary_axis is not implemented; use test_options / file an issue"
        )
    if evaluation.multiple_testing is not None:
        raise ValueError(
            "evaluation.multiple_testing is not implemented; use test_options / file an issue"
        )


def _parse_subsample_date(value: Any, *, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a parseable date string") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a parseable date string")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _normalize_subsamples(
    subsamples: Mapping[str, "SubsampleWindow"] | None,
) -> Mapping[str, "SubsampleWindow"] | None:
    if not subsamples:
        return None
    names = [str(name) for name in subsamples]
    if any(not name for name in names):
        raise ValueError("evaluation.subsamples names must be nonempty")
    if len(set(names)) != len(names):
        raise ValueError("evaluation.subsamples names must be unique")

    normalized: dict[str, SubsampleWindow] = {}
    for raw_name, window in subsamples.items():
        name = str(raw_name)
        if not isinstance(window, SubsampleWindow):
            raise ValueError(
                "evaluation.subsamples values must be SubsampleWindow instances"
            )
        start = _parse_subsample_date(window.start, label=f"subsample {name!r} start")
        end = _parse_subsample_date(window.end, label=f"subsample {name!r} end")
        if start is not None and end is not None and start >= end:
            raise ValueError(
                f"evaluation.subsamples[{name!r}] start must be before end"
            )
        excludes: list[tuple[str, str]] = []
        for idx, bounds in enumerate(window.exclude):
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, (str, bytes))
                or len(bounds) != 2
            ):
                raise ValueError(
                    f"evaluation.subsamples[{name!r}].exclude[{idx}] must have "
                    "exactly two date bounds"
                )
            raw_ex_start, raw_ex_end = bounds
            ex_start = _parse_subsample_date(
                raw_ex_start, label=f"subsample {name!r} exclude[{idx}] start"
            )
            ex_end = _parse_subsample_date(
                raw_ex_end, label=f"subsample {name!r} exclude[{idx}] end"
            )
            assert ex_start is not None and ex_end is not None
            if ex_start >= ex_end:
                raise ValueError(
                    f"evaluation.subsamples[{name!r}].exclude[{idx}] start "
                    "must be before end"
                )
            excludes.append((str(raw_ex_start), str(raw_ex_end)))
        normalized[name] = SubsampleWindow(
            start=None if window.start is None else str(window.start),
            end=None if window.end is None else str(window.end),
            exclude=tuple(excludes),
        )
    return normalized


def _normalize_eval_spec(evaluation: "EvalSpec") -> "EvalSpec":
    """Return the EvalSpec shape consumed by the evaluator."""

    _validate_unimplemented_eval_fields(evaluation)
    test_options = {
        str(test_name): dict(options)
        for test_name, options in (evaluation.test_options or {}).items()
    }
    if evaluation.dm_kwargs:
        dm_options = dict(evaluation.dm_kwargs)
        dm_options.update(test_options.get("dm", {}))
        test_options["dm"] = dm_options
    return replace(
        evaluation,
        test_options=test_options,
        subsamples=_normalize_subsamples(evaluation.subsamples),
    )


@dataclass(frozen=True)
class SubsampleWindow:
    """Evaluation-window filter applied to forecast target dates.

    Bounds are inclusive date strings. ``exclude`` removes inclusive date ranges
    after the optional start/end bounds are applied.
    """

    start: str | None = None
    end: str | None = None
    exclude: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvalSpec:
    """Automatic evaluation and significance-testing configuration.

    ``metrics`` entries are either a name resolved through the metric registry
    (:func:`macroforecast.metrics.get_metric`, e.g. ``"mae"``, ``"mape"``) or a
    callable ``metric(y_true, y_pred) -> float`` named by its ``__name__``.
    ``accuracy_table`` computes one column per listed metric, per contender, on
    the same pairwise-vs-benchmark sample it always has. The three defaults
    (``"rmse"``, ``"relative_mse"``, ``"r2_oos"``) keep their existing
    benchmark-relative formulas regardless of how they are requested.

    ``loss`` is a per-observation loss ``loss(y_true, y_pred) -> ndarray`` used
    by the Diebold-Mariano loss differential and the Model Confidence Set's loss
    matrix; ``None`` (default) is squared error, the prior, only behavior. Since
    the Clark-West adjustment is derived under quadratic loss, setting a custom
    ``loss`` makes ``significance_table`` skip CW (with a ``UserWarning``)
    rather than compute it against the wrong loss.

    ``tests`` lists which significance tests actually run; unsupported names
    raise at :func:`pipeline_spec` build time (see ``SUPPORTED_EVAL_TESTS``).
    Pairwise contender-vs-benchmark tests are ``"dm"``, ``"cw"``, ``"gw"``,
    ``"enc_new"``, ``"enc_t"``, ``"gr"``, and ``"mz"``. ``"mz"`` is the
    Mincer-Zarnowitz actual-on-forecast rationality regression. ``"pt"``, ``"hm"``, and
    ``"ag"`` are directional-accuracy tests for the contender's own sign
    forecasts, evaluated on the same benchmark-aligned sample for consistency
    with the pairwise tests. Joint multi-horizon pairwise tests are ``"uspa"``
    and ``"aspa"``; they require at least two horizons and use ``"joint"`` as
    their significance-table horizon sentinel. Full-set benchmark comparisons are ``"mcs"``,
    ``"spa"``, ``"rc"``, and ``"stepm"``; they populate ``PipelineReport.mcs``.
    ``"spa"``, ``"rc"``, and ``"stepm"`` require the ``arch`` extra
    (``pip install "macroforecast[arch]"``).
    ``"berkowitz"``/``"pit_autocorr"``/``"coverage"`` are PIT-based calibration
    diagnostics (Phase 1 density pipeline) -- they populate
    ``PipelineReport.calibration`` rather than ``significance``/``mcs`` and,
    like every other test name, are opt-in only (absent from the default).
    ``test_options`` maps a requested test name to keyword options for that
    test's underlying public callable. Option blocks are validated when
    :func:`pipeline_spec` is built: the key must appear in ``tests`` and every
    option name must be accepted by that test's callable.

    Density/interval accuracy metrics -- ``"crps"``, ``"gaussian_nll"``,
    ``"log_score"``, ``"negative_log_score"``, ``"qlike"``, ``"pinball_loss"``,
    ``"coverage_rate"``, ``"interval_width"``, ``"interval_score"`` -- are
    requested the SAME way as any other ``metrics`` entry; they land in
    ``PipelineReport.density`` instead of ``accuracy`` because they need a
    ``variance_prediction``/``quantile_predictions`` column rather than plain
    ``(y_true, y_pred)`` (see ``macroforecast.metrics.metric_kind``). Requesting
    one on a forecast frame that carries no such column raises the same
    actionable ``ValueError`` :func:`macroforecast.metrics.evaluate_forecasts`
    already raises. Absent from the defaults, so a default-EvalSpec run never
    computes them.

    ``calibration_alpha`` is the significance level for the calibration tests
    above (Berkowitz LR test, PIT autocorrelation, and the nominal coverage
    checked by the ``"coverage"`` test); it does not affect ``mcs_alpha``.

    ``subsamples`` optionally maps names to :class:`SubsampleWindow` values.
    These are evaluation-window splits of an already-produced POOS forecast
    frame: target-date rows are filtered before scoring and testing, without
    refitting models or creating new forecast cells.
    """

    benchmark: str
    metrics: tuple[str | Callable[..., float], ...] = ("rmse", "relative_mse", "r2_oos")
    tests: tuple[str, ...] = ("dm", "cw", "mcs")
    by: tuple[str, ...] = ("target", "horizon")
    primary_axis: str = "contender"
    cw_for_nested: bool = True
    mcs_alpha: float = 0.10
    mcs_method: str = "iterative"
    multiple_testing: str | None = None
    subsamples: Mapping[str, SubsampleWindow] | None = None
    dm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    loss: Callable[[Any, Any], Any] | None = None
    test_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    calibration_alpha: float = 0.05


@dataclass(frozen=True)
class PipelineSpec:
    """Validated, frozen configuration produced by :func:`pipeline_spec`."""

    data: Any
    targets: tuple[ResolvedTarget, ...]
    horizons: tuple[int, ...]
    window: Any
    arms: tuple[Arm, ...]
    evaluation: EvalSpec
    # Spec-level SHARED preprocessing (freeze/vary contract): preprocessing is a
    # shared, window-dependent stage applied identically to every arm (arms vary
    # only in features + model). Window-invariant ops (t-codes) are deterministic
    # and leak-free in any window; window-dependent ops (outliers, EM imputation,
    # standardisation, factors) refit on the policy's cadence (use update="on_retrain"
    # to match the paper's periodic refresh rather than every_origin). An arm may
    # still override with its own preprocessing when a genuinely different
    # transformation is part of the comparison.
    preprocessing: Any | None = None
    preprocessing_policy: Any | None = None
    combinations: tuple[CombinationContender, ...] = ()
    # Pipeline runs default to forecast/evaluation artifacts only. Enabling this
    # writes one pickle + JSON manifest per fitted model/origin/horizon under
    # ``model_store``; large horse races can create many files.
    save_models: bool = False
    model_store: str = "trained_model"
    # When set, each (target, arm, horizon) forecast cell persists its lean
    # forecast records incrementally under
    # ``<checkpoint_dir>/<target>__<arm>/h<h>/`` so a crashed multi-hour POOS run
    # resumes without recomputing finished origins. None (default) disables
    # checkpointing and is byte-for-byte the prior behavior.
    checkpoint_dir: str | None = None
    # When set, run_pipeline stores each digestible (target, horizon, arm) cell's
    # master-frame rows under ``<result_store>/cells`` and reuses matching cells
    # across separate run_pipeline calls. None (default) disables this path and
    # preserves the existing in-run execution graph byte-for-byte.
    result_store: str | None = None
    # Native fan-out: when >1, the (arm x target x horizon) cells run across a
    # process pool. Default 1 keeps the sequential, cross-horizon EM-sharing path
    # byte-for-byte unchanged. The parallel path is deterministic (every cell uses
    # the effective meta random seed and derived per-arm model seeds) and produces
    # forecasts numerically identical to ``n_jobs=1`` except where stochastic model
    # or selection streams are intentionally changed by ``seed``. Memory scales
    # with ``n_jobs`` because every worker holds the data panel.
    n_jobs: int = 1
    # Model-internal thread budget per cell worker, set by the AUTO allocator.
    # In parallel mode (n_jobs>1) each worker pins its tree-ensemble (RF/GBM/XGB/
    # LGBM) internal n_jobs to this value so cell_workers * model_threads <= cores
    # and the CPU is saturated without oversubscription. Default 1 (serial mode and
    # any explicit-int n_jobs leave each worker single-threaded internally, the
    # prior behavior). Only changes thread COUNT, never the numerical result.
    model_threads: int = 1
    # When set, a shared on-disk ``PreprocessorStore`` rooted at this directory lets
    # parallel cells reuse each per-(spec, target, origin) FittedPreprocessor instead
    # of recomputing it. Both backends construct a store pointing at this directory;
    # in parallel mode every worker process points at the SAME directory and shares
    # the persisted fits through the filesystem (the ~36x EM dedup).
    #
    # Three states:
    #   - an explicit str path: use it (persists across runs if reused).
    #   - None (default): "not configured". When n_jobs==1 this is a no-op, byte-
    #     for-byte the original behavior (no disk store; the serial backend already
    #     shares fits via its own in-memory cache). When n_jobs>1, run_pipeline/
    #     run_arms AUTO-CREATE a run-scoped temporary directory for the duration of
    #     the run (removed afterward) so parallel workers recover the cross-arm/
    #     cross-horizon EM dedup that they would otherwise silently lose (each
    #     worker previously got preprocessing_cache=None with no fallback).
    #   - False: explicit opt-out. Never auto-create a store even when n_jobs>1,
    #     matching the pre-this-change parallel behavior (each worker recomputes
    #     its own EM, no shared cache of any kind).
    # The store is namespaced by each arm's effective preprocessing_policy, so
    # sharing one directory (explicit or auto-created) across specs/arms that
    # resolve to different scopes never cross-contaminates (see
    # preprocessing/cache.py and pipeline/run.py::_effective_preprocessing_policy).
    preprocessing_cache_dir: str | Literal[False] | None = None
    seed: int | None = 42
    provenance: Mapping[str, Any] = field(default_factory=dict)
    # "full" (default): PipelineReport.provenance additionally carries an
    # "environment" block (git SHA/branch/dirty, Python/platform, pinned core
    # dependency versions -- via output.collect_provenance), a "data" identity
    # descriptor (dataset/source_family/vintage, panel shape/date range, and a
    # content fingerprint), and a "spec_echo" of the resolved spec's key choices
    # (targets/policies, horizons, window cutoffs, arms/models, seed, n_jobs).
    # "basic" keeps exactly the pre-existing provenance dict shape (package_
    # version/seed/targets/horizons/arms/benchmark/combinations) with none of
    # the three additive blocks -- for callers who build the dict themselves
    # from a lighter-weight report, or who find the extra collection cost (git/
    # env probing, one panel fingerprint) unnecessary. Distinct from the
    # ``provenance`` field/kwarg above, which carries caller-supplied NOTES
    # (e.g. the "warnings" list below) merged into whichever shape this
    # produces -- ``provenance`` is the mapping payload, ``provenance_level``
    # is only the additive-blocks toggle.
    provenance_level: Literal["full", "basic"] = "full"
    # Per-(arm, target) forecast-policy overrides, currently used by
    # on_unsupported_direct="reroute" to run guarded direct-like cells as recursive
    # without mutating the target policy for direct-capable arms.
    policy_overrides: Mapping[tuple[str, str], str] = field(default_factory=dict)



@dataclass
class PipelineReport:
    """Standard pipeline output (mutable: interpretation is filled in later)."""

    forecasts: "Any"
    accuracy: "Any"
    significance: "Any"
    mcs: "Any"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    leakage_audit: Mapping[str, Any] = field(default_factory=dict)
    interpretation: Mapping[str, Any] | None = None
    model_store: str = "trained_model"
    spec: "Any" = None
    # Cells (target, arm, horizon-group) that raised during the managed run. Each
    # entry records the cell identity and the error text. A non-empty list means
    # some arms are absent from the evaluation because their run() failed (the rest
    # of the set still ran). Empty (default) means every cell completed.
    failed_cells: "Sequence[Mapping[str, Any]]" = field(default_factory=tuple)
    # (target, horizon) cells that RAN without error but produced ZERO forecast
    # rows. These are NOT failures (no exception) yet are silently absent from the
    # evaluation, so a missing long-horizon output would otherwise be invisible.
    # Each entry is {"target", "horizon", "arms": [...]} listing the arm(s) that
    # yielded no rows for that cell. Empty (default) means every cell had rows.
    empty_cells: "Sequence[Mapping[str, Any]]" = field(default_factory=tuple)
    # Density/interval accuracy metrics (crps, gaussian_nll, pinball_loss, ...)
    # per (target, horizon, contender) -- see ``pipeline/evaluate.py::density_table``.
    # ``None`` only for a ``PipelineReport`` built by hand without this field (a
    # live ``run_pipeline``/``rescore`` always passes an actual, possibly-empty,
    # frame). Appended at the end of the dataclass (rather than near ``mcs``) so
    # any positional ``PipelineReport(...)`` construction elsewhere is unaffected.
    density: "Any" = None
    # PIT-based calibration diagnostics (Berkowitz, PIT autocorrelation, interval
    # coverage) per (target, horizon, contender) -- see
    # ``pipeline/evaluate.py::calibration_table``. Same ``None``-default contract
    # as ``density`` above.
    calibration: "Any" = None

    def to_frame(self) -> "Any":
        """Return the master forecast frame."""
        return self.forecasts

# --------------------------------------------------------------------------- #
# resolution helpers
# --------------------------------------------------------------------------- #

_POLICY_BY_TRANSFORM = {
    "level": "direct",
    "change": "direct_average",
    "growth": "direct_average",
    "log_growth": "direct_average",
}


def _tcode_for(data: Any, name: str) -> int | None:
    """Read a column's t-code from bundle/spec/frame metadata, if present."""
    try:
        from macroforecast.data import metadata as _metadata

        codes = _metadata(data).get("transform_codes", {}) or {}
    except Exception:
        codes = {}
    value = codes.get(name)
    return int(value) if value is not None else None


def resolve_target(
    target: str | TargetSpec,
    *,
    data: Any = None,
    tcode: int | None = None,
    tcode_map: Mapping[int, tuple[str, str]] | None = None,
    reduce_i2: bool = True,
) -> ResolvedTarget:
    """Resolve a target to its (forecast_policy, target_transform).

    Explicit ``TargetSpec.transform``/``policy`` win; otherwise the t-code (passed
    or read from ``data`` metadata) is mapped through ``tcode_map`` (defaults to
    :data:`TCODE_TARGET_MAP`). Raises if neither an explicit transform nor a
    t-code is available.
    """
    spec = target if isinstance(target, TargetSpec) else TargetSpec(name=str(target))
    code = tcode if tcode is not None else _tcode_for(data, spec.name)
    mapping = dict(TCODE_TARGET_MAP)
    if tcode_map is not None:
        mapping.update({int(k): cast("tuple[str, str]", tuple(v)) for k, v in tcode_map.items()})

    if spec.transform is not None:
        transform = str(spec.transform)
        policy = spec.policy or _POLICY_BY_TRANSFORM.get(transform, "direct_average")
    elif code is not None and int(code) in mapping:
        policy, transform = mapping[int(code)]
        if spec.policy is not None:
            policy = spec.policy
    else:
        raise ValueError(
            f"target {spec.name!r}: no explicit transform and no usable t-code "
            "(provide TargetSpec(transform=...) or transform_codes metadata)"
        )
    return ResolvedTarget(
        name=spec.name, policy=policy, transform=transform,
        tcode=(int(code) if code is not None else None), annualize=spec.annualize,
    )


def _is_multi_model(model: Any) -> bool:
    """True if ``model`` is a multi-model request (sequence or mapping).

    An ``Arm`` is exactly ONE model. A string is a single model name and a
    ``ModelSpec``/callable is a single model; a mapping or a non-string sequence
    is a multi-model request and is rejected by ``pipeline_spec``.
    """
    from macroforecast.models import ModelSpec

    if isinstance(model, (str, bytes, ModelSpec)):
        return False
    return isinstance(model, Mapping) or (
        isinstance(model, Sequence) and not isinstance(model, (str, bytes))
    )


def _arm_model_name(arm: Arm) -> str:
    """The single model name an arm contributes (for the contender key)."""
    model = arm.model
    if isinstance(model, str):
        return model
    name = getattr(model, "name", None) or getattr(model, "__name__", None)
    return str(name) if name else "model"


def _arm_model_names(arm: Arm) -> list[str]:
    """An arm is one model; this returns the single name as a one-element list.

    Retained for call sites that expect a list of contributed model names.
    """
    return [_arm_model_name(arm)]


def contender_names(arm: Arm) -> list[str]:
    """Display contender labels for an arm. A contender IS exactly an arm."""
    return [arm.name]


def is_vintage_aware(spec: PipelineSpec) -> bool:
    """Return whether a pipeline spec runs against per-origin vintage data."""

    from macroforecast.data import VintagePanelSpec

    return isinstance(spec.data, VintagePanelSpec)


# --------------------------------------------------------------------------- #
# direct-policy guard for iterated/state-space models
# --------------------------------------------------------------------------- #
# These models forecast a horizon h by ITERATING their own one-step dynamics
# (an AR/state-space recursion, a panel VAR/DFM update, or FAVAR's internal
# factor-VAR), not by fitting a genuine h-step-ahead projection. Under
# ``forecast_policy in {"direct", "direct_average"}`` the runner nonetheless
# constructs an h-step-ahead TARGET and asks the model to fit it directly; for
# these models that silently degrades toward a stale/persistence-like forecast
# at longer horizons rather than raising (the CHANGELOG [Unreleased] "stale
# persistence" defect originally found in ar/far, which now have a true
# direct-projection mode and are DELIBERATELY excluded here).
#
# The set below is every model whose ``ModelSpec.input_kind`` is "target" or
# "panel" and lacks a validated direct point-projection mode, PLUS "favar" (its
# input_kind is "supervised" -- the same bucket as ar/far -- because FAVAR's own
# factor-VAR update is still an iterated dynamics model; ar/far/var are excluded
# from this set because they have validated direct point-projection modes). This
# set is cross-checked against
# ``macroforecast.list_model_specs()`` by
# ``tests/pipeline/test_direct_policy_guard.py`` so it cannot silently rot as
# the models lane adds or removes models -- update it there, not just here, if
# that test starts failing.
DIRECT_POLICY_GUARD_MODELS: frozenset[str] = frozenset({
    # input_kind == "target": iterate their own one-step dynamics.
    "arima", "auto_arima", "ets", "holt_winters", "naive", "random_walk_drift",
    "seasonal_naive", "stlf", "theta_method",
    # input_kind == "panel": iterate their own dynamics at the panel level.
    "bvar_minnesota", "bvar_normal_inverse_wishart",
    "dfm_mixed_mariano_murasawa", "dfm_unrestricted_midas",
    # input_kind == "supervised" but genuinely iterated internally (unlike
    # ar/far, deliberately excluded -- see module docstring above).
    "favar",
})

# Models that support ``forecast_policy="direct"`` as an h-step point projection
# but not ``forecast_policy="direct_average"`` as a horizon-average target. Keep
# this policy-specific extension separate from ``DIRECT_POLICY_GUARD_MODELS`` so
# ``var`` remains valid under plain direct point forecasts.
DIRECT_AVERAGE_GUARD_MODELS: frozenset[str] = frozenset({"var"})

_DIRECT_LIKE_POLICIES = frozenset({"direct", "direct_average"})


def _direct_policy_guard_message(arm: Arm, model_name: str, policies: "Sequence[str]") -> str:
    """The guard text for one guarded arm under a direct-like policy."""
    policies_txt = ", ".join(sorted(set(policies)))
    if (
        model_name in DIRECT_AVERAGE_GUARD_MODELS
        and set(policies) == {"direct_average"}
    ):
        return (
            f"arm {arm.name!r} (model {model_name!r}) is run under forecast "
            f"policy/policies {{{policies_txt}}}: {model_name!r} supports "
            "forecast_policy='direct' as an h-step POINT projection, but it does "
            "not fit the horizon-average target required by "
            "forecast_policy='direct_average'. Labeling that point projection as "
            "direct_average would misstate the forecast object (see CHANGELOG "
            "[Unreleased] and docs/guide/model_policy_matrix.md). Prefer "
            "forecast_policy='recursive' or 'path_average' for "
            f"{model_name!r}, or give this target an explicit TargetSpec(policy=...) "
            "override if only some targets should differ. To opt out deliberately, "
            "set on_unsupported_direct='warn'; to reroute affected cells to "
            "recursive, set on_unsupported_direct='reroute'."
        )
    return (
        f"arm {arm.name!r} (model {model_name!r}) is run under forecast "
        f"policy/policies {{{policies_txt}}}: {model_name!r} forecasts by "
        "ITERATING its own dynamics rather than fitting a genuine h-step-ahead "
        "projection, so direct-policy semantics do not apply to it -- long-horizon "
        "forecasts may silently degrade toward a stale/persistence-like forecast "
        "(see CHANGELOG [Unreleased], GCLS replication Bug 3, and "
        "docs/guide/model_policy_matrix.md). Prefer forecast_policy='recursive' "
        "or 'path_average' for "
        f"{model_name!r}, or give this target an explicit "
        "TargetSpec(policy=...) override if only some targets should differ. "
        "To opt out deliberately, set on_unsupported_direct='warn'; to reroute "
        "affected cells to recursive, set on_unsupported_direct='reroute'."
    )


def _is_unsupported_direct_cell(model_name: str, policy: str) -> bool:
    if policy == "direct_average" and model_name in DIRECT_AVERAGE_GUARD_MODELS:
        return True
    return policy in _DIRECT_LIKE_POLICIES and model_name in DIRECT_POLICY_GUARD_MODELS


def _unsupported_direct_cells(
    arms: "Sequence[Arm]",
    targets: "Sequence[ResolvedTarget]",
) -> list[tuple[Arm, str, list[ResolvedTarget]]]:
    cells: list[tuple[Arm, str, list[ResolvedTarget]]] = []
    for arm in arms:
        model_name = _arm_model_name(arm)
        affected = [
            target
            for target in targets
            if _is_unsupported_direct_cell(model_name, target.policy)
        ]
        if affected:
            cells.append((arm, model_name, affected))
    return cells


def _resolve_unsupported_direct_policy(
    arms: "Sequence[Arm]",
    targets: "Sequence[ResolvedTarget]",
    *,
    mode: Literal["error", "warn", "reroute"],
) -> dict[tuple[str, str], str]:
    """Validate/warn/reroute guarded models under direct/direct_average.

    Grouped per-arm (not per (arm, target)) so a pipeline with many targets that
    all resolve to the same policy produces ONE informative diagnostic per
    affected arm rather than one per target.
    """
    affected_cells = _unsupported_direct_cells(arms, targets)
    if not affected_cells:
        return {}
    if mode == "error":
        details = "\n".join(
            "- " + _direct_policy_guard_message(
                arm,
                model_name,
                [target.policy for target in affected],
            )
            for arm, model_name, affected in affected_cells
        )
        raise ValueError(
            "unsupported direct-like forecast policy for iterated/state-space "
            f"model(s):\n{details}"
        )
    overrides: dict[tuple[str, str], str] = {}
    for arm, model_name, affected in affected_cells:
        policies = [target.policy for target in affected]
        if mode == "warn":
            warnings.warn(
                _direct_policy_guard_message(arm, model_name, policies),
                UserWarning,
                stacklevel=2,
            )
            continue
        for target in affected:
            overrides[(arm.name, target.name)] = "recursive"
        warnings.warn(
            _direct_policy_guard_message(arm, model_name, policies)
            + " Rerouting affected arm-target cell(s) to forecast_policy='recursive'; "
            "emitted rows will be labeled recursive.",
            UserWarning,
            stacklevel=2,
        )
    return overrides



# --------------------------------------------------------------------------- #
# multi-model convenience
# --------------------------------------------------------------------------- #

def _model_default_name(model: Any) -> str:
    """The default arm name for a single model (str / ModelSpec.name / __name__)."""
    from macroforecast.models import ModelSpec

    if isinstance(model, str):
        return model
    if isinstance(model, ModelSpec):
        return model.name
    name = getattr(model, "name", None) or getattr(model, "__name__", None)
    return str(name) if name else "model"


def _per_arm_or_shared(
    value: Any, arm_names: Sequence[str], *, label: str
) -> dict[str, Any]:
    """Resolve a config that is either shared by all arms or varied per arm.

    The disambiguation heuristic: if ``value`` is a Mapping whose key set is
    exactly the set of arm names, it is treated as a per-arm mapping (one entry
    per arm). Otherwise the whole ``value`` is treated as the shared value given
    to every arm. This means a single shared ``params``/``model_selection`` dict
    (whose keys are hyperparameter names, not arm names) is correctly shared, and
    per-arm variation requires keys that are *exactly* the arm names.
    """
    if isinstance(value, Mapping) and set(value.keys()) == set(arm_names):
        return {name: value[name] for name in arm_names}
    return {name: value for name in arm_names}


def model_arms(
    models: Sequence[Any] | Mapping[str, Any],
    *,
    names: Sequence[str] | None = None,
    preprocessing: Any | None = None,
    preprocessing_policy: Any | None = None,
    features: Any | None = None,
    feature_policy: Any | None = None,
    params: Mapping[str, Any] | None = None,
    model_selection: Any | None = None,
    model_selection_metric: str = "mse",
    interpret: InterpretSpec | tuple[str, ...] | None = None,
    nested_in_benchmark: bool | set[str] | Sequence[str] = False,
    metadata: Mapping[str, Any] | None = None,
) -> list[Arm]:
    """Build one :class:`Arm` per model for a pure model comparison.

    A model comparison is "several Arms identical except ``model``". This helper
    is the pipeline idiom for that: it returns a ``list[Arm]`` -- one arm (one
    contender; each (target, horizon) of it is one cell) per model -- all
    sharing the given preprocessing,
    features, and evaluation config, differing only in ``model`` (and in the
    per-model ``params``/``model_selection``/nesting when those are given as
    mappings/sets).

    ``models`` is a sequence of single models (``str`` | ``Callable`` |
    ``ModelSpec``) or a ``Mapping[name -> model]``. Arm names default to the
    model name (``str(model)`` / ``ModelSpec.name`` / ``callable.__name__``), or
    the mapping keys, or the explicit ``names``. Names must be unique.

    ``params`` and ``model_selection`` are shared by every arm unless they are a
    Mapping whose key set is exactly the arm names, in which case each entry is
    applied to its arm (see :func:`_per_arm_or_shared`). A single shared dict of
    hyperparameters is therefore unambiguous from a per-arm mapping.

    ``nested_in_benchmark`` is either a bool shared by all arms, or a set/sequence
    of arm names that nest the benchmark.

    The returned list is ready to pass to ``pipeline_spec(arms=...)``. Pure model
    comparison shares all config; comparing feature cases still needs explicit
    Arms (build them by hand, varying ``features``).
    """
    # Normalise the models into parallel (name, model) lists.
    if isinstance(models, Mapping):
        if names is not None:
            raise ValueError(
                "model_arms: pass either a Mapping of {name: model} or a sequence "
                "of models with names=, not a Mapping together with names="
            )
        arm_names = [str(k) for k in models.keys()]
        model_list = list(models.values())
    else:
        model_list = list(models)
        if not model_list:
            raise ValueError("model_arms requires at least one model")
        if names is not None:
            names = list(names)
            if len(names) != len(model_list):
                raise ValueError(
                    f"model_arms: names has length {len(names)} but there are "
                    f"{len(model_list)} models"
                )
            arm_names = [str(n) for n in names]
        else:
            arm_names = [_model_default_name(m) for m in model_list]

    if not model_list:
        raise ValueError("model_arms requires at least one model")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError(f"model_arms: arm names must be unique, got {arm_names}")

    # Each entry must be a SINGLE model; reject a nested sequence/mapping early
    # with a clear message (pipeline_spec would also catch it later).
    for name, model in zip(arm_names, model_list):
        if _is_multi_model(model):
            raise ValueError(
                f"model_arms: arm {name!r} was given a {type(model).__name__} of "
                "models, but each model must be a SINGLE model. Pass one model per "
                "arm; to vary features build Arms by hand."
            )

    params_by_arm = _per_arm_or_shared(params, arm_names, label="params")
    selection_by_arm = _per_arm_or_shared(
        model_selection, arm_names, label="model_selection"
    )

    # nested_in_benchmark: bool shared by all, or a collection of arm names.
    if isinstance(nested_in_benchmark, bool):
        nested_names: set[str] = set(arm_names) if nested_in_benchmark else set()
    else:
        nested_names = {str(n) for n in nested_in_benchmark}
        unknown = nested_names - set(arm_names)
        if unknown:
            raise ValueError(
                f"model_arms: nested_in_benchmark names {sorted(unknown)} are not "
                f"among the arm names {arm_names}"
            )

    shared_metadata = dict(metadata) if metadata is not None else {}

    return [
        Arm(
            name=name,
            model=model,
            preprocessing=preprocessing,
            preprocessing_policy=preprocessing_policy,
            features=features,
            feature_policy=feature_policy,
            params=params_by_arm[name],
            model_selection=selection_by_arm[name],
            model_selection_metric=model_selection_metric,
            interpret=interpret,
            nested_in_benchmark=(name in nested_names),
            metadata=dict(shared_metadata),
        )
        for name, model in zip(arm_names, model_list)
    ]


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #

def _resolve_preprocessing_cache_dir(value: str | bool | None) -> str | Literal[False] | None:
    """Normalize the raw ``preprocessing_cache_dir`` argument to its stored form.

    ``None`` -> ``None`` ("not configured"; run-time defaulting decides). ``False``
    -> ``False`` (explicit opt-out, stored as-is). Any other value is coerced to
    ``str`` (an explicit directory path). ``True`` is rejected -- it is not a valid
    directory and it is not the opt-out sentinel, so silently accepting it would
    invite exactly the "did the user mean an explicit path or the default" ambiguity
    this three-state contract exists to avoid. The stored form is therefore
    ``str | Literal[False] | None`` (the field annotation): ``True`` never
    survives validation, so truthiness-narrowing the FIELD yields ``str``.
    """
    if value is None or value is False:
        return value
    if value is True:
        raise ValueError(
            "preprocessing_cache_dir=True is not valid; pass a directory path "
            "(str), None (default: auto-managed temp dir when n_jobs>1), or "
            "False (explicit opt-out of any preprocessing cache)."
        )
    return str(value)


def pipeline_spec(
    *,
    data: Any,
    targets: Sequence[str | TargetSpec],
    horizons: Sequence[int] | int,
    window: Any,
    arms: Sequence[Arm],
    evaluation: EvalSpec,
    combinations: Sequence[CombinationContender] = (),
    preprocessing: Any | None = None,
    preprocessing_policy: Any | None = None,
    tcode_target_map: Mapping[int, tuple[str, str]] | None = None,
    save_models: bool = False,
    model_store: str = "trained_model",
    checkpoint_dir: str | None = None,
    result_store: str | Path | None = None,
    n_jobs: int | str = 1,
    preprocessing_cache_dir: str | bool | None = None,
    seed: int | None = 42,
    provenance: Mapping[str, Any] | None = None,
    provenance_level: Literal["full", "basic"] = "full",
    on_unsupported_direct: Literal["error", "warn", "reroute"] = "error",
) -> PipelineSpec:
    """Validate and build a :class:`PipelineSpec`.

    ``n_jobs`` is a positive int (explicit cell-worker count) or the literal
    ``"auto"``. ``"auto"`` inspects the core budget and the work structure
    (``len(targets) * len(arms) * len(horizons)`` cells) via
    :func:`auto_parallelism` and splits the cores between cell workers
    (stored as the resolved ``PipelineSpec.n_jobs``) and per-cell model-internal
    threads (stored as ``PipelineSpec.model_threads``).

    ``preprocessing_cache_dir`` is a path, ``None``, or ``False`` -- see the field
    docstring on :class:`PipelineSpec` for the full three-state contract. In short:
    a path pins a persistent shared on-disk preprocessing-fit cache; ``None``
    (default) auto-manages a temporary one for the duration of the run when
    ``n_jobs>1`` (a no-op when ``n_jobs==1``); ``False`` explicitly opts out of any
    disk-backed cache even when parallel. ``True`` is invalid and raises.

    ``provenance_level`` ("full" default, or "basic") controls how much
    ``PipelineReport.provenance`` a live ``run_pipeline``/``rescore`` call
    attaches -- see the field docstring on :class:`PipelineSpec`. Note this is
    independent of ``provenance=`` above (caller-supplied notes merged into
    whichever shape results); "basic" does not drop caller-supplied notes, it
    only omits the "environment"/"data"/"spec_echo" blocks.

    ``result_store`` is an optional directory for cross-run reuse of completed
    forecast cells. When left at ``None`` (the default), the runner follows the
    original execution path exactly.

    ``seed`` is the pipeline run seed. During :func:`run_pipeline`, it temporarily
    becomes the package meta ``random_seed`` so model selection and parallel
    workers see the same effective seed. For model specs with a ``random_state``
    default, pipeline fits derive a stable per-arm value from ``(seed, arm)`` when
    the caller did not explicitly supply ``random_state``; explicit model params
    always take precedence. Low-level :func:`macroforecast.forecasting.run` keeps
    the model registry defaults unless it is called from this pipeline context.

    ``on_unsupported_direct`` controls what happens when a model that only
    iterates its own dynamics is combined with ``direct`` or ``direct_average``:
    ``"error"`` (default) rejects the spec, ``"warn"`` preserves the old weak
    benchmark behavior, and ``"reroute"`` runs only the affected arm-target
    cells as ``recursive``.
    """
    arms = tuple(arms)
    if not arms:
        raise ValueError("pipeline requires at least one arm")
    if on_unsupported_direct not in ("error", "warn", "reroute"):
        raise ValueError(
            "on_unsupported_direct must be one of 'error', 'warn', or 'reroute'"
        )
    if provenance_level not in ("full", "basic"):
        raise ValueError(
            f"provenance_level must be 'full' or 'basic', got {provenance_level!r}"
        )
    # n_jobs is a positive int OR the literal "auto"; the auto split is computed
    # below once the cell count (targets x arms x horizons) is known.
    auto_jobs = n_jobs == "auto"
    if not auto_jobs:
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be a positive integer (>= 1) or 'auto'")
        n_jobs = int(n_jobs)
        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer (>= 1) or 'auto'")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")

    # An Arm is exactly ONE model. Comparing models means multiple Arms that are
    # identical except for ``model``; comparing feature cases means Arms that
    # differ in features (and model). A sequence/mapping of models in one arm is
    # the old multi-model arm and is no longer supported.
    for arm in arms:
        if _is_multi_model(arm.model):
            raise ValueError(
                f"arm {arm.name!r} was given a {type(arm.model).__name__} of "
                "models, but an Arm is exactly ONE model. Use one Arm per model "
                "(identical arms differing only in 'model') to compare models."
            )

    horizon_tuple = (
        (int(horizons),) if isinstance(horizons, int) else tuple(int(h) for h in horizons)
    )
    if not horizon_tuple or any(h < 1 for h in horizon_tuple):
        raise ValueError("horizons must be a non-empty set of integers >= 1")

    resolved = tuple(
        resolve_target(t, data=data, tcode_map=tcode_target_map) for t in targets
    )
    if not resolved:
        raise ValueError("at least one target is required")

    # AUTO allocator: split the core budget between cell workers and per-cell
    # model-internal threads from the (target x arm x horizon) cell count. The
    # resolved cell-worker count becomes n_jobs; model_threads is the per-worker
    # tree-ensemble thread budget (used only when n_jobs>1, see _parallel_cell_worker).
    model_threads = 1
    if auto_jobs:
        from macroforecast.pipeline.parallelism import auto_parallelism

        n_cells = len(resolved) * len(arms) * len(horizon_tuple)
        n_jobs, model_threads = auto_parallelism(n_cells)

    # By here n_jobs is a resolved int: either int(n_jobs) above or the int
    # returned by auto_parallelism when auto_jobs was set.
    assert isinstance(n_jobs, int)
    # benchmark must resolve to an existing contender (an arm name, since a
    # contender IS an arm, or a combination contender name)
    all_contenders = {c for a in arms for c in contender_names(a)}
    all_contenders |= {c.name for c in combinations}
    if evaluation.benchmark not in all_contenders:
        raise ValueError(
            f"evaluation.benchmark {evaluation.benchmark!r} is not among the "
            f"contenders {sorted(all_contenders)}"
        )
    evaluation = _normalize_eval_spec(evaluation)

    # evaluation.tests must be names the evaluator actually wires; a typo or an
    # aspirational test name raises here rather than being silently ignored.
    unknown_tests = set(evaluation.tests) - SUPPORTED_EVAL_TESTS
    if unknown_tests:
        raise ValueError(
            f"evaluation.tests contains unsupported name(s) {sorted(unknown_tests)}; "
            f"supported tests are {sorted(SUPPORTED_EVAL_TESTS)}."
        )
    requested_arch_tests = _ARCH_BACKED_EVAL_TESTS & set(evaluation.tests)
    if requested_arch_tests and not _arch_available():
        raise ImportError(
            f"evaluation.tests contains arch-backed test(s) "
            f"{sorted(requested_arch_tests)}, but the optional arch backend is "
            'not installed; install it with pip install "macroforecast[arch]".'
        )
    if {"uspa", "aspa"} & set(evaluation.tests) and len(horizon_tuple) < 2:
        raise ValueError(
            "evaluation.tests contains multi-horizon test(s) 'uspa'/'aspa', "
            "but the spec has only one horizon; request at least two horizons "
            "or remove the joint multi-horizon test."
        )
    _validate_eval_test_options(evaluation)

    policy_overrides = _resolve_unsupported_direct_policy(
        arms,
        resolved,
        mode=on_unsupported_direct,
    )

    notes = dict(provenance or {})
    if not save_models and any(a.interpret for a in arms):
        notes.setdefault(
            "warnings", []
        )
        notes["warnings"] = [*notes.get("warnings", []),
                             "save_models=False but an arm requests interpretation; "
                             "interpretation will re-fit models."]

    return PipelineSpec(
        data=data, targets=resolved, horizons=horizon_tuple, window=window,
        arms=arms, evaluation=evaluation, combinations=tuple(combinations),
        preprocessing=preprocessing, preprocessing_policy=preprocessing_policy,
        save_models=bool(save_models), model_store=str(model_store),
        checkpoint_dir=(str(checkpoint_dir) if checkpoint_dir is not None else None),
        result_store=(str(result_store) if result_store is not None else None),
        n_jobs=n_jobs,
        model_threads=int(model_threads),
        preprocessing_cache_dir=_resolve_preprocessing_cache_dir(preprocessing_cache_dir),
        seed=seed,
        provenance=notes,
        provenance_level=provenance_level,
        policy_overrides=policy_overrides,
    )
