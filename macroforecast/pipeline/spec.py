"""Pipeline specification: schema, t-code -> target resolution, and validation.

Stage 0 of the comprehensive POOS pipeline. Defines the declarative configuration
objects (frozen dataclasses) and the validating generator ``pipeline_spec``. The
execution (``run_pipeline``), interpretation, and reporting layers build on these.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


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
    """A forecast target and how its forecast object is defined."""

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


@dataclass(frozen=True)
class EvalSpec:
    """Automatic evaluation and significance-testing configuration."""

    benchmark: str
    metrics: tuple[str, ...] = ("rmse", "relative_mse", "r2_oos")
    tests: tuple[str, ...] = ("dm", "cw", "mcs")
    by: tuple[str, ...] = ("target", "horizon")
    primary_axis: str = "contender"
    cw_for_nested: bool = True
    mcs_alpha: float = 0.10
    mcs_method: str = "iterative"
    multiple_testing: str | None = None
    subsamples: Mapping[str, tuple[Any, Any]] = field(default_factory=dict)
    dm_kwargs: Mapping[str, Any] = field(default_factory=dict)


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
    save_models: bool = True
    model_store: str = "trained_model"
    # When set, each (target, arm, horizon) forecast cell persists its lean
    # forecast records incrementally under
    # ``<checkpoint_dir>/<target>__<arm>/h<h>/`` so a crashed multi-hour POOS run
    # resumes without recomputing finished origins. None (default) disables
    # checkpointing and is byte-for-byte the prior behavior.
    checkpoint_dir: str | None = None
    # Native fan-out: when >1, the (arm x target x horizon) cells run across a
    # process pool. Default 1 keeps the sequential, cross-horizon EM-sharing path
    # byte-for-byte unchanged. The parallel path is deterministic (every cell uses
    # ``seed``) and produces forecasts numerically identical to ``n_jobs=1``; it
    # trades the shared per-origin preprocessing cache (each worker recomputes its
    # own EM) for wall-clock parallelism. Memory scales with ``n_jobs`` because
    # every worker holds the data panel.
    n_jobs: int = 1
    # Model-internal thread budget per cell worker, set by the AUTO allocator.
    # In parallel mode (n_jobs>1) each worker pins its tree-ensemble (RF/GBM/XGB/
    # LGBM) internal n_jobs to this value so cell_workers * model_threads <= cores
    # and the CPU is saturated without oversubscription. Default 1 (serial mode and
    # any explicit-int n_jobs leave each worker single-threaded internally, the
    # prior behavior). Only changes thread COUNT, never the numerical result.
    model_threads: int = 1
    seed: int | None = 42
    provenance: Mapping[str, Any] = field(default_factory=dict)



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
        mapping.update({int(k): tuple(v) for k, v in tcode_map.items()})

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
    save_models: bool = True,
    model_store: str = "trained_model",
    checkpoint_dir: str | None = None,
    n_jobs: int | str = 1,
    seed: int | None = 42,
    provenance: Mapping[str, Any] | None = None,
) -> PipelineSpec:
    """Validate and build a :class:`PipelineSpec`.

    ``n_jobs`` is a positive int (explicit cell-worker count) or the literal
    ``"auto"``. ``"auto"`` inspects the core budget and the work structure
    (``len(targets) * len(arms) * len(horizons)`` cells) via
    :func:`auto_parallelism` and splits the cores between cell workers
    (stored as the resolved ``PipelineSpec.n_jobs``) and per-cell model-internal
    threads (stored as ``PipelineSpec.model_threads``).
    """
    arms = tuple(arms)
    if not arms:
        raise ValueError("pipeline requires at least one arm")
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

    # benchmark must resolve to an existing contender (an arm name, since a
    # contender IS an arm, or a combination contender name)
    all_contenders = {c for a in arms for c in contender_names(a)}
    all_contenders |= {c.name for c in combinations}
    if evaluation.benchmark not in all_contenders:
        raise ValueError(
            f"evaluation.benchmark {evaluation.benchmark!r} is not among the "
            f"contenders {sorted(all_contenders)}"
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
        n_jobs=n_jobs,
        model_threads=int(model_threads),
        seed=seed,
        provenance=notes,
    )
