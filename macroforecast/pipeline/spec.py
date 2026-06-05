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
    """One comparison unit: a full (preprocessing, features, model) configuration."""

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


def _arm_model_names(arm: Arm) -> list[str]:
    """Enumerate the model names an arm contributes (for contender keys)."""
    model = arm.model
    if isinstance(model, str):
        return [model]
    if isinstance(model, Mapping):
        return [str(k) for k in model]
    if isinstance(model, Sequence) and not isinstance(model, (str, bytes)):
        names: list[str] = []
        for item in model:
            names.extend(_arm_model_names(Arm(name=arm.name, model=item)))
        return names
    name = getattr(model, "name", None) or getattr(model, "__name__", None)
    return [str(name) if name else "model"]


def contender_names(arm: Arm) -> list[str]:
    """Display contender labels for an arm: ``arm`` (single model) or ``arm:model``."""
    models = _arm_model_names(arm)
    if len(models) == 1:
        return [arm.name]
    return [f"{arm.name}:{m}" for m in models]


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
    seed: int | None = 42,
    provenance: Mapping[str, Any] | None = None,
) -> PipelineSpec:
    """Validate and build a :class:`PipelineSpec`."""
    arms = tuple(arms)
    if not arms:
        raise ValueError("pipeline requires at least one arm")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")

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

    # benchmark must resolve to an existing contender (arm name or arm:model)
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
        save_models=bool(save_models), model_store=str(model_store), seed=seed,
        provenance=notes,
    )
