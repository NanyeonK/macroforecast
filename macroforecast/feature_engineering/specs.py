from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, attach_metadata, panel_info, validate_panel
from macroforecast.feature_engineering.shared import (
    _deterministic_pca,
    TargetMode,
    TargetTransform,
    _apply_sparse_pca_chen_rohe,
    _apply_varimax_rotation,
    _component_records,
    _coerce_input,
    _effective_pls_components,
    _fit_sparse_factor_var1,
    _fit_sparse_pca_chen_rohe,
    _fit_varimax_rotation,
    _group_component_prefix,
    _maf_component_prefix,
    _metadata_frame,
    _normalize_column_groups,
    _normalize_feature_method,
    _normalize_lags,
    _normalize_maf_lags,
    _normalize_min_train_size,
    _normalize_positive_ints,
    _normalize_scale_method,
    _normalize_target_mode,
    _normalize_target_transform,
    _reject_extra_params,
    _records_for_columns,
    _resolve_columns,
    _resolve_group_components,
    _resolve_horizons,
    _resolve_predictors,
    _resolve_targets,
    _scale_parameters,
    _target_metadata_frame,
    _target_transform_to_path_transform,
)
from macroforecast.feature_engineering.feature_selection import (
    normalize_feature_selection_method,
    select_features,
)
from macroforecast.feature_engineering.screening import (
    fit_predictor_screen,
)
from macroforecast.feature_engineering.targets import direct_target, path_targets
from macroforecast.feature_engineering.transforms import (
    fourier_features,
    interaction_features,
    lag,
    moving_average_ladder,
    polynomial_features,
    rolling_mean,
    season_dummy,
    seasonal_lag,
    time_features,
    transform_features,
)
from macroforecast.feature_engineering.types import FeatureInput, FeatureSet


_FEATURE_SPEC_METHODS = frozenset(
    {
        "lag",
        "rolling_mean",
        "moving_average_ladder",
        "marx",
        "transform",
        "seasonal_lag",
        "season_dummy",
        "fourier",
        "polynomial",
        "interaction",
        "time",
        "scale",
        "pca",
        "sparse_pca_chen_rohe",
        "varimax",
        "group_pca",
        "maf",
        "hamilton_filter",
        "random_projection",
        "nystroem",
        "partial_least_squares",
        "sliced_inverse_regression",
        "predictor_screen",
        "variance_selection",
        "correlation_selection",
        "lasso_selection",
        "lasso_path_selection",
        "rfe_selection",
        "boruta_selection",
        "stability_selection",
        "genetic_selection",
        "custom",
    }
)

@dataclass(frozen=True)
class FeatureSpec:
    """Reusable feature-building callable for forecasting runners."""

    target: str | None = None
    targets: tuple[str, ...] = ()
    horizon: int | None = None
    horizons: tuple[int, ...] = ()
    predictors: Literal["all"] | tuple[str, ...] | None = None
    lags: tuple[int, ...] = (0, 1)
    target_lags: tuple[int, ...] = ()
    rolling_windows: tuple[int, ...] = ()
    rolling_min_periods: int | None = None
    add_time: bool = False
    time_trend: bool = True
    time_month: bool = False
    time_quarter: bool = False
    time_year: bool = False
    pca_components: int | None = None
    pca_columns: tuple[str, ...] | None = None
    pca_scale: bool = True
    pca_prefix: str = "pc"
    feature_steps: tuple[dict[str, Any], ...] = ()
    include_original: bool = False
    target_transform: TargetTransform = "level"
    target_mode: TargetMode = "direct"
    drop_missing: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, data: FeatureInput, *, metadata: Mapping[str, Any] | None = None) -> FittedFeatureBuilder:
        """Fit feature-building state on a training panel."""

        base = _coerce_input(data, metadata=metadata)
        panel = base.panel
        validate_panel(panel)
        target_values = _resolve_targets(
            panel,
            base=base,
            target=self.target,
            targets=self.targets or None,
        )
        horizon_values = _resolve_horizons(
            base=base,
            horizon=self.horizon,
            horizons=self.horizons or None,
        )
        predictor_values = _resolve_predictors(
            panel,
            base=base,
            predictors=self.predictors,
            targets=target_values,
        ) if not (self.predictors == () and self.target_lags) else ()
        step_states = _fit_feature_step_pipeline(
            panel,
            predictors=predictor_values,
            targets=target_values,
            steps=self.feature_steps,
            target_frame=_target_frame_for_feature_steps(
                panel,
                targets=target_values,
                horizons=horizon_values,
                target_mode=self.target_mode,
                target_transform=self.target_transform,
            )
            if self.feature_steps
            else None,
        )
        pca_state = (
            None
            if self.feature_steps
            else _fit_pca_state(
                panel,
                predictors=predictor_values,
                columns=self.pca_columns,
                n_components=self.pca_components,
                scale=self.pca_scale,
                prefix=self.pca_prefix,
            )
        )
        return FittedFeatureBuilder(
            spec=self,
            fit_panel=panel.copy(),
            fit_metadata=dict(base.metadata),
            targets=target_values,
            horizons=horizon_values,
            predictors=predictor_values,
            pca_state=pca_state,
            step_states=step_states,
        )

    def fit_transform(self, data: FeatureInput, *, metadata: Mapping[str, Any] | None = None) -> FeatureSet:
        """Fit on ``data`` and return a feature set for the same panel."""

        return self.fit(data, metadata=metadata).transform(data, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready feature choices."""

        return {
            "target": self.target,
            "targets": list(self.targets),
            "horizon": self.horizon,
            "horizons": list(self.horizons),
            "predictors": self.predictors if self.predictors == "all" else list(self.predictors or ()),
            "lags": [] if self.feature_steps else list(self.lags),
            "target_lags": list(self.target_lags),
            "rolling_windows": list(self.rolling_windows),
            "rolling_min_periods": self.rolling_min_periods,
            "add_time": self.add_time,
            "time": {
                "trend": self.time_trend,
                "month": self.time_month,
                "quarter": self.time_quarter,
                "year": self.time_year,
            },
            "pca_components": None if self.feature_steps else self.pca_components,
            "pca_columns": [] if self.feature_steps else list(self.pca_columns or ()),
            "pca_scale": None if self.feature_steps else self.pca_scale,
            "pca_prefix": None if self.feature_steps else self.pca_prefix,
            "feature_steps": [_json_ready_step(step) for step in self.feature_steps],
            "include_original": self.include_original,
            "target_transform": self.target_transform,
            "target_mode": self.target_mode,
            "drop_missing": self.drop_missing,
            "metadata": dict(self.metadata),
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return compact metadata for runners."""

        return {"feature_engineering": self.to_dict()}


@dataclass(frozen=True)
class FittedFeatureBuilder:
    """Feature builder fitted on a training window."""

    spec: FeatureSpec
    fit_panel: pd.DataFrame
    fit_metadata: dict[str, Any]
    targets: tuple[str, ...]
    horizons: tuple[int, ...]
    predictors: tuple[str, ...]
    pca_state: _PCAState | None = None
    step_states: tuple[_FittedFeatureStep, ...] = ()

    def transform(
        self,
        data: FeatureInput,
        *,
        metadata: Mapping[str, Any] | None = None,
        index: Iterable[Any] | pd.Index | None = None,
    ) -> FeatureSet:
        """Create model-ready ``X`` and ``y`` from a panel."""

        base = _coerce_input(data, metadata=metadata)
        panel = base.panel
        validate_panel(panel)
        X, feature_metadata = _build_predictors(panel, self)
        y = _build_targets(panel, self)
        target_metadata = y.attrs.get("macroforecast_target_metadata", _target_metadata_frame([]))

        if index is not None:
            labels = pd.Index(index)
            X = X.reindex(labels)
            y = y.reindex(labels)

        if self.spec.drop_missing:
            aligned = pd.concat([X, y], axis=1).dropna()
            X = aligned.loc[:, X.columns]
            y = aligned.loc[:, y.columns]
        if X.empty or y.empty:
            raise ValueError("feature builder leaves an empty aligned sample")

        stage = {
            "spec": self.spec.to_dict(),
            "fit_panel": panel_info(DataBundle(self.fit_panel, self.fit_metadata)),
            "input_panel": panel_info(DataBundle(panel, base.metadata)),
            "predictors": list(self.predictors),
            "targets": list(self.targets),
            "horizons": list(self.horizons),
            "output": {
                "n_observations": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_targets": int(y.shape[1]),
            },
        }
        updated_metadata = attach_metadata(base.metadata, "feature_spec", stage)
        X = X.copy()
        y = y.copy()
        feature_metadata = feature_metadata.copy()
        target_metadata = target_metadata.copy()
        X.attrs["macroforecast_metadata"] = updated_metadata
        y.attrs["macroforecast_metadata"] = updated_metadata
        y.attrs["macroforecast_target_metadata"] = target_metadata
        feature_metadata.attrs["macroforecast_metadata"] = updated_metadata
        target_metadata.attrs["macroforecast_metadata"] = updated_metadata
        return FeatureSet(
            X=X,
            y=y,
            metadata=updated_metadata,
            feature_metadata=feature_metadata,
            target_metadata=target_metadata,
            target=self.targets[0] if len(self.targets) == 1 else None,
            targets=self.targets,
            horizons=self.horizons,
            predictors=self.predictors,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return fit metadata for forecasting records."""

        return {
            "feature_spec": self.spec.to_dict(),
            "fit_panel": panel_info(DataBundle(self.fit_panel, self.fit_metadata)),
            "targets": list(self.targets),
            "horizons": list(self.horizons),
            "predictors": list(self.predictors),
            "feature_steps": [step.to_metadata() for step in self.step_states],
            "pca": None if self.pca_state is None else self.pca_state.to_dict(),
        }


@dataclass(frozen=True)
class _PCAState:
    columns: tuple[str, ...]
    n_components: int
    scale: bool
    prefix: str
    center: pd.Series | None
    divisor: pd.Series | None
    model: Any

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        valid = source.dropna()
        columns = [f"{self.prefix}{idx}" for idx in range(1, self.n_components + 1)]
        result = pd.DataFrame(index=panel.index, columns=columns, dtype=float)
        if valid.empty:
            return result
        values = valid
        if self.scale and self.center is not None and self.divisor is not None:
            values = (values - self.center) / self.divisor
        transformed = pd.DataFrame(
            self.model.transform(values),
            index=valid.index,
            columns=columns,
        )
        result.loc[transformed.index, :] = transformed
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "n_components": self.n_components,
            "scale": self.scale,
            "prefix": self.prefix,
        }


@dataclass(frozen=True)
class _SparsePCAChenRoheState:
    columns: tuple[str, ...]
    n_components: int
    resolved_n_components: int
    zeta: float
    zeta_resolved: float
    max_iter: int
    var_innovations: bool
    prefix: str
    random_state: int | None
    min_train_size: int
    center: pd.Series
    theta: np.ndarray
    var_coef: np.ndarray | None
    n_fit_rows: int
    n_iter: int
    objective: float

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        return _apply_sparse_pca_chen_rohe(
            panel,
            columns=self.columns,
            center=self.center,
            theta=self.theta,
            prefix=self.prefix,
            var_coef=self.var_coef,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "n_components": self.n_components,
            "resolved_n_components": self.resolved_n_components,
            "zeta": self.zeta,
            "zeta_resolved": self.zeta_resolved,
            "max_iter": self.max_iter,
            "var_innovations": self.var_innovations,
            "prefix": self.prefix,
            "random_state": self.random_state,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel",
            "n_fit_rows": self.n_fit_rows,
            "n_iter": self.n_iter,
            "objective": self.objective,
        }


@dataclass(frozen=True)
class _VarimaxState:
    columns: tuple[str, ...]
    max_iter: int
    tol: float
    prefix: str
    min_train_size: int
    rotation: np.ndarray
    n_fit_rows: int
    n_iter: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        return _apply_varimax_rotation(
            panel,
            columns=self.columns,
            rotation=self.rotation,
            prefix=self.prefix,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "max_iter": self.max_iter,
            "tol": self.tol,
            "prefix": self.prefix,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel",
            "n_fit_rows": self.n_fit_rows,
            "n_iter": self.n_iter,
        }


@dataclass(frozen=True)
class _ScaleState:
    columns: tuple[str, ...]
    method: str
    center: pd.Series
    divisor: pd.Series

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        result = (source - self.center) / self.divisor
        result = result.add_suffix(f"_{self.method}")
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "method": self.method}


@dataclass(frozen=True)
class _GroupPCAState:
    groups: dict[str, tuple[str, ...]]
    states: dict[str, _PCAState]
    n_components: dict[str, int]
    scale: bool
    prefix: str | None

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        pieces = [self.states[group_name].transform(panel) for group_name in self.groups]
        result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": {group: list(columns) for group, columns in self.groups.items()},
            "n_components": dict(self.n_components),
            "scale": self.scale,
            "prefix": self.prefix,
        }


@dataclass(frozen=True)
class _MAFColumnState:
    source_column: str
    lag_values: tuple[int, ...]
    pca_state: _PCAState

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        lag_block = pd.DataFrame(
            {
                f"{self.source_column}_lag{lag_value}": panel[self.source_column].shift(lag_value)
                for lag_value in self.lag_values
            },
            index=panel.index,
        )
        lag_block.index.name = "date"
        return self.pca_state.transform(lag_block)


@dataclass(frozen=True)
class _MAFState:
    columns: tuple[str, ...]
    lag_values: tuple[int, ...]
    n_components: int
    scale: bool
    prefix: str
    column_states: tuple[_MAFColumnState, ...]

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        pieces = [state.transform(panel) for state in self.column_states]
        result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "lags": list(self.lag_values),
            "n_components": self.n_components,
            "scale": self.scale,
            "prefix": self.prefix,
        }


@dataclass(frozen=True)
class _MARXState:
    columns: tuple[str, ...]
    max_lag: int
    scale_lags: bool
    min_train_size: int | None
    center: pd.Series | None = None
    divisor: pd.Series | None = None

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        lag_values = tuple(range(1, self.max_lag + 1))
        lag_matrix = _marx_lag_matrix(panel, columns=self.columns, lag_values=lag_values)
        if self.scale_lags:
            if self.center is None or self.divisor is None:
                raise ValueError("scaled MARX state is missing fitted scale parameters")
            lag_matrix = (lag_matrix - self.center) / self.divisor
        result = _marx_from_lag_matrix(
            lag_matrix,
            columns=self.columns,
            lag_values=lag_values,
        )
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "max_lag": self.max_lag,
            "scale_lags": self.scale_lags,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel" if self.scale_lags else "deterministic",
        }


@dataclass(frozen=True)
class _HamiltonState:
    columns: tuple[str, ...]
    h: int
    p: int
    component: str
    min_train_size: int
    beta_by_column: dict[str, np.ndarray]
    fit_rows_by_column: dict[str, int]

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=panel.index)
        for column in self.columns:
            if column not in self.beta_by_column:
                raise ValueError(f"Hamilton state is missing fitted coefficients for {column!r}")
            cycle, trend = _hamilton_apply_beta(
                panel[column],
                beta=self.beta_by_column[column],
                h=self.h,
                p=self.p,
            )
            if self.component in {"cycle", "both"}:
                result[f"{column}_hamilton_cycle"] = cycle
            if self.component in {"trend", "both"}:
                result[f"{column}_hamilton_trend"] = trend
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "h": self.h,
            "p": self.p,
            "component": self.component,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel",
            "fit_rows_by_column": dict(self.fit_rows_by_column),
            "label_alignment": "components are labeled at t+h",
        }


@dataclass(frozen=True)
class _RandomProjectionState:
    columns: tuple[str, ...]
    n_components: int
    prefix: str
    random_state: int | None
    min_train_size: int
    transformer: Any
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        valid = source.dropna()
        columns = [f"{self.prefix}{idx}" for idx in range(1, self.n_components + 1)]
        result = pd.DataFrame(index=panel.index, columns=columns, dtype=float)
        if valid.empty:
            return result
        transformed = pd.DataFrame(
            self.transformer.transform(valid.to_numpy(dtype=float)),
            index=valid.index,
            columns=columns,
        )
        result.loc[transformed.index, :] = transformed
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "n_components": self.n_components,
            "prefix": self.prefix,
            "random_state": self.random_state,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _NystroemState:
    columns: tuple[str, ...]
    n_components: int
    kernel: str
    gamma: float | None
    prefix: str
    random_state: int | None
    min_train_size: int
    transformer: Any
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        valid = source.dropna()
        columns = [f"{self.prefix}{idx}" for idx in range(1, self.n_components + 1)]
        result = pd.DataFrame(index=panel.index, columns=columns, dtype=float)
        if valid.empty:
            return result
        transformed = pd.DataFrame(
            self.transformer.transform(valid.to_numpy(dtype=float)),
            index=valid.index,
            columns=columns,
        )
        result.loc[transformed.index, :] = transformed
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "n_components": self.n_components,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "prefix": self.prefix,
            "random_state": self.random_state,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _PartialLeastSquaresState:
    columns: tuple[str, ...]
    target: str
    n_components: int
    resolved_n_components: int
    prefix: str
    model: Any
    min_train_size: int
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        valid = source.dropna()
        output_columns = [f"{self.prefix}{idx}" for idx in range(1, self.n_components + 1)]
        result = pd.DataFrame(index=panel.index, columns=output_columns, dtype=float)
        if valid.empty or self.resolved_n_components == 0 or self.model is None:
            result.index.name = "date"
            return result
        scores = np.asarray(self.model.transform(valid.to_numpy(dtype=float)), dtype=float)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        resolved_columns = output_columns[: self.resolved_n_components]
        result.loc[valid.index, resolved_columns] = scores[:, : self.resolved_n_components]
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "target": self.target,
            "n_components": self.n_components,
            "resolved_n_components": self.resolved_n_components,
            "prefix": self.prefix,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel_target_aligned_rows",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _SlicedInverseRegressionState:
    columns: tuple[str, ...]
    target: str
    n_components: int
    resolved_n_components: int
    n_slices: int
    resolved_n_slices: int
    scaling_policy: str
    prefix: str
    center: pd.Series
    divisor: pd.Series
    beta: np.ndarray | None
    directions: np.ndarray
    min_train_size: int
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].astype(float)
        standardized = ((source - self.center) / self.divisor).fillna(0.0)
        if self.beta is not None:
            standardized = standardized * self.beta
        scores = standardized.to_numpy(dtype=float) @ self.directions
        if scores.shape[1] < self.n_components:
            scores = np.hstack(
                [scores, np.zeros((scores.shape[0], self.n_components - scores.shape[1]))]
            )
        result = pd.DataFrame(
            scores[:, : self.n_components],
            index=panel.index,
            columns=[f"{self.prefix}{idx}" for idx in range(1, self.n_components + 1)],
        )
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "target": self.target,
            "n_components": self.n_components,
            "resolved_n_components": self.resolved_n_components,
            "n_slices": self.n_slices,
            "resolved_n_slices": self.resolved_n_slices,
            "scaling_policy": self.scaling_policy,
            "prefix": self.prefix,
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel_target_aligned_rows",
            "missing_policy": "standardized_mean_fill_for_projection",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _PredictorScreenState:
    columns: tuple[str, ...]
    selected_columns: tuple[str, ...]
    candidate_columns: tuple[str, ...]
    controls: tuple[str, ...]
    target: str
    method: str
    threshold: float
    top_k: int | None
    min_k: int | None
    scores: dict[str, float]
    selection_params: dict[str, Any]
    min_train_size: int | None
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        result = panel.loc[:, self.selected_columns].copy()
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "selected_columns": list(self.selected_columns),
            "candidate_columns": list(self.candidate_columns),
            "controls": list(self.controls),
            "target": self.target,
            "method": self.method,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "min_k": self.min_k,
            "scores": dict(self.scores),
            "selection_params": dict(self.selection_params),
            "min_train_size": self.min_train_size,
            "fit_policy": "fixed_fit_panel_target_aligned_rows",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _FeatureSelectionState:
    columns: tuple[str, ...]
    selected_columns: tuple[str, ...]
    target: str | None
    n_features: int | float | None
    resolved_n_features: int
    method: str
    lasso_alpha: float
    scores: dict[str, float]
    selection_params: dict[str, Any]
    min_train_size: int | None
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        result = panel.loc[:, self.selected_columns].copy()
        result.index.name = "date"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "selected_columns": list(self.selected_columns),
            "target": self.target,
            "n_features": self.n_features,
            "resolved_n_features": self.resolved_n_features,
            "method": self.method,
            "lasso_alpha": self.lasso_alpha,
            "scores": dict(self.scores),
            "selection_params": dict(self.selection_params),
            "min_train_size": self.min_train_size,
            "fit_policy": (
                "fixed_fit_panel_columns"
                if self.method == "variance_selection"
                else "fixed_fit_panel_target_aligned_rows"
            ),
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _CustomFeatureState:
    columns: tuple[str, ...]
    func: Callable[..., Any] | None
    fit_func: Callable[..., Any] | None
    transform_func: Callable[..., Any] | None
    fit_state: Any
    requires_target: bool
    target: str | None
    params: dict[str, Any]
    name: str
    output_prefix: str
    n_fit_rows: int

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        source = panel.loc[:, self.columns].copy()
        if self.fit_state is not None:
            if self.transform_func is not None:
                output = self.transform_func(
                    source,
                    state=self.fit_state,
                    metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
                    **self.params,
                )
            elif hasattr(self.fit_state, "transform"):
                output = self.fit_state.transform(source)
            elif callable(self.fit_state):
                output = self.fit_state(source)
            elif self.func is not None:
                output = self.func(
                    source,
                    state=self.fit_state,
                    metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
                    **self.params,
                )
            else:
                raise TypeError("custom fitted feature step needs transform_func or fitted state with transform()")
        elif self.func is not None:
            output = self.func(
                source,
                metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
                **self.params,
            )
        else:
            raise TypeError("custom feature step requires func or fit_func")
        return _coerce_custom_step_output(output, index=source.index, prefix=self.output_prefix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "callable": _callable_name(self.func),
            "fit_callable": _callable_name(self.fit_func),
            "transform_callable": _callable_name(self.transform_func),
            "fit_state_type": None if self.fit_state is None else type(self.fit_state).__name__,
            "requires_target": self.requires_target,
            "target": self.target,
            "params": _json_ready_step(self.params),
            "fit_policy": "fixed_fit_panel_target_aligned_rows"
            if self.requires_target
            else "fixed_fit_panel",
            "n_fit_rows": self.n_fit_rows,
        }


@dataclass(frozen=True)
class _FeatureStepPlan:
    name: str
    method: str
    input_name: str
    include: bool
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready_step(
            {
                "name": self.name,
                "method": self.method,
                "input": self.input_name,
                "include": self.include,
                **self.params,
            }
        )


@dataclass(frozen=True)
class _FittedFeatureStep:
    name: str
    method: str
    input_name: str
    include: bool
    params: dict[str, Any]
    state: (
        _ScaleState
        | _PCAState
        | _SparsePCAChenRoheState
        | _VarimaxState
        | _GroupPCAState
        | _MAFState
        | _MARXState
        | _HamiltonState
        | _RandomProjectionState
        | _NystroemState
        | _PartialLeastSquaresState
        | _SlicedInverseRegressionState
        | _PredictorScreenState
        | _FeatureSelectionState
        | _CustomFeatureState
        | None
    ) = None

    def transform(self, source: pd.DataFrame) -> pd.DataFrame:
        out = _transform_feature_step(self, source)
        if out.columns.has_duplicates:
            duplicate_columns = out.columns[out.columns.duplicated()].unique()
            raise ValueError(f"feature step {self.name!r} produced duplicate columns: {list(map(str, duplicate_columns))}")
        return out

    def to_metadata(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "method": self.method,
            "input": self.input_name,
            "include": self.include,
            "params": _json_ready_step(self.params),
            "fit_state": None if self.state is None else self.state.to_dict(),
        }


def feature_spec(
    *,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    predictors: Literal["all"] | Iterable[str] | None = None,
    lags: Iterable[int] | int | None = (0, 1),
    target_lags: Iterable[int] | int | None = None,
    rolling_windows: Iterable[int] | int | None = None,
    rolling_min_periods: int | None = None,
    add_time: bool = False,
    time_trend: bool = True,
    time_month: bool = False,
    time_quarter: bool = False,
    time_year: bool = False,
    pca_components: int | None = None,
    pca_columns: Iterable[str] | None = None,
    pca_scale: bool = True,
    pca_prefix: str = "pc",
    steps: Iterable[Mapping[str, Any]] | None = None,
    feature_steps: Iterable[Mapping[str, Any]] | None = None,
    include_original: bool = False,
    target_transform: TargetTransform = "level",
    target_mode: TargetMode = "direct",
    drop_missing: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSpec:
    """Create a reusable feature-building specification.

    Parameters define the target columns, horizons, predictor columns, simple
    lag/rolling/PCA shortcuts, or an explicit ``feature_steps`` pipeline. The
    returned spec is inert until a runner calls ``fit(...)`` or
    ``fit_transform(...)`` on a training panel, so stateful steps such as PCA,
    sparse PCA, scaling, and feature selection are fitted inside the training
    window rather than on the full sample.

    ``target``/``targets`` select the source series to forecast.
    ``horizon``/``horizons`` select direct forecast horizons. ``predictors`` may
    be ``"all"``, an iterable of column names, ``None`` for metadata/default
    resolution, or an empty iterable for target-only designs. ``lags`` and
    ``target_lags`` build simple lag matrices when no explicit step pipeline is
    supplied. ``steps`` is an alias for ``feature_steps``.

    Returns
    -------
    FeatureSpec
        Frozen feature-builder configuration with ``fit``, ``fit_transform``,
        ``to_dict``, and ``to_metadata`` methods.

    Example
    -------
    >>> import macroforecast as mf
    >>> features = mf.feature_engineering.feature_spec(
    ...     target="INDPRO",
    ...     predictors=["UNRATE", "CPIAUCSL"],
    ...     horizons=[1, 3],
    ...     lags=(0, 1, 2),
    ... )
    """

    if steps is not None and feature_steps is not None:
        raise ValueError("provide either steps or feature_steps, not both")
    step_values = feature_steps if feature_steps is not None else steps
    normalized_steps = _normalize_feature_steps(step_values)
    target_lag_values = (
        ()
        if target_lags is None
        else _normalize_lags(target_lags, allow_zero=True)
    )
    if normalized_steps and (
        rolling_windows is not None
        or add_time
        or pca_components is not None
    ):
        raise ValueError(
            "feature_spec steps replace rolling/time/PCA shortcut options; "
            "use rolling_step(), pca_step(), sparse_pca_chen_rohe_step(), varimax_step(), "
            "scale_step(), group_pca_step(), or maf_step() inside steps instead"
        )

    return FeatureSpec(
        target=target,
        targets=tuple(str(value) for value in (targets or ())),
        horizon=horizon,
        horizons=_normalize_optional_positive_ints(horizons, name="horizons"),
        predictors=_normalize_predictor_input(predictors),
        lags=() if lags is None else _normalize_lags(lags, allow_zero=True),
        target_lags=target_lag_values,
        rolling_windows=()
        if rolling_windows is None
        else _normalize_positive_ints(rolling_windows, name="rolling_windows"),
        rolling_min_periods=rolling_min_periods,
        add_time=bool(add_time),
        time_trend=bool(time_trend),
        time_month=bool(time_month),
        time_quarter=bool(time_quarter),
        time_year=bool(time_year),
        pca_components=None if pca_components is None else int(pca_components),
        pca_columns=None if pca_columns is None else tuple(str(value) for value in pca_columns),
        pca_scale=bool(pca_scale),
        pca_prefix=str(pca_prefix),
        feature_steps=normalized_steps,
        include_original=bool(include_original),
        target_transform=_normalize_target_transform(target_transform),
        target_mode=_normalize_target_mode(target_mode),
        drop_missing=bool(drop_missing),
        metadata=dict(metadata or {}),
    )


def _build_predictors(
    panel: pd.DataFrame,
    fitted: FittedFeatureBuilder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fitted.step_states:
        return _build_step_predictors(panel, fitted)

    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    spec = fitted.spec

    if spec.lags and fitted.predictors:
        lagged = lag(panel, columns=fitted.predictors, lags=spec.lags)
        parts.append(lagged)
        records.extend(_records_for_columns(lagged, operation="lag", sources=fitted.predictors, included=True))
    if spec.target_lags:
        target_lagged = lag(panel, columns=fitted.targets, lags=spec.target_lags)
        parts.append(target_lagged)
        records.extend(
            _records_for_columns(
                target_lagged,
                operation="target_lag",
                sources=fitted.targets,
                included=True,
            )
        )
    if spec.rolling_windows:
        rolled = rolling_mean(
            panel,
            columns=fitted.predictors,
            windows=spec.rolling_windows,
            min_periods=spec.rolling_min_periods,
        )
        parts.append(rolled)
        records.extend(_records_for_columns(rolled, operation="rolling_mean", sources=fitted.predictors, included=True))
    if spec.add_time:
        timed = time_features(
            panel,
            trend=spec.time_trend,
            month=spec.time_month,
            quarter=spec.time_quarter,
            year=spec.time_year,
        )
        parts.append(timed)
        records.extend(_records_for_columns(timed, operation="time", sources=(), included=True))
    if fitted.pca_state is not None:
        pca = fitted.pca_state.transform(panel)
        parts.append(pca)
        records.extend(
            {
                "feature": str(column),
                "operation": "pca",
                "source": ",".join(fitted.pca_state.columns),
                "parameter": f"n_components={fitted.pca_state.n_components}",
                "component": idx,
                "inputs": ",".join(fitted.pca_state.columns),
                "included": True,
            }
            for idx, column in enumerate(pca.columns, start=1)
        )
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=panel.index)
    X.index.name = "date"
    return X, _metadata_frame(records)


def _build_step_predictors(
    panel: pd.DataFrame,
    fitted: FittedFeatureBuilder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = panel.loc[:, fitted.predictors].copy()
    root.index.name = "date"
    outputs: dict[str, pd.DataFrame] = {"panel": root}
    if fitted.targets:
        target_root = panel.loc[:, fitted.targets].copy()
        target_root.index.name = "date"
        outputs["target_panel"] = target_root
    included: list[pd.DataFrame] = []
    feature_records: list[dict[str, Any]] = []
    if fitted.spec.include_original:
        included.append(root)
        records = _records_for_columns(
            root,
            operation="original",
            sources=fitted.predictors,
            included=True,
        )
        for record in records:
            record["step"] = "panel"
        feature_records.extend(records)

    for step in fitted.step_states:
        if step.input_name not in outputs:
            raise ValueError(f"feature step {step.name!r} references unknown input {step.input_name!r}")
        source = outputs[step.input_name]
        out = step.transform(source)
        outputs[step.name] = out
        if step.include:
            included.append(out)
        records = _feature_records_for_step(
            out,
            step=step,
            source_columns=tuple(str(column) for column in source.columns),
        )
        feature_records.extend(records)

    if fitted.spec.target_lags:
        target_lagged = lag(panel, columns=fitted.targets, lags=fitted.spec.target_lags)
        included.append(target_lagged)
        records = _records_for_columns(
            target_lagged,
            operation="target_lag",
            sources=fitted.targets,
            included=True,
        )
        for record in records:
            record["step"] = "target_lags"
        feature_records.extend(records)

    if not included:
        raise ValueError("feature_spec steps produced no included feature blocks")
    X = pd.concat(included, axis=1)
    duplicate_columns = X.columns[X.columns.duplicated()].unique()
    if len(duplicate_columns):
        raise ValueError(f"feature_spec steps contain duplicate columns: {list(map(str, duplicate_columns))}")
    X.index.name = "date"
    return X, _metadata_frame(feature_records)


def _build_targets(panel: pd.DataFrame, fitted: FittedFeatureBuilder) -> pd.DataFrame:
    if fitted.spec.target_mode == "path":
        return path_targets(
            panel,
            targets=fitted.targets,
            horizons=fitted.horizons,
            transform=_target_transform_to_path_transform(fitted.spec.target_transform),
        )
    return direct_target(
        panel,
        targets=fitted.targets,
        horizons=fitted.horizons,
        transform=fitted.spec.target_transform,
    )


def _target_frame_for_feature_steps(
    panel: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    target_mode: TargetMode,
    target_transform: TargetTransform,
) -> pd.DataFrame:
    if target_mode == "path":
        return path_targets(
            panel,
            targets=targets,
            horizons=horizons,
            transform=_target_transform_to_path_transform(target_transform),
        )
    return direct_target(
        panel,
        targets=targets,
        horizons=horizons,
        transform=target_transform,
    )


def _fit_pca_state(
    panel: pd.DataFrame,
    *,
    predictors: tuple[str, ...],
    columns: tuple[str, ...] | None,
    n_components: int | None,
    scale: bool,
    prefix: str,
) -> _PCAState | None:
    if n_components is None:
        return None
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("pca_components must be positive")
    selected = columns or predictors
    missing = [column for column in selected if column not in panel.columns]
    if missing:
        raise ValueError(f"pca columns are not in the panel: {missing}")
    if n_value > len(selected):
        raise ValueError("pca_components must be <= the number of PCA columns")
    sub = panel.loc[:, list(selected)].astype(float)
    # Drop series with no observation in this window (ragged-start predictors,
    # e.g. a FRED-MD series that begins after the early estimation window). The
    # factor space is estimated from the series available at this origin.
    usable = [column for column in sub.columns if sub[column].notna().any()]
    if not usable:
        raise ValueError("all PCA columns are empty in this window")
    sub = sub.loc[:, usable]
    n_value = min(n_value, len(usable))
    selected = tuple(usable)
    train = sub.dropna()
    if len(train) < n_value + 1:
        raise ValueError("not enough complete training rows to fit PCA")
    center = None
    divisor = None
    fit_values = train
    if scale:
        center = train.mean(axis=0)
        divisor = train.std(axis=0, ddof=0).replace(0.0, np.nan).fillna(1.0)
        fit_values = (train - center) / divisor
    from sklearn.decomposition import PCA

    model = _deterministic_pca(n_value, *fit_values.shape)
    model.fit(fit_values)
    return _PCAState(
        columns=tuple(selected),
        n_components=n_value,
        scale=bool(scale),
        prefix=str(prefix),
        center=center,
        divisor=divisor,
        model=model,
    )


def _fit_feature_step_pipeline(
    panel: pd.DataFrame,
    *,
    predictors: tuple[str, ...],
    targets: tuple[str, ...],
    steps: tuple[dict[str, Any], ...],
    target_frame: pd.DataFrame | None = None,
) -> tuple[_FittedFeatureStep, ...]:
    if not steps:
        return ()
    root = panel.loc[:, predictors].copy()
    root.index.name = "date"
    outputs: dict[str, pd.DataFrame] = {"panel": root}
    if targets:
        # Explicit target-derived feature transforms are needed by forecasting
        # designs such as Goulet Coulombe et al. (2021), where MARX_y and MAF_y
        # are separate from the plain autoregressive target-lag block. The
        # resolved predictor set still rejects target overlap; users must opt in
        # by setting a feature step's input to "target_panel".
        target_root = panel.loc[:, targets].copy()
        target_root.index.name = "date"
        outputs["target_panel"] = target_root
    fitted_steps: list[_FittedFeatureStep] = []
    for step in steps:
        plan = _feature_step_plan(step)
        if plan.name in outputs:
            raise ValueError(f"duplicate feature step name: {plan.name!r}")
        if plan.input_name not in outputs:
            raise ValueError(f"feature step {plan.name!r} references unknown input {plan.input_name!r}")
        fitted = _fit_feature_step(
            outputs[plan.input_name],
            plan,
            target_frame=target_frame,
        )
        outputs[plan.name] = fitted.transform(outputs[plan.input_name])
        fitted_steps.append(fitted)
    return tuple(fitted_steps)


def _fit_feature_step(
    source: pd.DataFrame,
    plan: _FeatureStepPlan,
    *,
    target_frame: pd.DataFrame | None = None,
) -> _FittedFeatureStep:
    params = dict(plan.params)
    columns = params.pop("columns", None)
    if plan.method == "custom":
        selected = _resolve_columns(source, columns=columns)
        func = params.pop("func", params.pop("callable", None))
        fit_func = params.pop("fit_func", None)
        transform_func = params.pop("transform_func", None)
        requires_target = bool(params.pop("requires_target", False))
        min_train_size = params.pop("min_train_size", None)
        output_prefix = str(params.pop("prefix", plan.name))
        drop_missing = bool(params.pop("drop_missing", False))
        if func is not None and not callable(func):
            raise TypeError("custom feature step func must be callable")
        if fit_func is not None and not callable(fit_func):
            raise TypeError("custom feature step fit_func must be callable")
        if transform_func is not None and not callable(transform_func):
            raise TypeError("custom feature step transform_func must be callable")
        if func is None and fit_func is None:
            raise TypeError("custom feature step requires func or fit_func")
        source_selected = source.loc[:, selected].copy()
        target_name = None
        target_series = None
        fit_source = source_selected.dropna()
        if requires_target:
            target_name, target_series = _single_feature_step_target(
                target_frame,
                step_name=plan.name,
                method=plan.method,
            )
            joined = pd.concat([source_selected, target_series.rename("__target__")], axis=1).dropna()
            fit_source = joined.loc[:, selected]
            target_series = joined["__target__"]
        min_size = _normalize_min_train_size(min_train_size, minimum=1) if min_train_size is not None else 1
        if len(fit_source) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} complete rows to fit custom step"
            )
        fit_state = None
        if fit_func is not None:
            fit_state = fit_func(
                fit_source,
                target=target_series,
                metadata=dict(source.attrs.get("macroforecast_metadata", {})),
                **params,
            )
        state = _CustomFeatureState(
            columns=selected,
            func=func,
            fit_func=fit_func,
            transform_func=transform_func,
            fit_state=fit_state,
            requires_target=requires_target,
            target=target_name,
            params=dict(params),
            name=plan.name,
            output_prefix=output_prefix,
            n_fit_rows=int(len(fit_source)),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "func": func,
                "fit_func": fit_func,
                "transform_func": transform_func,
                "requires_target": requires_target,
                "min_train_size": min_train_size,
                "prefix": output_prefix,
                "drop_missing": drop_missing,
                **params,
            },
            state=state,
        )
    if plan.method == "lag":
        lag_values = _normalize_lags(params.pop("lags", params.pop("n_lag", (1,))), allow_zero=True)
        drop_missing = bool(params.pop("drop_missing", False))
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={"columns": _normalize_optional_columns(columns), "lags": lag_values, "drop_missing": drop_missing},
        )
    if plan.method == "rolling_mean":
        windows = _normalize_positive_ints(params.pop("windows", params.pop("window", (3,))), name="windows")
        min_periods = params.pop("min_periods", None)
        shift = int(params.pop("shift", 0))
        drop_missing = bool(params.pop("drop_missing", False))
        if min_periods is not None and int(min_periods) <= 0:
            raise ValueError("min_periods must be positive")
        if shift < 0:
            raise ValueError("shift must be non-negative")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": _normalize_optional_columns(columns),
                "windows": windows,
                "min_periods": min_periods,
                "shift": shift,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "moving_average_ladder":
        windows = params.pop("windows", None)
        max_window = int(params.pop("max_window", 12))
        min_periods = params.pop("min_periods", None)
        shift = int(params.pop("shift", 0))
        drop_missing = bool(params.pop("drop_missing", False))
        if windows is not None:
            windows = _normalize_positive_ints(windows, name="windows")
        if max_window <= 0:
            raise ValueError("max_window must be positive")
        if min_periods is not None and int(min_periods) <= 0:
            raise ValueError("min_periods must be positive")
        if shift < 0:
            raise ValueError("shift must be non-negative")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": _normalize_optional_columns(columns),
                "windows": windows,
                "max_window": max_window,
                "min_periods": min_periods,
                "shift": shift,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "marx":
        selected = _resolve_columns(source, columns=columns)
        max_lag = int(params.pop("max_lag", 12))
        if max_lag <= 0:
            raise ValueError("max_lag must be positive")
        scale_lags = bool(params.pop("scale_lags", False))
        min_train_size = params.pop("min_train_size", None)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        # Drop predictor columns with no observation over the fit window (ragged-start
        # series, e.g. a FRED-MD series that begins after the early estimation window,
        # or one left all-NaN by leak-free per-origin EM imputation). Mirrors the PCA
        # step (_fit_pca_state): an all-NaN predictor would otherwise produce an
        # all-NaN MARX block, and the downstream feature-level dropna (drop_missing)
        # would then wipe out EVERY row -- silently emptying the whole arm. The fitted
        # (reduced) column set is fixed for the transform, so the column set stays
        # consistent across fit/transform at this origin.
        usable = tuple(
            column for column in selected if source.loc[:, column].notna().any()
        )
        if not usable:
            raise ValueError(
                f"feature step {plan.name!r} has no MARX columns with any observation in this window"
            )
        selected = usable
        marx_min_size: int | None = None
        center = None
        divisor = None
        if scale_lags:
            marx_min_size = _normalize_min_train_size(min_train_size, minimum=2)
            lag_values = tuple(range(1, max_lag + 1))
            lag_matrix = _marx_lag_matrix(source, columns=selected, lag_values=lag_values)
            train = lag_matrix.dropna().astype(float)
            if len(train) < marx_min_size:
                raise ValueError(
                    f"feature step {plan.name!r} has fewer than {marx_min_size} complete rows to fit MARX"
                )
            # Match the author R-code option: scale each lag-matrix column
            # with sample standard deviations before taking increasing lag
            # averages. The fitted center/divisor are then fixed for test rows.
            center = train.mean(axis=0)
            divisor = train.std(axis=0, ddof=1).replace(0.0, np.nan)
        marx_state = _MARXState(
            columns=selected,
            max_lag=max_lag,
            scale_lags=scale_lags,
            min_train_size=marx_min_size,
            center=center,
            divisor=divisor,
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "max_lag": max_lag,
                "scale_lags": scale_lags,
                "min_train_size": marx_min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel" if scale_lags else "deterministic",
            },
            state=marx_state,
        )
    if plan.method == "transform":
        transform = str(params.pop("transform", plan.name))
        periods = int(params.pop("periods", 1))
        drop_missing = bool(params.pop("drop_missing", False))
        if periods <= 0:
            raise ValueError("periods must be positive")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": _normalize_optional_columns(columns),
                "transform": transform,
                "periods": periods,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "seasonal_lag":
        season_length = int(params.pop("season_length", 12))
        lag_values = _normalize_lags(params.pop("lags", (1,)), allow_zero=False)
        drop_missing = bool(params.pop("drop_missing", False))
        if season_length <= 0:
            raise ValueError("season_length must be positive")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": _normalize_optional_columns(columns),
                "season_length": season_length,
                "lags": lag_values,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "season_dummy":
        frequency = str(params.pop("frequency", "auto"))
        drop_first = bool(params.pop("drop_first", False))
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "frequency": frequency,
                "drop_first": drop_first,
                "drop_missing": False,
            },
        )
    if plan.method == "fourier":
        period = int(params.pop("period", 12))
        order = int(params.pop("order", 2))
        prefix = str(params.pop("prefix", "fourier"))
        if period <= 1:
            raise ValueError("period must be greater than 1")
        if order <= 0:
            raise ValueError("order must be positive")
        if order > period // 2:
            raise ValueError("order must be <= period // 2")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "period": period,
                "order": order,
                "prefix": prefix,
                "drop_missing": False,
            },
        )
    if plan.method == "polynomial":
        degree = int(params.pop("degree", 2))
        include_bias = bool(params.pop("include_bias", False))
        interaction_only = bool(params.pop("interaction_only", False))
        drop_missing = bool(params.pop("drop_missing", False))
        if degree < 1:
            raise ValueError("degree must be at least 1")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": _normalize_optional_columns(columns),
                "degree": degree,
                "include_bias": include_bias,
                "interaction_only": interaction_only,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "interaction":
        selected = _resolve_columns(source, columns=columns)
        order = int(params.pop("order", 2))
        drop_missing = bool(params.pop("drop_missing", False))
        if order < 2:
            raise ValueError("order must be at least 2 for interaction features")
        if order > len(selected):
            raise ValueError("order must be <= the number of selected columns")
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "order": order,
                "drop_missing": drop_missing,
            },
        )
    if plan.method == "time":
        trend = bool(params.pop("trend", True))
        month = bool(params.pop("month", False))
        quarter = bool(params.pop("quarter", False))
        year = bool(params.pop("year", False))
        _reject_extra_params(params, plan.name)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={"trend": trend, "month": month, "quarter": quarter, "year": year},
        )
    if plan.method == "scale":
        selected = _resolve_columns(source, columns=columns)
        method = _normalize_scale_method(str(params.pop("scale_method", params.pop("method_name", "zscore"))))
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=2)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        train = source.loc[:, selected].dropna().astype(float)
        if len(train) < min_size:
            raise ValueError(f"feature step {plan.name!r} has fewer than {min_size} complete rows to fit scaling")
        center, divisor = _scale_parameters(train, method=method)
        scale_state = _ScaleState(columns=selected, method=method, center=center, divisor=divisor)
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "scale_method": method,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=scale_state,
        )
    if plan.method == "pca":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 1))
        params.pop("fit_policy", None)
        min_train_size = params.pop("min_train_size", None)
        scale = bool(params.pop("scale", True))
        prefix = str(params.pop("prefix", plan.name))
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        pca_state = _fit_pca_state(
            source,
            predictors=selected,
            columns=selected,
            n_components=n_components,
            scale=scale,
            prefix=prefix,
        )
        if pca_state is None:
            raise ValueError(f"feature step {plan.name!r} did not fit PCA state")
        if min_train_size is not None:
            min_size = _normalize_min_train_size(min_train_size, minimum=n_components + 1)
            if len(source.loc[:, selected].dropna()) < min_size:
                raise ValueError(f"feature step {plan.name!r} has fewer than {min_size} complete rows to fit PCA")
        if random_state is not None:
            pca_state = _fit_pca_state_with_random_state(
                source,
                columns=selected,
                n_components=n_components,
                scale=scale,
                prefix=prefix,
                random_state=int(random_state),
                min_train_size=min_train_size,
                step_name=plan.name,
            )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "n_components": n_components,
                "scale": scale,
                "prefix": prefix,
                "min_train_size": min_train_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=pca_state,
        )
    if plan.method == "sparse_pca_chen_rohe":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 4))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        zeta = float(params.pop("zeta", 0.0))
        if zeta < 0:
            raise ValueError("zeta must be non-negative")
        max_iter = int(params.pop("max_iter", 200))
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        var_innovations = bool(params.pop("var_innovations", False))
        prefix = str(params.pop("prefix", plan.name))
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=3 if var_innovations else 1)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", 0)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        train = source.loc[:, selected].dropna().astype(float)
        if len(train) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} complete rows "
                "to fit Chen-Rohe sparse components"
            )
        center, theta, zeta_resolved, n_iter, objective = _fit_sparse_pca_chen_rohe(
            train,
            n_components=n_components,
            zeta=zeta,
            max_iter=max_iter,
            random_state=None if random_state is None else int(random_state),
        )
        train_scores = (train - center).to_numpy(dtype=float) @ theta
        var_coef = _fit_sparse_factor_var1(train_scores) if var_innovations else None
        if var_innovations and var_coef is None:
            raise ValueError("var_innovations requires at least three complete rows")
        sparse_state = _SparsePCAChenRoheState(
            columns=selected,
            n_components=n_components,
            resolved_n_components=int(theta.shape[1]),
            zeta=zeta,
            zeta_resolved=float(zeta_resolved),
            max_iter=max_iter,
            var_innovations=var_innovations,
            prefix=prefix,
            random_state=None if random_state is None else int(random_state),
            min_train_size=min_size,
            center=center,
            theta=theta,
            var_coef=var_coef,
            n_fit_rows=int(len(train)),
            n_iter=int(n_iter),
            objective=float(objective),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "n_components": n_components,
                "resolved_n_components": int(theta.shape[1]),
                "zeta": zeta,
                "zeta_resolved": float(zeta_resolved),
                "max_iter": max_iter,
                "var_innovations": var_innovations,
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "random_state": None if random_state is None else int(random_state),
                "fit_policy": "fixed_fit_panel",
            },
            state=sparse_state,
        )
    if plan.method == "varimax":
        selected = _resolve_columns(source, columns=columns)
        max_iter = int(params.pop("max_iter", 50))
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        tol = float(params.pop("tol", 1e-7))
        if tol < 0:
            raise ValueError("tol must be non-negative")
        prefix = str(params.pop("prefix", plan.name))
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=1)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        train = source.loc[:, selected].dropna().astype(float)
        if len(train) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} complete rows to fit varimax rotation"
            )
        rotation, n_iter = _fit_varimax_rotation(train, max_iter=max_iter, tol=tol)
        varimax_state = _VarimaxState(
            columns=selected,
            max_iter=max_iter,
            tol=tol,
            prefix=prefix,
            min_train_size=min_size,
            rotation=rotation,
            n_fit_rows=int(len(train)),
            n_iter=int(n_iter),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "max_iter": max_iter,
                "tol": tol,
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=varimax_state,
        )
    if plan.method == "group_pca":
        groups = params.pop("groups", None)
        if groups is None:
            raise ValueError(f"feature step {plan.name!r} requires groups")
        group_map = _normalize_column_groups(groups)
        component_counts = _resolve_group_components(params.pop("n_components", 1), groups=group_map)
        params.pop("fit_policy", None)
        min_train_size = params.pop("min_train_size", None)
        scale = bool(params.pop("scale", True))
        prefix = params.pop("prefix", None)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        states: dict[str, _PCAState] = {}
        resolved_groups: dict[str, tuple[str, ...]] = {}
        for group_name, group_columns in group_map.items():
            selected = _resolve_columns(source, columns=group_columns)
            resolved_groups[group_name] = selected
            n_value = component_counts[group_name]
            min_size = _normalize_min_train_size(min_train_size, minimum=n_value + 1)
            if len(source.loc[:, selected].dropna()) < min_size:
                raise ValueError(
                    f"feature step {plan.name!r} group {group_name!r} has fewer than "
                    f"{min_size} complete rows to fit PCA"
                )
            states[group_name] = _fit_pca_state_with_random_state(
                source,
                columns=selected,
                n_components=n_value,
                scale=scale,
                prefix=_group_component_prefix(group_name, prefix=prefix),
                random_state=None if random_state is None else int(random_state),
                min_train_size=min_size,
                step_name=plan.name,
            )
        group_state = _GroupPCAState(
            groups=resolved_groups,
            states=states,
            n_components=component_counts,
            scale=scale,
            prefix=None if prefix is None else str(prefix),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "groups": {group: list(cols) for group, cols in resolved_groups.items()},
                "n_components": dict(component_counts),
                "scale": scale,
                "prefix": None if prefix is None else str(prefix),
                "min_train_size": min_train_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=group_state,
        )
    if plan.method == "maf":
        selected = _resolve_columns(source, columns=columns)
        max_lag = int(params.pop("max_lag", 12))
        lag_values = _normalize_maf_lags(max_lag=max_lag, lags=params.pop("lags", None))
        n_components = int(params.pop("n_components", 2))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_components > len(lag_values):
            raise ValueError("n_components must be <= the number of MAF lag columns")
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=n_components + 1)
        scale = bool(params.pop("scale", False))
        prefix = str(params.pop("prefix", "maf"))
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        column_states: list[_MAFColumnState] = []
        for column in selected:
            lag_block = pd.DataFrame(
                {
                    f"{column}_lag{lag_value}": source[column].shift(lag_value)
                    for lag_value in lag_values
                },
                index=source.index,
            )
            if len(lag_block.dropna()) < min_size:
                raise ValueError(
                    f"feature step {plan.name!r} column {column!r} has fewer than "
                    f"{min_size} complete rows to fit MAF"
                )
            pca_state = _fit_pca_state_with_random_state(
                lag_block,
                columns=tuple(lag_block.columns),
                n_components=n_components,
                scale=scale,
                prefix=_maf_component_prefix(column, prefix=prefix),
                random_state=None if random_state is None else int(random_state),
                min_train_size=min_size,
                step_name=plan.name,
            )
            column_states.append(_MAFColumnState(source_column=column, lag_values=lag_values, pca_state=pca_state))
        maf_state = _MAFState(
            columns=selected,
            lag_values=lag_values,
            n_components=n_components,
            scale=scale,
            prefix=prefix,
            column_states=tuple(column_states),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "lags": lag_values,
                "max_lag": max_lag,
                "n_components": n_components,
                "scale": scale,
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=maf_state,
        )
    if plan.method == "hamilton_filter":
        selected = _resolve_columns(source, columns=columns)
        h_value = int(params.pop("h", 8))
        p_value = int(params.pop("p", 4))
        component = str(params.pop("component", "cycle")).lower()
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=p_value + 1)
        missing = str(params.pop("missing", "drop")).lower()
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        if h_value <= 0:
            raise ValueError("h must be positive")
        if p_value <= 0:
            raise ValueError("p must be positive")
        if component not in {"cycle", "trend", "both"}:
            raise ValueError("component must be 'cycle', 'trend', or 'both'")
        if missing != "drop":
            raise ValueError(
                "hamilton_step() inside feature_spec() only supports missing='drop'. "
                "Impute or reprocess the panel before runner-safe feature construction."
            )
        _reject_extra_params(params, plan.name)
        beta_by_column: dict[str, np.ndarray] = {}
        fit_rows_by_column: dict[str, int] = {}
        for column in selected:
            _, x_matrix, y_vector = _hamilton_design_arrays(source[column], h=h_value, p=p_value)
            if len(y_vector) < min_size:
                raise ValueError(
                    f"feature step {plan.name!r} column {column!r} has fewer than "
                    f"{min_size} complete rows to fit Hamilton filter"
                )
            beta_by_column[column] = _ols_beta(x_matrix, y_vector)
            fit_rows_by_column[column] = int(len(y_vector))
        hamilton_state = _HamiltonState(
            columns=selected,
            h=h_value,
            p=p_value,
            component=component,
            min_train_size=min_size,
            beta_by_column=beta_by_column,
            fit_rows_by_column=fit_rows_by_column,
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "h": h_value,
                "p": p_value,
                "component": component,
                "min_train_size": min_size,
                "missing": missing,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=hamilton_state,
        )
    if plan.method == "random_projection":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 2))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        random_state = params.pop("random_state", None)
        prefix = str(params.pop("prefix", plan.name))
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=1)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        train = source.loc[:, selected].dropna().astype(float)
        if len(train) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} complete rows "
                "to fit random projection"
            )
        from sklearn.random_projection import GaussianRandomProjection

        transformer = GaussianRandomProjection(
            n_components=n_components,
            random_state=None if random_state is None else int(random_state),
        )
        transformer.fit(train.to_numpy(dtype=float))
        random_projection_state = _RandomProjectionState(
            columns=selected,
            n_components=n_components,
            prefix=prefix,
            random_state=None if random_state is None else int(random_state),
            min_train_size=min_size,
            transformer=transformer,
            n_fit_rows=int(len(train)),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "n_components": n_components,
                "random_state": None if random_state is None else int(random_state),
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=random_projection_state,
        )
    if plan.method == "nystroem":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 10))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        kernel = str(params.pop("kernel", "rbf"))
        gamma = params.pop("gamma", None)
        random_state = params.pop("random_state", None)
        prefix = str(params.pop("prefix", plan.name))
        params.pop("fit_policy", None)
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=n_components)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        train = source.loc[:, selected].dropna().astype(float)
        if len(train) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} complete rows to fit Nystroem"
            )
        from sklearn.kernel_approximation import Nystroem

        transformer = Nystroem(
            kernel=kernel,
            gamma=None if gamma is None else float(gamma),
            n_components=n_components,
            random_state=None if random_state is None else int(random_state),
        )
        transformer.fit(train.to_numpy(dtype=float))
        nystroem_state = _NystroemState(
            columns=selected,
            n_components=n_components,
            kernel=kernel,
            gamma=None if gamma is None else float(gamma),
            prefix=prefix,
            random_state=None if random_state is None else int(random_state),
            min_train_size=min_size,
            transformer=transformer,
            n_fit_rows=int(len(train)),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "n_components": n_components,
                "kernel": kernel,
                "gamma": None if gamma is None else float(gamma),
                "random_state": None if random_state is None else int(random_state),
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel",
            },
            state=nystroem_state,
        )
    if plan.method == "partial_least_squares":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 3))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        prefix = str(params.pop("prefix", plan.name))
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=2)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("fit_policy", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        target_name, target_series = _single_feature_step_target(
            target_frame,
            step_name=plan.name,
            method=plan.method,
        )
        joined = pd.concat(
            [source.loc[:, selected].astype(float), target_series.rename("__target__")],
            axis=1,
        ).dropna()
        if len(joined) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} target-aligned complete rows "
                "to fit partial least squares"
            )
        resolved = _effective_pls_components(joined.loc[:, selected], n_components)
        from sklearn.cross_decomposition import PLSRegression

        model = None
        if resolved > 0:
            model = PLSRegression(n_components=resolved)
            model.fit(
                joined.loc[:, selected].to_numpy(dtype=float),
                joined["__target__"].to_numpy(dtype=float).reshape(-1, 1),
            )
        pls_state = _PartialLeastSquaresState(
            columns=selected,
            target=target_name,
            n_components=n_components,
            resolved_n_components=resolved,
            prefix=prefix,
            model=model,
            min_train_size=min_size,
            n_fit_rows=int(len(joined)),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "target": target_name,
                "n_components": n_components,
                "resolved_n_components": resolved,
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel_target_aligned_rows",
            },
            state=pls_state,
        )
    if plan.method == "sliced_inverse_regression":
        selected = _resolve_columns(source, columns=columns)
        n_components = int(params.pop("n_components", 3))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        n_slices = int(params.pop("n_slices", 10))
        if n_slices < 2:
            raise ValueError("n_slices must be at least 2")
        scaling_policy = str(params.pop("scaling_policy", "scaled_pca"))
        if scaling_policy not in {"scaled_pca", "marginal_R2", "none"}:
            raise ValueError("scaling_policy must be 'scaled_pca', 'marginal_R2', or 'none'")
        prefix = str(params.pop("prefix", plan.name))
        min_size = _normalize_min_train_size(params.pop("min_train_size", None), minimum=2)
        drop_missing = bool(params.pop("drop_missing", False))
        params.pop("fit_policy", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        target_name, target_series = _single_feature_step_target(
            target_frame,
            step_name=plan.name,
            method=plan.method,
        )
        joined = pd.concat(
            [source.loc[:, selected].astype(float), target_series.rename("__target__")],
            axis=1,
        ).dropna()
        if len(joined) < min_size:
            raise ValueError(
                f"feature step {plan.name!r} has fewer than {min_size} target-aligned complete rows "
                "to fit sliced inverse regression"
            )
        train_x = joined.loc[:, selected]
        train_y = joined["__target__"]
        resolved_components = min(n_components, train_x.shape[1])
        center = train_x.mean(axis=0)
        divisor = train_x.std(axis=0, ddof=1).replace(0.0, np.nan).fillna(1.0)
        x_scaled = (train_x - center) / divisor
        beta = None
        if scaling_policy in {"scaled_pca", "marginal_R2"}:
            beta = np.array(
                [_univariate_slope(x_scaled[column], train_y) for column in x_scaled.columns],
                dtype=float,
            )
            if scaling_policy == "marginal_R2":
                beta = np.sign(beta) * np.abs(beta)
            x_scaled = x_scaled * beta
        order = np.argsort(train_y.to_numpy(dtype=float))
        z_sorted = x_scaled.to_numpy(dtype=float)[order]
        n_total = z_sorted.shape[0]
        resolved_slices = min(n_slices, n_total)
        slice_size = max(1, n_total // resolved_slices)
        slice_means: list[np.ndarray] = []
        slice_weights: list[float] = []
        for slice_index in range(resolved_slices):
            start = slice_index * slice_size
            end = (slice_index + 1) * slice_size if slice_index < resolved_slices - 1 else n_total
            values = z_sorted[start:end]
            if values.size == 0:
                slice_means.append(np.zeros(z_sorted.shape[1]))
                slice_weights.append(0.0)
            else:
                slice_means.append(values.mean(axis=0))
                slice_weights.append(values.shape[0] / max(n_total, 1))
        mean_matrix = np.vstack(slice_means)
        weights = np.asarray(slice_weights, dtype=float)
        between_slice_cov = (mean_matrix * weights[:, None]).T @ mean_matrix
        try:
            values, vectors = np.linalg.eigh(between_slice_cov)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate numerical fallback
            values = np.zeros(between_slice_cov.shape[0])
            vectors = np.eye(between_slice_cov.shape[0])
        selected_order = np.argsort(-np.abs(values))[:resolved_components]
        directions = vectors[:, selected_order]
        for component_index in range(directions.shape[1]):
            max_index = int(np.argmax(np.abs(directions[:, component_index])))
            if directions[max_index, component_index] < 0:
                directions[:, component_index] = -directions[:, component_index]
        sir_state = _SlicedInverseRegressionState(
            columns=selected,
            target=target_name,
            n_components=n_components,
            resolved_n_components=resolved_components,
            n_slices=n_slices,
            resolved_n_slices=resolved_slices,
            scaling_policy=scaling_policy,
            prefix=prefix,
            center=center,
            divisor=divisor,
            beta=beta,
            directions=directions,
            min_train_size=min_size,
            n_fit_rows=int(len(joined)),
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "target": target_name,
                "n_components": n_components,
                "resolved_n_components": resolved_components,
                "n_slices": n_slices,
                "resolved_n_slices": resolved_slices,
                "scaling_policy": scaling_policy,
                "prefix": prefix,
                "min_train_size": min_size,
                "drop_missing": drop_missing,
                "fit_policy": "fixed_fit_panel_target_aligned_rows",
            },
            state=sir_state,
        )
    if plan.method == "predictor_screen":
        selected = _resolve_columns(source, columns=columns)
        screen_method = str(params.pop("screen_method", "t_stat"))
        threshold = params.pop("threshold", None)
        top_k = params.pop("top_k", None)
        min_k = params.pop("min_k", None)
        controls = params.pop("controls", None)
        alpha = float(params.pop("alpha", 0.001))
        l1_ratio = float(params.pop("l1_ratio", 0.5))
        lambda_search = params.pop("lambda_search", None)
        max_iter = int(params.pop("max_iter", 20000))
        min_train_size = params.pop("min_train_size", None)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", 0)
        params.pop("fit_policy", None)
        params.pop("warn_full_sample", None)
        _reject_extra_params(params, plan.name)
        target_name, target_series = _single_feature_step_target(
            target_frame,
            step_name=plan.name,
            method=plan.method,
        )
        screen = fit_predictor_screen(
            source,
            target_series,
            columns=selected,
            method=screen_method,
            threshold=None if threshold is None else float(threshold),
            top_k=None if top_k is None else int(top_k),
            min_k=None if min_k is None else int(min_k),
            controls=None if controls is None else tuple(str(value) for value in controls),
            alpha=alpha,
            l1_ratio=l1_ratio,
            lambda_search=lambda_search,
            max_iter=max_iter,
            random_state=None if random_state is None else int(random_state),
            min_train_size=min_train_size,
        )
        screen_columns = tuple(
            column
            for column in source.columns
            if column in set(screen.candidate_columns).union(screen.controls)
        )
        screen_state = _PredictorScreenState(
            columns=screen_columns,
            selected_columns=screen.selected_columns,
            candidate_columns=screen.candidate_columns,
            controls=screen.controls,
            target=target_name,
            method=screen.method,
            threshold=screen.threshold,
            top_k=screen.top_k,
            min_k=screen.min_k,
            scores=screen.scores,
            selection_params=screen.metadata,
            min_train_size=min_train_size,
            n_fit_rows=screen.n_fit_rows,
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "selected_columns": screen.selected_columns,
                "candidate_columns": screen.candidate_columns,
                "controls": screen.controls,
                "target": target_name,
                "screen_method": screen.method,
                "threshold": screen.threshold,
                "top_k": screen.top_k,
                "min_k": screen.min_k,
                "selection_params": screen.metadata,
                "min_train_size": min_train_size,
                "drop_missing": drop_missing,
                "random_state": None if random_state is None else int(random_state),
                "fit_policy": "fixed_fit_panel_target_aligned_rows",
            },
            state=screen_state,
        )
    if plan.method in {
        "variance_selection",
        "correlation_selection",
        "lasso_selection",
        "lasso_path_selection",
        "rfe_selection",
        "boruta_selection",
        "stability_selection",
        "genetic_selection",
    }:
        selected = _resolve_columns(source, columns=columns)
        n_features = params.pop("n_features", 0.5)
        method = normalize_feature_selection_method(plan.method)
        lasso_alpha = float(params.pop("lasso_alpha", params.get("alpha", 0.001)))
        if method == "lasso_selection" and "alpha" not in params:
            params["lasso_alpha"] = lasso_alpha
        min_train_size = params.pop("min_train_size", None)
        drop_missing = bool(params.pop("drop_missing", False))
        random_state = params.pop("random_state", 0)
        params.pop("fit_policy", None)
        params.pop("warn_full_sample", None)
        source_numeric = source.loc[:, selected].astype(float)
        target_name = None
        target_series = None
        if method != "variance_selection":
            target_name, target_series = _single_feature_step_target(
                target_frame,
                step_name=plan.name,
                method=plan.method,
            )
        selection = select_features(
            source_numeric,
            target_series,
            n_features=n_features,
            method=method,
            min_train_size=min_train_size,
            random_state=None if random_state is None else int(random_state),
            **params,
        )
        keep = selection.selected_columns
        resolved_lasso_alpha = float(selection.metadata.get("alpha", lasso_alpha))
        selection_state = _FeatureSelectionState(
            columns=selected,
            selected_columns=keep,
            target=target_name,
            n_features=n_features,
            resolved_n_features=selection.resolved_n_features,
            method=method,
            lasso_alpha=resolved_lasso_alpha,
            scores=selection.scores,
            selection_params=selection.metadata,
            min_train_size=min_train_size,
            n_fit_rows=selection.n_fit_rows,
        )
        return _FittedFeatureStep(
            name=plan.name,
            method=plan.method,
            input_name=plan.input_name,
            include=plan.include,
            params={
                "columns": selected,
                "selected_columns": keep,
                "target": target_name,
                "n_features": n_features,
                "resolved_n_features": selection.resolved_n_features,
                "method": method,
                "lasso_alpha": resolved_lasso_alpha,
                "selection_params": selection.metadata,
                "min_train_size": min_train_size,
                "drop_missing": drop_missing,
                "random_state": None if random_state is None else int(random_state),
                "fit_policy": selection.fit_policy,
            },
            state=selection_state,
        )
    raise ValueError(f"unsupported feature method {plan.method!r}")


def _fit_pca_state_with_random_state(
    panel: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    n_components: int,
    scale: bool,
    prefix: str,
    random_state: int | None,
    min_train_size: int | None,
    step_name: str,
) -> _PCAState:
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    if n_value > len(columns):
        raise ValueError("n_components must be <= the number of selected columns")
    min_size = _normalize_min_train_size(min_train_size, minimum=n_value + 1)
    train = panel.loc[:, columns].dropna().astype(float)
    if len(train) < min_size:
        raise ValueError(f"feature step {step_name!r} has fewer than {min_size} complete rows to fit PCA")
    center = None
    divisor = None
    fit_values = train
    if scale:
        center = train.mean(axis=0)
        divisor = train.std(axis=0, ddof=0).replace(0.0, np.nan).fillna(1.0)
        fit_values = (train - center) / divisor
    from sklearn.decomposition import PCA

    model = _deterministic_pca(n_value, *fit_values.shape, random_state=random_state)
    model.fit(fit_values)
    return _PCAState(
        columns=tuple(columns),
        n_components=n_value,
        scale=bool(scale),
        prefix=str(prefix),
        center=center,
        divisor=divisor,
        model=model,
    )


def _single_feature_step_target(
    target_frame: pd.DataFrame | None,
    *,
    step_name: str,
    method: str,
) -> tuple[str, pd.Series]:
    if target_frame is None or target_frame.empty:
        raise ValueError(
            f"feature step {step_name!r} with method {method!r} requires a target; "
            "set target/horizon on feature_spec()"
        )
    if target_frame.shape[1] != 1:
        raise ValueError(
            f"feature step {step_name!r} with method {method!r} requires exactly one "
            "resolved target column. Use one target and one horizon for target-aware steps."
        )
    target_name = str(target_frame.columns[0])
    return target_name, target_frame.iloc[:, 0].astype(float)


def _univariate_slope(x: pd.Series, y: pd.Series) -> float:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 2:
        return 0.0
    x_values = x.loc[common].astype(float).to_numpy()
    y_values = y.loc[common].astype(float).to_numpy()
    x_centered = x_values - float(np.mean(x_values))
    y_centered = y_values - float(np.mean(y_values))
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def _resolve_feature_keep_count(n_features: int | float, *, n_columns: int) -> int:
    if n_columns <= 0:
        raise ValueError("at least one column is required")
    if isinstance(n_features, float) and 0.0 < n_features <= 1.0:
        return max(1, min(n_columns, int(np.ceil(n_features * n_columns))))
    count = int(n_features)
    if count <= 0:
        raise ValueError("n_features must be a positive count or a fraction in (0, 1]")
    return max(1, min(n_columns, count))


def _transform_feature_step(step: _FittedFeatureStep, source: pd.DataFrame) -> pd.DataFrame:
    params = dict(step.params)
    drop_missing = bool(params.pop("drop_missing", False))
    if step.method == "lag":
        out = lag(source, columns=params["columns"], lags=params["lags"], drop_missing=False)
    elif step.method == "rolling_mean":
        out = rolling_mean(
            source,
            columns=params["columns"],
            windows=params["windows"],
            min_periods=params["min_periods"],
            shift=params["shift"],
            drop_missing=False,
        )
    elif step.method == "moving_average_ladder":
        out = moving_average_ladder(
            source,
            columns=params["columns"],
            windows=params["windows"],
            max_window=params["max_window"],
            min_periods=params["min_periods"],
            shift=params["shift"],
            drop_missing=False,
        )
    elif step.method == "transform":
        out = transform_features(
            source,
            columns=params["columns"],
            transform=params["transform"],
            periods=params["periods"],
            drop_missing=False,
        )
    elif step.method == "seasonal_lag":
        out = seasonal_lag(
            source,
            columns=params["columns"],
            season_length=params["season_length"],
            lags=params["lags"],
            drop_missing=False,
        )
    elif step.method == "season_dummy":
        out = season_dummy(
            source,
            frequency=params["frequency"],
            drop_first=params["drop_first"],
        )
    elif step.method == "fourier":
        out = fourier_features(
            source,
            period=params["period"],
            order=params["order"],
            prefix=params["prefix"],
        )
    elif step.method == "polynomial":
        out = polynomial_features(
            source,
            columns=params["columns"],
            degree=params["degree"],
            include_bias=params["include_bias"],
            interaction_only=params["interaction_only"],
            drop_missing=False,
        )
    elif step.method == "interaction":
        out = interaction_features(
            source,
            columns=params["columns"],
            order=params["order"],
            drop_missing=False,
        )
    elif step.method == "time":
        out = time_features(
            source,
            trend=params["trend"],
            month=params["month"],
            quarter=params["quarter"],
            year=params["year"],
        )
    elif isinstance(step.state, _ScaleState):
        out = step.state.transform(source)
    elif isinstance(step.state, _PCAState):
        out = step.state.transform(source)
    elif isinstance(step.state, _SparsePCAChenRoheState):
        out = step.state.transform(source)
    elif isinstance(step.state, _VarimaxState):
        out = step.state.transform(source)
    elif isinstance(step.state, _GroupPCAState):
        out = step.state.transform(source)
    elif isinstance(step.state, _MAFState):
        out = step.state.transform(source)
    elif isinstance(step.state, _MARXState):
        out = step.state.transform(source)
    elif isinstance(step.state, _HamiltonState):
        out = step.state.transform(source)
    elif isinstance(step.state, _RandomProjectionState):
        out = step.state.transform(source)
    elif isinstance(step.state, _NystroemState):
        out = step.state.transform(source)
    elif isinstance(step.state, _PartialLeastSquaresState):
        out = step.state.transform(source)
    elif isinstance(step.state, _SlicedInverseRegressionState):
        out = step.state.transform(source)
    elif isinstance(step.state, _PredictorScreenState):
        out = step.state.transform(source)
    elif isinstance(step.state, _FeatureSelectionState):
        out = step.state.transform(source)
    elif isinstance(step.state, _CustomFeatureState):
        out = step.state.transform(source)
    else:
        raise ValueError(f"feature step {step.name!r} has no fitted state for method {step.method!r}")
    if drop_missing:
        out = out.dropna()
    out.index.name = "date"
    return out


def _feature_records_for_step(
    out: pd.DataFrame,
    *,
    step: _FittedFeatureStep,
    source_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]]
    if isinstance(step.state, _ScaleState):
        records = [
            {
                "feature": str(feature),
                "operation": "scale",
                "source": str(feature).removesuffix(f"_{step.state.method}"),
                "parameter": f"method={step.state.method}",
                "fit_policy": "fixed_fit_panel",
                "inputs": ",".join(step.state.columns),
                "included": step.include,
            }
            for feature in out.columns
        ]
    elif isinstance(step.state, _PCAState):
        records = _component_records(
            out,
            operation="pca",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel",
            included=step.include,
        )
    elif isinstance(step.state, _SparsePCAChenRoheState):
        records = _component_records(
            out,
            operation="sparse_pca_chen_rohe",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel",
            included=step.include,
        )
        for record in records:
            record["parameter"] = (
                f"component={record.get('component')};zeta={step.state.zeta};"
                f"zeta_resolved={step.state.zeta_resolved};var_innovations={step.state.var_innovations}"
            )
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _VarimaxState):
        records = _component_records(
            out,
            operation="varimax",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel",
            included=step.include,
        )
        for record in records:
            record["parameter"] = (
                f"component={record.get('component')};max_iter={step.state.max_iter};tol={step.state.tol}"
            )
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _GroupPCAState):
        records = []
        for group_name, group_columns in step.state.groups.items():
            state = step.state.states[group_name]
            for idx in range(1, state.n_components + 1):
                feature = f"{state.prefix}{idx}"
                if feature not in out.columns:
                    continue
                records.append(
                    {
                        "feature": feature,
                        "operation": "group_pca",
                        "source": group_name,
                        "parameter": f"columns={list(group_columns)};component={idx}",
                        "component": idx,
                        "fit_policy": "fixed_fit_panel",
                        "inputs": ",".join(group_columns),
                        "included": step.include,
                    }
                )
    elif isinstance(step.state, _MAFState):
        records = []
        for column_state in step.state.column_states:
            prefix = column_state.pca_state.prefix
            for idx in range(1, column_state.pca_state.n_components + 1):
                feature = f"{prefix}{idx}"
                if feature not in out.columns:
                    continue
                records.append(
                    {
                        "feature": feature,
                        "operation": "maf",
                        "source": column_state.source_column,
                        "parameter": f"lags={list(column_state.lag_values)};component={idx}",
                        "component": idx,
                        "fit_policy": "fixed_fit_panel",
                        "inputs": ",".join(f"{column_state.source_column}_lag{lag}" for lag in column_state.lag_values),
                        "included": step.include,
                    }
                )
    elif isinstance(step.state, _MARXState):
        records = []
        for column in step.state.columns:
            for lag_order in range(1, step.state.max_lag + 1):
                records.append(
                    {
                        "feature": f"{column}_ma{lag_order}_lag1",
                        "operation": "marx",
                        "source": column,
                        "parameter": f"window={lag_order};lag=1;scale_lags={step.state.scale_lags}",
                        "lag": 1,
                        "window": lag_order,
                        "fit_policy": "fixed_fit_panel" if step.state.scale_lags else None,
                        "inputs": ",".join(f"{column}_lag{lag}" for lag in range(1, lag_order + 1)),
                        "included": step.include,
                    }
                )
    elif isinstance(step.state, _HamiltonState):
        records = []
        for column in step.state.columns:
            components = []
            if step.state.component in {"cycle", "both"}:
                components.append("cycle")
            if step.state.component in {"trend", "both"}:
                components.append("trend")
            for component in components:
                records.append(
                    {
                        "feature": f"{column}_hamilton_{component}",
                        "operation": "hamilton_filter",
                        "source": column,
                        "parameter": f"h={step.state.h};p={step.state.p};component={step.state.component}",
                        "fit_policy": "fixed_fit_panel",
                        "inputs": column,
                        "included": step.include,
                        "h": step.state.h,
                        "p": step.state.p,
                        "fit_rows": step.state.fit_rows_by_column.get(column),
                        "label_alignment": "components are labeled at t+h",
                    }
                )
    elif isinstance(step.state, _RandomProjectionState):
        records = _component_records(
            out,
            operation="random_projection",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel",
            included=step.include,
        )
        for record in records:
            record["parameter"] = f"n_components={step.state.n_components};random_state={step.state.random_state}"
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _NystroemState):
        records = _component_records(
            out,
            operation="nystroem",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel",
            included=step.include,
        )
        for record in records:
            record["parameter"] = (
                f"n_components={step.state.n_components};kernel={step.state.kernel};gamma={step.state.gamma}"
            )
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _PartialLeastSquaresState):
        records = _component_records(
            out,
            operation="partial_least_squares",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel_target_aligned_rows",
            included=step.include,
        )
        for record in records:
            record["parameter"] = f"component={record.get('component')};target={step.state.target}"
            record["target"] = step.state.target
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _SlicedInverseRegressionState):
        records = _component_records(
            out,
            operation="sliced_inverse_regression",
            source=",".join(step.state.columns),
            inputs=step.state.columns,
            fit_policy="fixed_fit_panel_target_aligned_rows",
            included=step.include,
        )
        for record in records:
            record["parameter"] = (
                f"component={record.get('component')};target={step.state.target};"
                f"n_slices={step.state.n_slices};scaling_policy={step.state.scaling_policy}"
            )
            record["target"] = step.state.target
            record["n_fit_rows"] = step.state.n_fit_rows
    elif isinstance(step.state, _PredictorScreenState):
        records = []
        control_set = set(step.state.controls)
        for rank, column in enumerate(step.state.selected_columns, start=1):
            records.append(
                {
                    "feature": column,
                    "operation": "predictor_screen",
                    "source": column,
                    "parameter": f"method={step.state.method};rank={rank};threshold={step.state.threshold}",
                    "fit_policy": "fixed_fit_panel_target_aligned_rows",
                    "inputs": ",".join(step.state.columns),
                    "included": step.include,
                    "target": step.state.target,
                    "score": step.state.scores.get(column),
                    "selection_method": step.state.method,
                    "screen_control": column in control_set,
                    "n_fit_rows": step.state.n_fit_rows,
                }
            )
    elif isinstance(step.state, _FeatureSelectionState):
        records = []
        for rank, column in enumerate(step.state.selected_columns, start=1):
            records.append(
                {
                    "feature": column,
                    "operation": step.method,
                    "source": column,
                    "parameter": f"method={step.state.method};rank={rank}",
                    "fit_policy": (
                        "fixed_fit_panel_columns"
                        if step.state.method == "variance_selection"
                        else "fixed_fit_panel_target_aligned_rows"
                    ),
                    "inputs": ",".join(step.state.columns),
                    "included": step.include,
                    "target": step.state.target,
                    "score": step.state.scores.get(column),
                    "selection_method": step.state.method,
                    "n_fit_rows": step.state.n_fit_rows,
                }
            )
    elif isinstance(step.state, _CustomFeatureState):
        records = _records_for_columns(
            out,
            operation="custom",
            sources=step.state.columns,
            included=step.include,
        )
        for record in records:
            record["parameter"] = f"name={step.state.name}"
            record["fit_policy"] = (
                "fixed_fit_panel_target_aligned_rows"
                if step.state.requires_target
                else "fixed_fit_panel"
            )
            record["target"] = step.state.target
            record["n_fit_rows"] = step.state.n_fit_rows
            record["callable"] = _callable_name(step.state.func)
    else:
        raw_metadata = out.attrs.get("macroforecast_feature_metadata")
        if isinstance(raw_metadata, pd.DataFrame) and not raw_metadata.empty:
            records = raw_metadata.to_dict("records")
        else:
            records = _records_for_columns(
                out,
                operation=step.method,
                sources=source_columns,
                included=step.include,
            )
    for record in records:
        record["step"] = step.name
        record["included"] = step.include
        if not record.get("source"):
            record["source"] = step.input_name
        if not record.get("parameter"):
            record["parameter"] = step.name
    return records


def _marx_lag_matrix(
    panel: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    lag_values: tuple[int, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{column}_lag{lag_value}": panel[column].shift(lag_value)
            for lag_value in lag_values
            for column in columns
        },
        index=panel.index,
    )


def _marx_from_lag_matrix(
    lag_matrix: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    lag_values: tuple[int, ...],
) -> pd.DataFrame:
    if lag_values == tuple(range(1, max(lag_values, default=0) + 1)):
        data: dict[str, np.ndarray | pd.Series] = {}
        denominators: np.ndarray = np.arange(1, len(lag_values) + 1, dtype=float)
        for column in columns:
            lag_columns = [f"{column}_lag{lag_value}" for lag_value in lag_values]
            values = lag_matrix.loc[:, lag_columns].to_numpy(dtype=float, copy=False)
            # MARX matches the author R loop: for lag order l, replace the
            # l-th lag block by the row mean of lag 1..l for the same
            # variable. np.cumsum preserves skipna=False semantics because any
            # NaN in the cumulative block propagates to that and longer
            # averages.
            averages = np.cumsum(values, axis=1) / denominators
            for position, lag_order in enumerate(lag_values):
                data[f"{column}_ma{lag_order}_lag1"] = averages[:, position]
        result = pd.DataFrame(data, index=lag_matrix.index)
        result.index.name = "date"
        return result

    data = {}
    for column in columns:
        for lag_order in lag_values:
            lag_columns = [f"{column}_lag{step}" for step in range(1, lag_order + 1)]
            data[f"{column}_ma{lag_order}_lag1"] = lag_matrix.loc[
                :, lag_columns
            ].mean(axis=1, skipna=False)
    result = pd.DataFrame(data, index=lag_matrix.index)
    result.index.name = "date"
    return result


def _hamilton_design_arrays(
    series: pd.Series,
    *,
    h: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = pd.Series(series, index=series.index, dtype=float)
    if len(values) <= h + p:
        return (
            np.array([], dtype=int),
            np.empty((0, p + 1), dtype=float),
            np.array([], dtype=float),
        )
    rows: list[tuple[int, np.ndarray, float]] = []
    arr = values.to_numpy(dtype=float)
    for anchor_pos in range(p - 1, len(arr) - h):
        target_pos = anchor_pos + h
        regressors = np.array([arr[anchor_pos - lag] for lag in range(p)], dtype=float)
        target = float(arr[target_pos])
        if not np.isfinite(target) or not np.isfinite(regressors).all():
            continue
        rows.append((target_pos, np.r_[1.0, regressors], target))
    if not rows:
        return (
            np.array([], dtype=int),
            np.empty((0, p + 1), dtype=float),
            np.array([], dtype=float),
        )
    target_positions = np.array([row[0] for row in rows], dtype=int)
    x_matrix = np.vstack([row[1] for row in rows]).astype(float)
    y_vector = np.array([row[2] for row in rows], dtype=float)
    return target_positions, x_matrix, y_vector


def _hamilton_apply_beta(
    series: pd.Series,
    *,
    beta: np.ndarray,
    h: int,
    p: int,
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(series, index=series.index, dtype=float)
    cycle = pd.Series(np.nan, index=values.index, name="cycle", dtype=float)
    trend = pd.Series(np.nan, index=values.index, name="trend", dtype=float)
    target_positions, x_matrix, y_vector = _hamilton_design_arrays(values, h=h, p=p)
    if len(y_vector) == 0:
        return cycle, trend
    fitted = np.asarray(x_matrix @ beta, dtype=float).reshape(-1)
    trend.iloc[target_positions] = fitted
    cycle.iloc[target_positions] = y_vector - fitted
    return cycle, trend


def _ols_beta(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return np.asarray(beta, dtype=float).reshape(-1)


def _normalize_predictor_input(
    predictors: Literal["all"] | Iterable[str] | None,
) -> Literal["all"] | tuple[str, ...] | None:
    if predictors is None or predictors == "all":
        return predictors
    if isinstance(predictors, str):
        return (predictors,)
    return tuple(str(value) for value in predictors)


def _normalize_feature_steps(
    steps: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    if steps is None:
        return ()
    normalized: list[dict[str, Any]] = []
    seen = {"panel", "target_panel"}
    for position, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise TypeError("each feature step must be a mapping")
        raw = dict(step)
        method_value = str(raw.pop("method")) if "method" in raw else str(raw.pop("op", ""))
        method = _normalize_feature_method(method_value)
        if method not in _FEATURE_SPEC_METHODS:
            raise ValueError(
                "feature_spec() supports feature methods "
                f"{sorted(_FEATURE_SPEC_METHODS)}; got {method_value!r}"
            )
        if method == "transform" and "transform" not in raw and method_value.lower() not in {
            "transform",
            "transform_features",
            "feature_transform",
        }:
            raw["transform"] = method_value
        name = str(raw.pop("name", f"{method}_{position}"))
        if not name:
            raise ValueError("feature step name must be non-empty")
        if name in seen:
            raise ValueError(f"duplicate feature step name: {name!r}")
        seen.add(name)
        input_name = str(raw.pop("input", "panel"))
        include = bool(raw.pop("include", True))
        if method in {
            "scale",
            "pca",
            "sparse_pca_chen_rohe",
            "varimax",
            "group_pca",
            "maf",
            "marx",
            "hamilton_filter",
            "random_projection",
            "nystroem",
            "partial_least_squares",
            "sliced_inverse_regression",
            "predictor_screen",
            "variance_selection",
            "correlation_selection",
            "lasso_selection",
            "lasso_path_selection",
            "rfe_selection",
            "boruta_selection",
            "stability_selection",
            "genetic_selection",
        }:
            raw.pop("fit_policy", None)
            raw.pop("warn_full_sample", None)
        payload = {
            "name": name,
            "method": method,
            "input": input_name,
            "include": include,
            **raw,
        }
        normalized.append(payload if method == "custom" else _json_ready_step(payload))
    return tuple(normalized)


def _feature_step_plan(step: Mapping[str, Any]) -> _FeatureStepPlan:
    raw = dict(step)
    method_value = str(raw.pop("method")) if "method" in raw else str(raw.pop("op", ""))
    method = _normalize_feature_method(method_value)
    if method not in _FEATURE_SPEC_METHODS:
        raise ValueError(
            "feature_spec() supports feature methods "
            f"{sorted(_FEATURE_SPEC_METHODS)}; got {method_value!r}"
        )
    if method == "transform" and "transform" not in raw and method_value.lower() not in {
        "transform",
        "transform_features",
        "feature_transform",
    }:
        raw["transform"] = method_value
    name = str(raw.pop("name", method))
    input_name = str(raw.pop("input", "panel"))
    include = bool(raw.pop("include", True))
    return _FeatureStepPlan(
        name=name,
        method=method,
        input_name=input_name,
        include=include,
        params=dict(raw),
    )


def _normalize_optional_columns(columns: Any) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(str(value) for value in columns)


def _normalize_optional_positive_ints(
    values: Iterable[int] | int | None,
    *,
    name: str,
) -> tuple[int, ...]:
    if values is None:
        return ()
    return _normalize_positive_ints(values, name=name)


def _json_ready_step(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready_step(item) for key, item in value.items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready_step(value.to_dict())
    if isinstance(value, tuple):
        return [_json_ready_step(item) for item in value]
    if isinstance(value, list):
        return [_json_ready_step(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return _callable_name(value)
    return value


def _coerce_custom_step_output(output: Any, *, index: pd.Index, prefix: str) -> pd.DataFrame:
    if isinstance(output, pd.Series):
        frame = output.to_frame()
    elif isinstance(output, pd.DataFrame):
        frame = output.copy()
    else:
        values = np.asarray(output)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise TypeError("custom feature step output must be a Series, DataFrame, or 1D/2D array-like")
        frame = pd.DataFrame(
            values,
            columns=[f"{prefix}{idx}" for idx in range(1, values.shape[1] + 1)],
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        if len(frame.index) == len(index):
            frame.index = index
        else:
            raise ValueError("custom feature step output must keep the input DatetimeIndex or have matching length")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validate_panel(frame)
    return frame.sort_index()


def _callable_name(func: Callable[..., Any] | None) -> str | None:
    if func is None:
        return None
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


__all__ = ["FeatureSpec", "FittedFeatureBuilder", "feature_spec"]
