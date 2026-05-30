from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, attach_metadata, panel_info, validate_panel
from macroforecast.feature_engineering.shared import (
    TargetMode,
    TargetTransform,
    _coerce_input,
    _metadata_frame,
    _normalize_lags,
    _normalize_positive_ints,
    _normalize_target_mode,
    _normalize_target_transform,
    _records_for_columns,
    _resolve_horizons,
    _resolve_predictors,
    _resolve_targets,
    _target_metadata_frame,
    _target_transform_to_path_transform,
)
from macroforecast.feature_engineering.targets import direct_target, path_targets
from macroforecast.feature_engineering.transforms import lag, rolling_mean, time_features
from macroforecast.feature_engineering.types import FeatureInput, FeatureSet


@dataclass(frozen=True)
class FeatureSpec:
    """Reusable feature-building callable for forecasting runners."""

    target: str | None = None
    targets: tuple[str, ...] = ()
    horizon: int | None = None
    horizons: tuple[int, ...] = ()
    predictors: Literal["all"] | tuple[str, ...] | None = None
    lags: tuple[int, ...] = (0, 1)
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
        )
        pca_state = _fit_pca_state(
            panel,
            predictors=predictor_values,
            columns=self.pca_columns,
            n_components=self.pca_components,
            scale=self.pca_scale,
            prefix=self.pca_prefix,
        )
        return FittedFeatureBuilder(
            spec=self,
            fit_panel=panel.copy(),
            fit_metadata=dict(base.metadata),
            targets=target_values,
            horizons=horizon_values,
            predictors=predictor_values,
            pca_state=pca_state,
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
            "lags": list(self.lags),
            "rolling_windows": list(self.rolling_windows),
            "rolling_min_periods": self.rolling_min_periods,
            "add_time": self.add_time,
            "time": {
                "trend": self.time_trend,
                "month": self.time_month,
                "quarter": self.time_quarter,
                "year": self.time_year,
            },
            "pca_components": self.pca_components,
            "pca_columns": list(self.pca_columns or ()),
            "pca_scale": self.pca_scale,
            "pca_prefix": self.pca_prefix,
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


def feature_spec(
    *,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    predictors: Literal["all"] | Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1),
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
    target_transform: TargetTransform = "level",
    target_mode: TargetMode = "direct",
    drop_missing: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureSpec:
    """Create a reusable feature-building specification."""

    return FeatureSpec(
        target=target,
        targets=tuple(str(value) for value in (targets or ())),
        horizon=horizon,
        horizons=_normalize_optional_positive_ints(horizons, name="horizons"),
        predictors=_normalize_predictor_input(predictors),
        lags=_normalize_lags(lags, allow_zero=True),
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
        target_transform=_normalize_target_transform(target_transform),
        target_mode=_normalize_target_mode(target_mode),
        drop_missing=bool(drop_missing),
        metadata=dict(metadata or {}),
    )


def _build_predictors(
    panel: pd.DataFrame,
    fitted: FittedFeatureBuilder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    spec = fitted.spec

    if spec.lags:
        lagged = lag(panel, columns=fitted.predictors, lags=spec.lags)
        parts.append(lagged)
        records.extend(_records_for_columns(lagged, operation="lag", sources=fitted.predictors, included=True))
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
    train = panel.loc[:, selected].dropna().astype(float)
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

    model = PCA(n_components=n_value)
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


def _normalize_predictor_input(
    predictors: Literal["all"] | Iterable[str] | None,
) -> Literal["all"] | tuple[str, ...] | None:
    if predictors is None or predictors == "all":
        return predictors
    if isinstance(predictors, str):
        return (predictors,)
    return tuple(str(value) for value in predictors)


def _normalize_optional_positive_ints(
    values: Iterable[int] | int | None,
    *,
    name: str,
) -> tuple[int, ...]:
    if values is None:
        return ()
    return _normalize_positive_ints(values, name=name)


__all__ = ["FeatureSpec", "FittedFeatureBuilder", "feature_spec"]
