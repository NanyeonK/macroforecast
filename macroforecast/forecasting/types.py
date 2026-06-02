from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """Forecast runner output."""

    forecasts: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    sidecars: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the forecast table."""

        return self.forecasts.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready forecast result."""

        return {
            "forecasts": _json_ready(self.forecasts.to_dict(orient="records")),
            "metadata": _json_ready(self.metadata),
            "sidecars": _json_ready(self.sidecars),
        }

    def evaluate(self, **kwargs: Any) -> pd.DataFrame:
        """Evaluate this forecast result with ``macroforecast.metrics``."""

        from macroforecast.metrics import evaluate_forecasts

        return evaluate_forecasts(self, **kwargs)

    def anatomy_explain(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Explain a precomputed ``anatomy.Anatomy`` object for this run."""

        from macroforecast.interpretation import anatomy_explain

        table = anatomy_explain(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def anatomy_pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute backend PBSV rows for a precomputed ``anatomy.Anatomy`` object."""

        return self.pbsv(anatomy, **kwargs)

    def pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute PBSV rows for a precomputed forecast-Shapley backend object."""

        from macroforecast.interpretation import pbsv

        table = pbsv(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def anatomy_oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute backend oShapley-VI rows for a precomputed anatomy object."""

        return self.oshapley_vi(anatomy, **kwargs)

    def oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute oShapley-VI rows for a precomputed forecast-Shapley object."""

        from macroforecast.interpretation import oshapley_vi

        table = oshapley_vi(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def with_sidecar(self, name: str, value: Any) -> "ForecastResult":
        """Return a copy with a named runtime sidecar attached."""

        key = str(name)
        if not key:
            raise ValueError("sidecar name must not be empty")
        sidecars = dict(self.sidecars)
        sidecars[key] = value
        metadata = dict(self.metadata)
        registry = dict(metadata.get("sidecars", {}))
        registry[key] = _sidecar_metadata(value)
        metadata["sidecars"] = registry
        return ForecastResult(self.forecasts.copy(), metadata=metadata, sidecars=sidecars)

    def get_sidecar(self, name: str, default: Any = None) -> Any:
        """Return a named sidecar, or ``default`` when it is absent."""

        return self.sidecars.get(str(name), default)

    def sidecar_names(self) -> tuple[str, ...]:
        """Return attached sidecar names."""

        return tuple(self.sidecars)

    def with_anatomy(
        self,
        X: Any,
        y: Any,
        models: Any,
        *,
        window: Any,
        sidecar_name: str = "anatomy",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach a forecast-accuracy anatomy sidecar.

        ``window`` is required because a completed forecast table does not
        contain the feature matrix, target vector, and origin-wise refit design
        needed by the anatomy backend.
        """

        from macroforecast.interpretation import anatomy_from_forecast_result

        return anatomy_from_forecast_result(
            self,
            X,
            y,
            models,
            window=window,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def with_oshapley(
        self,
        X: Any,
        y: Any,
        models: Any,
        *,
        window: Any,
        sidecar_name: str = "oshapley",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach an oShapley/PBSV forecast-accuracy sidecar."""

        from macroforecast.interpretation import oshapley_from_forecast_result

        return oshapley_from_forecast_result(
            self,
            X,
            y,
            models,
            window=window,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def with_dual(
        self,
        model: Any | None,
        X_train: Any,
        y_train: Any,
        X_test: Any | None = None,
        *,
        sidecar_name: str = "dual",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach a dual interpretation sidecar."""

        from macroforecast.interpretation import dual_from_forecast_result

        return dual_from_forecast_result(
            self,
            model,
            X_train,
            y_train,
            X_test,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.Series):
        return {
            "name": value.name,
            "index": [_json_ready(item) for item in value.index],
            "data": [_json_ready(item) for item in value.to_list()],
        }
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [_json_ready(item) for item in value.index],
            "data": _json_ready(value.to_dict(orient="list")),
        }
    if hasattr(value, "to_dict") and not isinstance(value, type):
        try:
            return _json_ready(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _sidecar_metadata(value: Any) -> dict[str, Any]:
    schema = getattr(value, "metadata_schema", None)
    if callable(schema):
        schema = schema()
    metadata = getattr(value, "metadata", None)
    return {
        "object_type": f"{type(value).__module__}.{type(value).__name__}",
        "metadata_schema": _json_ready(schema),
        "metadata_keys": sorted(str(key) for key in metadata)
        if isinstance(metadata, Mapping)
        else [],
    }


__all__ = ["ForecastResult"]
