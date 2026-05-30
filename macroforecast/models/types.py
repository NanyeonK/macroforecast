from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelFit:
    """Fitted model wrapper returned by macroforecast model callables."""

    estimator: Any
    model: str
    feature_names: tuple[str, ...] = ()
    target_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: Any) -> pd.Series:
        """Return point predictions as a pandas Series."""

        if not hasattr(self.estimator, "predict"):
            raise AttributeError(f"{self.model!r} estimator does not expose predict()")
        frame = _prediction_frame(X, self.feature_names)
        values = np.asarray(self.estimator.predict(frame), dtype=float).reshape(-1)
        return pd.Series(values, index=frame.index, name="prediction")

    def summary(self) -> str:
        """Return a compact text summary of the fitted object."""

        lines = [f"Model: {self.model}", f"No. features: {len(self.feature_names)}"]
        if self.target_name is not None:
            lines.append(f"Target: {self.target_name}")
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.estimator, name):
            return getattr(self.estimator, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


@dataclass
class VolatilityFit(ModelFit):
    """Fitted volatility model wrapper."""

    def predict_variance(self, horizon: int = 1) -> pd.Series:
        if not hasattr(self.estimator, "predict_variance"):
            raise AttributeError(f"{self.model!r} estimator does not expose predict_variance()")
        values = np.asarray(self.estimator.predict_variance(horizon), dtype=float).reshape(-1)
        return pd.Series(values, index=pd.RangeIndex(len(values)), name="variance")

    @property
    def conditional_volatility(self) -> pd.Series | None:
        values = getattr(self.estimator, "conditional_volatility_", None)
        if values is None:
            return None
        return pd.Series(np.asarray(values, dtype=float), name="conditional_volatility")


def _prediction_frame(X: Any, feature_names: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        columns = list(feature_names) if feature_names and len(feature_names) == arr.shape[1] else [
            f"x{i}" for i in range(arr.shape[1])
        ]
        frame = pd.DataFrame(arr, columns=columns)
    if feature_names:
        frame = frame.reindex(columns=list(feature_names), fill_value=0.0)
    return frame


__all__ = ["ModelFit", "VolatilityFit"]
