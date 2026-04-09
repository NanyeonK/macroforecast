"""Result containers for the macrocast forecast experiment."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from macrocast.pipeline.components import (
    CVSchemeType,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)


@dataclass
class FailureRecord:
    experiment_id: str
    model_id: str
    horizon: int
    failure_stage: str
    exception_class: str
    exception_message: str
    severity: str
    retry_count: int = 0
    cell_id: str | None = None
    warning_only: bool = False

    def to_dict(self) -> dict:
        return {
            'experiment_id': self.experiment_id,
            'model_id': self.model_id,
            'horizon': self.horizon,
            'failure_stage': self.failure_stage,
            'exception_class': self.exception_class,
            'exception_message': self.exception_message,
            'severity': self.severity,
            'retry_count': self.retry_count,
            'cell_id': self.cell_id,
            'warning_only': self.warning_only,
        }


@dataclass
class ForecastRecord:
    experiment_id: str
    model_id: str
    nonlinearity: Nonlinearity
    regularization: Regularization
    cv_scheme: CVSchemeType
    loss_function: LossFunction
    window: Window
    horizon: int
    train_end: pd.Timestamp
    forecast_date: pd.Timestamp
    y_hat: float
    y_true: float
    n_train: int
    n_factors: int | None
    n_lags: int
    hp_selected: dict = field(default_factory=dict)
    target_scheme: str = "direct"
    feature_set: str = ""
    feature_importances: dict[str, float] | None = None
    benchmark_id: str | None = None
    evaluation_scale: str | None = None
    cell_id: str | None = None
    degraded_run: bool = False

    @property
    def error(self) -> float:
        return self.y_true - self.y_hat

    @property
    def squared_error(self) -> float:
        return self.error**2

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "nonlinearity": self.nonlinearity.value,
            "regularization": self.regularization.value,
            "cv_scheme": repr(self.cv_scheme),
            "loss_function": self.loss_function.value,
            "window": self.window.value,
            "horizon": self.horizon,
            "train_end": self.train_end,
            "forecast_date": self.forecast_date,
            "y_hat": self.y_hat,
            "y_true": self.y_true,
            "n_train": self.n_train,
            "n_factors": self.n_factors,
            "n_lags": self.n_lags,
            **{f"hp_{k}": v for k, v in self.hp_selected.items()},
            "target_scheme": self.target_scheme,
            "feature_set": self.feature_set,
            "feature_importances": json.dumps(self.feature_importances) if self.feature_importances else None,
            'benchmark_id': self.benchmark_id,
            'evaluation_scale': self.evaluation_scale,
            'cell_id': self.cell_id,
            'degraded_run': self.degraded_run,
        }


@dataclass
class ResultSet:
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    records: list[ForecastRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)
    degraded: bool = False

    def add(self, record: ForecastRecord) -> None:
        self.records.append(record)
        if record.degraded_run:
            self.degraded = True

    def extend(self, records: list[ForecastRecord]) -> None:
        for record in records:
            self.add(record)

    def add_failure(self, failure: FailureRecord) -> None:
        self.failures.append(failure)
        if failure.severity in {'degraded_run', 'hard_error'}:
            self.degraded = True

    def failure_summary(self) -> list[str]:
        return [f"{f.model_id}|h={f.horizon}|{f.failure_stage}|{f.exception_class}" for f in self.failures]

    def failures_dataframe(self) -> pd.DataFrame:
        if not self.failures:
            return pd.DataFrame(columns=['experiment_id','model_id','horizon','failure_stage','exception_class','exception_message','severity','retry_count','cell_id','warning_only'])
        return pd.DataFrame([f.to_dict() for f in self.failures])

    def to_dataframe(self) -> pd.DataFrame:
        if self.records:
            rows = [r.to_dict() for r in self.records]
            df = pd.DataFrame(rows)
            df["train_end"] = pd.to_datetime(df["train_end"])
            df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        else:
            df = pd.DataFrame()
        combo: pd.DataFrame | None = getattr(self, "_combo_df", None)
        if combo is not None and not combo.empty:
            df = pd.concat([df, combo], ignore_index=True)
        return df

    def to_parquet(self, path: str | Path, **kwargs) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(path, index=False, **kwargs)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> 'ResultSet':
        path = Path(path)
        df = pd.read_parquet(path)
        rs = cls(metadata={"source_parquet": str(path)})
        rs._df_cache = df  # type: ignore[attr-defined]
        return rs

    def to_dataframe_cached(self) -> pd.DataFrame:
        if hasattr(self, "_df_cache"):
            return self._df_cache  # type: ignore[attr-defined]
        return self.to_dataframe()

    def msfe_by_model(self, horizon: int | None = None) -> pd.DataFrame:
        df = self.to_dataframe_cached()
        if df.empty:
            return pd.DataFrame(columns=["model_id", "horizon", "msfe", "n_obs"])
        if horizon is not None:
            df = df[df["horizon"] == horizon]
        df = df.copy()
        df["se"] = (df["y_true"] - df["y_hat"]) ** 2
        summary = (
            df.groupby(["model_id", "horizon"])["se"]
            .agg(msfe="mean", n_obs="count")
            .reset_index()
        )
        return summary

    def with_combination(self, methods: str | list[str] = "mean", trim_pct: float = 0.10, window: int | None = None) -> 'ResultSet':
        from macrocast.evaluation.combination import combine_forecasts
        if isinstance(methods, str):
            methods = [methods]
        base = self.to_dataframe_cached()
        if base.empty:
            raise ValueError('empty result set cannot compute combinations')
        combo_frames = [combine_forecasts(base, method=m, trim_pct=trim_pct, window=window) for m in methods]
        existing_combo = getattr(self, '_combo_df', pd.DataFrame())
        out = ResultSet(experiment_id=self.experiment_id, records=list(self.records), metadata=dict(self.metadata), failures=list(self.failures), degraded=self.degraded)
        frames = [df for df in [existing_combo, *combo_frames] if df is not None and not df.empty]
        out._combo_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # type: ignore[attr-defined]
        return out

    def combination_methods(self) -> list[str]:
        combo = getattr(self, '_combo_df', pd.DataFrame())
        if combo is None or combo.empty or 'model_id' not in combo:
            return []
        return sorted(combo['model_id'].dropna().unique().tolist())

    def __len__(self) -> int:
        return len(self.records)
