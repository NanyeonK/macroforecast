"""Result containers for the macrocast forecast experiment.

Two dataclasses form the result layer:

* ``ForecastRecord`` -- a single model × horizon × vintage evaluation cell.
  One row in the final result table.
* ``ResultSet`` -- the full collection of records for one experiment run,
  with helpers for export to parquet and basic summary statistics.

These containers are intentionally thin.  All evaluation logic lives in
``macrocast.evaluation``; this module only holds and serialises the raw
forecast-vs-realisation pairs.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# ForecastRecord
# ---------------------------------------------------------------------------


@dataclass
class ForecastRecord:
    """One evaluation cell: a single model fitted on one training window,
    predicting one horizon h, producing one out-of-sample forecast.

    Fields
    ------
    experiment_id : str
        UUID of the parent ForecastExperiment run.
    model_id : str
        Human-readable model name, e.g. ``"krr_factors_kfold_l2"``.
    nonlinearity : Nonlinearity
        Nonlinearity component of the four-part decomposition.
    regularization : Regularization
        Regularization component.
    cv_scheme : CVSchemeType
        CV scheme used for HP selection in this cell.
    loss_function : LossFunction
        Loss function optimised during training.
    window : Window
        Outer evaluation window strategy (expanding or rolling).
    horizon : int
        Forecast horizon h (months/quarters ahead).
    train_end : pd.Timestamp
        Last date in the training window.
    forecast_date : pd.Timestamp
        Date being forecast (train_end + h periods).
    y_hat : float
        Point forecast (on the original, untransformed scale if the caller
        applies inverse transforms; otherwise on the transformed scale).
    y_true : float
        Realised value at forecast_date.
    n_train : int
        Number of training observations used.
    n_factors : Optional[int]
        Number of PCA factors used (None if AR-only mode).
    n_lags : int
        Number of AR lags used.
    hp_selected : dict
        Best hyperparameter values selected by the CV scheme (e.g.
        ``{"alpha": 0.01, "gamma": 0.1}``).
    """

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

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def error(self) -> float:
        """Forecast error e_t = y_true - y_hat."""
        return self.y_true - self.y_hat

    @property
    def squared_error(self) -> float:
        """Squared forecast error (y_true - y_hat)²."""
        return self.error**2

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Flatten to a plain dict suitable for pd.DataFrame construction."""
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
        }


# ---------------------------------------------------------------------------
# ResultSet
# ---------------------------------------------------------------------------


@dataclass
class ResultSet:
    """Collection of ForecastRecords from one experiment run.

    Parameters
    ----------
    experiment_id : str
        Shared UUID across all records in this set.  Auto-generated if not
        provided.
    records : list of ForecastRecord
        Raw forecast cells.  Populated incrementally by the experiment runner.
    metadata : dict
        Arbitrary experiment-level metadata (dataset name, target variable,
        date range, config hash, etc.).
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    records: list[ForecastRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record: ForecastRecord) -> None:
        """Append a single ForecastRecord."""
        self.records.append(record)

    def extend(self, records: list[ForecastRecord]) -> None:
        """Append multiple records at once."""
        self.records.extend(records)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all records to a tidy pandas DataFrame.

        Each row is one forecast cell.  Component columns use string values
        (Enum.value) for compatibility with downstream R reads via arrow.
        """
        if not self.records:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self.records]
        df = pd.DataFrame(rows)
        df["train_end"] = pd.to_datetime(df["train_end"])
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        return df

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_parquet(self, path: str | Path, **kwargs) -> Path:
        """Write ResultSet to a parquet file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created if needed.
        **kwargs
            Forwarded to ``pd.DataFrame.to_parquet``.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(path, index=False, **kwargs)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> ResultSet:
        """Read a ResultSet previously written by ``to_parquet``.

        Note: ForecastRecord objects are NOT reconstructed — use the returned
        DataFrame via ``ResultSet.to_dataframe()`` for most downstream work.
        This classmethod returns a ResultSet with an empty ``records`` list
        but populates ``metadata`` with the parquet file path for provenance.
        """
        path = Path(path)
        df = pd.read_parquet(path)
        rs = cls(metadata={"source_parquet": str(path)})
        # Re-hydrate lightweight: keep as DataFrame attribute for fast access
        rs._df_cache = df  # type: ignore[attr-defined]
        return rs

    def to_dataframe_cached(self) -> pd.DataFrame:
        """Return cached DataFrame if loaded from parquet, else compute."""
        if hasattr(self, "_df_cache"):
            return self._df_cache  # type: ignore[attr-defined]
        return self.to_dataframe()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def msfe_by_model(self, horizon: int | None = None) -> pd.DataFrame:
        """MSFE per model_id, optionally filtered to a single horizon.

        Returns
        -------
        pd.DataFrame with columns [model_id, horizon, msfe, n_obs].
        """
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

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"ResultSet(experiment_id={self.experiment_id!r}, "
            f"n_records={len(self.records)})"
        )
