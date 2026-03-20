"""ForecastExperiment: outer pseudo-OOS loop orchestrator.

ForecastExperiment takes a MacroFrame (Layer 1 output) and a model registry,
then runs the full expanding- or rolling-window pseudo-OOS evaluation for every
model × horizon combination.  Results are accumulated in a ResultSet and
optionally written to parquet.

Pseudo-OOS discipline
---------------------
For each evaluation date t* in the OOS range:
  1. Construct training window [t_start .. t*-h] (expanding) or
     [t*-h-window_size .. t*-h] (rolling).
  2. Fit FeatureBuilder on the training window (no look-ahead).
  3. Transform the single test row at t*-h to get Z_{t*-h}.
  4. Fit the model on (Z_train, y_train).
  5. Predict ŷ_{t*} = model.predict(Z_test).
  6. Record (ŷ_{t*}, y_{t*}) in ForecastRecord.

This is the "direct" h-step-ahead approach: a separate model is trained for
each horizon h.  Iterated forecasting is out of scope for v1.

Parallelism
-----------
joblib is used for the outer loop over (model, horizon, date) triples.
The inner CV loop inside each model is already parallelised by sklearn's
n_jobs=-1.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from macrocast.pipeline.components import (
    CVSchemeType,
    LossFunction,
    Regularization,
    Window,
)
from macrocast.pipeline.estimator import MacrocastEstimator, SequenceEstimator
from macrocast.pipeline.features import FeatureBuilder
from macrocast.pipeline.results import ForecastRecord, ResultSet

# Imported lazily inside _run_single to avoid circular imports; only used for
# isinstance check when wiring AR-specific data.
_AR_MODEL_CLS: type | None = None


def _get_ar_model_cls() -> type | None:
    global _AR_MODEL_CLS
    if _AR_MODEL_CLS is None:
        try:
            from macrocast.pipeline.r_models import ARModel  # noqa: PLC0415
            _AR_MODEL_CLS = ARModel
        except ImportError:
            pass
    return _AR_MODEL_CLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelSpec: bind a model class to its component labels
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Configuration bundle for a single model in the experiment grid.

    Parameters
    ----------
    model_cls : type
        Class of the estimator (subclass of MacrocastEstimator or
        SequenceEstimator).
    regularization : Regularization
        Regularization component label for the four-part decomposition.
    cv_scheme : CVSchemeType
        HP-selection scheme used by this model.
    loss_function : LossFunction
        Loss function label.
    model_kwargs : dict
        Keyword arguments forwarded to ``model_cls(**model_kwargs)``.
    model_id : str or None
        Human-readable identifier.  Auto-generated from component values if
        None.
    """

    model_cls: type[MacrocastEstimator | SequenceEstimator]
    regularization: Regularization
    cv_scheme: CVSchemeType
    loss_function: LossFunction
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None

    def __post_init__(self) -> None:
        if self.model_id is None:
            # Auto-generate from component enum values
            nonlin = self.model_cls.nonlinearity_type.value  # type: ignore[attr-defined]
            reg = self.regularization.value
            cv = repr(self.cv_scheme)
            loss = self.loss_function.value
            self.model_id = f"{nonlin}__{reg}__{cv}__{loss}"

    def build(self) -> MacrocastEstimator | SequenceEstimator:
        """Instantiate a fresh estimator from this spec."""
        return self.model_cls(**self.model_kwargs)


# ---------------------------------------------------------------------------
# FeatureSpec: FeatureBuilder configuration
# ---------------------------------------------------------------------------


def _auto_label(spec: "FeatureSpec") -> str:
    """Generate a human-readable FeatureSpec label following CLSS 2021 naming."""
    parts: list[str] = []
    if spec.use_factors:
        parts.append("F")
    if spec.include_raw_x:
        parts.append("X")
    if spec.use_marx:
        if spec.use_maf or (spec.use_factors and spec.marx_for_pca):
            parts.append("MAF")
        else:
            parts.append("MARX")
    if spec.include_levels:
        parts.append("Level")
    return "-".join(parts) if parts else "AR"


@dataclass
class FeatureSpec:
    """Configuration for FeatureBuilder used in a given experiment cell.

    Parameters
    ----------
    use_factors : bool
        Whether to include PCA factors.
    n_factors : int
        Number of PCA factors.
    n_lags : int
        Number of AR lags.
    standardize_X : bool
        Standardize predictor panel before PCA.
    standardize_Z : bool
        Standardize the output feature matrix.
    lookback : int
        Sequence look-back window length for LSTM.  Ignored for cross-sectional
        models.
    use_marx : bool
        Whether to include MARX (mixed-frequency AR with cross-section) features.
        Requires FeatureBuilder support.
    p_marx : int
        Number of lags for MARX feature construction.
    use_maf : bool
        Whether to use MAF (factor-augmented) features.  Requires
        ``use_factors=True``.
    include_levels : bool
        Whether to append the untransformed level panel alongside the
        stationary-transformed panel.  Requires ``panel_levels`` to be
        provided to ForecastExperiment.
    target_scheme : str
        Multi-step targeting strategy.  ``"direct"`` trains on y_{t+h};
        ``"path_average"`` trains on the mean of y_{t+1}, ..., y_{t+h}.
    """

    use_factors: bool = True
    n_factors: int = 8
    n_lags: int = 4
    standardize_X: bool = True
    standardize_Z: bool = False
    lookback: int = 12  # for LSTM; months
    # CLSS 2021 additions
    use_marx: bool = False
    p_marx: int = 12
    use_maf: bool = False
    include_levels: bool = False
    target_scheme: str = "direct"  # "direct" | "path_average"
    include_raw_x: bool = False
    marx_for_pca: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = _auto_label(self)


# ---------------------------------------------------------------------------
# ForecastExperiment
# ---------------------------------------------------------------------------


class ForecastExperiment:
    """Outer pseudo-OOS loop that evaluates a model grid over a date range.

    Parameters
    ----------
    panel : pd.DataFrame
        Stationary-transformed predictor panel of shape (T, N), indexed by
        pd.DatetimeIndex (monthly or quarterly).
    target : pd.Series
        Target series y_t, same DatetimeIndex as ``panel``.
    horizons : list of int
        Forecast horizons h (periods ahead).  A separate model is trained per h.
    model_specs : list of ModelSpec
        Model grid to evaluate.
    feature_spec : FeatureSpec or None
        Feature construction configuration.  Uses defaults if None.
    window : Window
        Outer evaluation window strategy (EXPANDING or ROLLING).
    rolling_size : int or None
        Training window size for rolling window.  Required when
        ``window == Window.ROLLING``.
    oos_start : pd.Timestamp or str or None
        Start of the out-of-sample evaluation period.  Defaults to the 80th
        percentile of the sample.
    oos_end : pd.Timestamp or str or None
        End of the OOS period.  Defaults to the last date in the panel minus
        the maximum horizon.
    n_jobs : int
        Number of parallel workers for the outer loop.  -1 uses all cores.
    experiment_id : str or None
        UUID for this run.  Auto-generated if None.
    output_dir : Path or str or None
        If provided, ResultSet is written to parquet under this directory after
        the run completes.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        target: pd.Series,
        horizons: list[int],
        model_specs: list[ModelSpec],
        feature_spec: FeatureSpec | None = None,
        panel_levels: pd.DataFrame | None = None,
        window: Window = Window.EXPANDING,
        rolling_size: int | None = None,
        oos_start: pd.Timestamp | str | None = None,
        oos_end: pd.Timestamp | str | None = None,
        n_jobs: int = 1,
        experiment_id: str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.panel = panel
        self.target = target
        self.horizons = horizons
        self.model_specs = model_specs
        self.feature_spec = feature_spec or FeatureSpec()
        self.panel_levels = panel_levels
        self.window = window
        self.rolling_size = rolling_size
        self.n_jobs = n_jobs
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.output_dir = Path(output_dir) if output_dir else None

        # Validate and resolve OOS range
        dates = panel.index
        if oos_start is None:
            cutoff_idx = int(0.8 * len(dates))
            self._oos_start = dates[cutoff_idx]
        else:
            self._oos_start = pd.Timestamp(oos_start)

        max_h = max(horizons)
        if oos_end is None:
            self._oos_end = dates[-max_h - 1]
        else:
            self._oos_end = pd.Timestamp(oos_end)

        if self.window == Window.ROLLING and rolling_size is None:
            raise ValueError(
                "rolling_size must be provided when window=Window.ROLLING."
            )

        feat_spec = self.feature_spec
        if feat_spec.use_maf and not feat_spec.use_factors:
            raise ValueError("use_maf=True requires use_factors=True in FeatureSpec.")
        if feat_spec.include_levels and panel_levels is None:
            raise ValueError("include_levels=True requires panel_levels to be provided.")
        if feat_spec.target_scheme not in {"direct", "path_average"}:
            raise ValueError(
                f"target_scheme must be 'direct' or 'path_average', got {feat_spec.target_scheme!r}."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> ResultSet:
        """Execute the full experiment and return a ResultSet.

        Returns
        -------
        ResultSet
            All forecast records accumulated during the run.
        """
        result_set = ResultSet(
            experiment_id=self.experiment_id,
            metadata={
                "horizons": self.horizons,
                "n_models": len(self.model_specs),
                "window": self.window.value,
                "oos_start": str(self._oos_start),
                "oos_end": str(self._oos_end),
            },
        )

        # Build list of (model_spec, horizon, forecast_date) tasks
        oos_dates = self.panel.index[
            (self.panel.index >= self._oos_start) & (self.panel.index <= self._oos_end)
        ]

        tasks: list[tuple[ModelSpec, int, pd.Timestamp]] = [
            (spec, h, t_star)
            for spec in self.model_specs
            for h in self.horizons
            for t_star in oos_dates
        ]

        logger.info(
            "ForecastExperiment %s: %d tasks (%d models × %d horizons × %d dates)",
            self.experiment_id,
            len(tasks),
            len(self.model_specs),
            len(self.horizons),
            len(oos_dates),
        )

        # Execute tasks — parallelise over (model, horizon, date) triples
        records: list[ForecastRecord | None] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_single)(spec, h, t_star) for spec, h, t_star in tasks
        )

        # Filter out None (failed cells)
        for r in records:
            if r is not None:
                result_set.add(r)

        logger.info(
            "ForecastExperiment %s complete: %d records",
            self.experiment_id,
            len(result_set),
        )

        if self.output_dir is not None:
            out_path = self.output_dir / f"{self.experiment_id}.parquet"
            result_set.to_parquet(out_path)
            logger.info("Results written to %s", out_path)

        return result_set

    # ------------------------------------------------------------------
    # Single evaluation cell
    # ------------------------------------------------------------------

    def _run_single(
        self,
        spec: ModelSpec,
        h: int,
        t_star: pd.Timestamp,
    ) -> ForecastRecord | None:
        """Evaluate one model × horizon × date cell.

        Returns None on failure (logged as warning).
        """
        try:
            dates = self.panel.index

            # Training window end: h periods before forecast date
            t_star_pos = dates.get_loc(t_star)
            train_end_pos = t_star_pos - h
            if train_end_pos < 1:
                return None  # insufficient history
            train_end = dates[train_end_pos]

            # Training window start
            if self.window == Window.EXPANDING:
                train_start = dates[0]
            else:
                start_pos = max(0, train_end_pos - self.rolling_size + 1)  # type: ignore
                train_start = dates[start_pos]

            # Slice panel and target
            X_train = self.panel.loc[train_start:train_end].values
            y_train_full = self.target.loc[train_start:train_end].values

            # y_train for h-step direct: y_{t+h} aligned with X_t
            # Drop last h observations from X (no y_{t+h} available for them)
            # and shift y forward by h
            T_tr = X_train.shape[0]
            if T_tr <= h:
                return None
            X_tr_aligned = X_train[: T_tr - h]  # rows 0 .. T-h-1

            if self.feature_spec.target_scheme == "path_average" and h > 1:
                # Path average: ȳ_{t+h} = (1/h) Σ_{h'=1}^{h} y_{t+h'}
                # For position i: mean(y[i+1], ..., y[i+h])
                # Using cumsum: cs[k] = sum(y[0..k-1])
                # path_avg[i] = (cs[i+h+1] - cs[i+1]) / h
                cs = np.concatenate([[0.0], np.cumsum(y_train_full)])
                y_tr_aligned = (cs[1 + h : T_tr + 1] - cs[1 : T_tr - h + 1]) / h
            else:
                y_tr_aligned = y_train_full[h:]  # rows h .. T-1 → y_{t+h}

            # Test row: single observation at train_end (forecast Z_{train_end})
            X_test_row = self.panel.loc[train_end:train_end].values  # (1, N)

            if self.feature_spec.target_scheme == "path_average" and h > 1:
                # y_true is the path average over the next h periods
                # t_star_pos is the position of the forecast date (h steps ahead)
                # We want mean(target[t_star_pos-h+1], ..., target[t_star_pos])
                y_true = float(
                    np.mean(self.target.iloc[t_star_pos - h + 1 : t_star_pos + 1])
                )
            else:
                y_true = float(self.target.loc[t_star])

            # Feature construction
            feat_spec = self.feature_spec
            is_sequence = issubclass(
                spec.model_cls,
                SequenceEstimator,  # type: ignore[arg-type]
            )

            if is_sequence:
                # LSTM: build (T, L, N) tensors
                Z_train, Z_test = self._build_sequence_features(
                    X_tr_aligned, y_tr_aligned, X_test_row, feat_spec
                )
            else:
                builder = FeatureBuilder(
                    n_factors=feat_spec.n_factors,
                    n_lags=feat_spec.n_lags,
                    use_factors=feat_spec.use_factors,
                    standardize_X=feat_spec.standardize_X,
                    standardize_Z=feat_spec.standardize_Z,
                    use_marx=feat_spec.use_marx,
                    p_marx=feat_spec.p_marx,
                    use_maf=feat_spec.use_maf,
                    include_levels=feat_spec.include_levels,
                    include_raw_x=feat_spec.include_raw_x,
                    marx_for_pca=feat_spec.marx_for_pca,
                )

                # Prepare levels slices if needed
                X_levels_tr: NDArray[np.floating] | None = None
                X_levels_test: NDArray[np.floating] | None = None
                if feat_spec.include_levels and self.panel_levels is not None:
                    X_levels_tr = self.panel_levels.loc[train_start:train_end].values[: T_tr - h]
                    X_levels_test = self.panel_levels.loc[train_end:train_end].values

                # For the direct-h target, y_tr_aligned is the shifted target;
                # we pass the *un-shifted* y for AR lag construction
                Z_train = builder.fit_transform(
                    X_tr_aligned, y_train_full[: T_tr - h], X_levels=X_levels_tr
                )
                Z_test = builder.transform(
                    X_test_row, y_train_full[-feat_spec.n_lags :], X_levels=X_levels_test
                )

                if Z_train.shape[0] == 0 or Z_test.shape[0] == 0:
                    return None

                # Align y with Z_train row count.  When MARX is active with
                # p_marx > n_lags + 1 the FeatureBuilder drops max(p_marx-1, n_lags)
                # rows instead of just n_lags; take the last Z_train.shape[0] rows.
                y_tr_for_fit = y_tr_aligned[len(y_tr_aligned) - Z_train.shape[0] :]

            # Build and fit model
            model = spec.build()

            # Inject AR-specific data before fit/predict if needed.
            # ARModel ignores the FeatureBuilder Z matrix and instead operates
            # on the raw target series; we provide the un-shifted y and lags here.
            ar_cls = _get_ar_model_cls()
            if ar_cls is not None and isinstance(model, ar_cls):
                model._y_train_full = y_train_full  # un-shifted target
                max_lag = model.model_kwargs.get("max_lag", 12)
                model._y_test_lags = y_train_full[-max(feat_spec.n_lags, max_lag):]

            if is_sequence:
                model.fit(Z_train, y_tr_aligned)
                y_hat = float(model.predict(Z_test)[0])
                hp_selected: dict[str, Any] = getattr(model, "best_params_", {})
                n_factors = None
            else:
                model.fit(Z_train, y_tr_for_fit)
                y_hat = float(model.predict(Z_test)[0])
                hp_selected = getattr(model, "best_params_", {})
                n_factors = feat_spec.n_factors if feat_spec.use_factors else None

            return ForecastRecord(
                experiment_id=self.experiment_id,
                model_id=spec.model_id,
                nonlinearity=spec.model_cls.nonlinearity_type,  # type: ignore[attr-defined]
                regularization=spec.regularization,
                cv_scheme=spec.cv_scheme,
                loss_function=spec.loss_function,
                window=self.window,
                horizon=h,
                train_end=train_end,
                forecast_date=t_star,
                y_hat=y_hat,
                y_true=y_true,
                n_train=len(Z_train),
                n_factors=n_factors,
                n_lags=feat_spec.n_lags,
                hp_selected=hp_selected,
                target_scheme=feat_spec.target_scheme,
                feature_set=self.feature_spec.label,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cell failed (model=%s, h=%d, date=%s): %s",
                spec.model_id,
                h,
                t_star,
                exc,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Sequence feature builder (LSTM)
    # ------------------------------------------------------------------

    def _build_sequence_features(
        self,
        X_tr: NDArray[np.floating],
        y_tr: NDArray[np.floating],
        X_test_row: NDArray[np.floating],
        feat_spec: FeatureSpec,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Build (T, L, N_feat) tensors for LSTM training and test.

        Uses a simple sliding-window over the combined [X panel, AR lags]
        feature matrix with look-back length ``feat_spec.lookback``.
        """
        L = feat_spec.lookback

        # Concatenate panel columns (we pass raw X; no PCA for LSTM in v1)
        # AR lag columns appended manually
        p = feat_spec.n_lags
        T = X_tr.shape[0]

        ar_lags = np.column_stack(
            [
                y_tr[p - lag - 1 : T - lag - 1] if lag < p else np.zeros(T - p)
                for lag in range(p)
            ]
        )  # shape (T-p, p)
        X_with_lags = np.concatenate([X_tr[p:], ar_lags], axis=1)  # (T-p, N+p)

        # Sliding window sequences: need at least L rows
        T2 = X_with_lags.shape[0]
        if T2 < L + 1:
            raise ValueError(
                f"Training window ({T2} rows after lag trimming) is shorter "
                f"than LSTM look-back ({L}). Increase oos_start or reduce lookback."
            )

        # Training sequences: rows L-1 .. T2-1 (each window is rows i-L+1..i)
        n_seq = T2 - L + 1
        X_seq = np.stack(
            [X_with_lags[i : i + L] for i in range(n_seq)]
        )  # (n_seq, L, N+p)

        # Test sequence: last L rows of training window as one sequence
        # Append the test row for the final step
        X_test_feat = np.concatenate(
            [X_test_row, np.zeros((1, p))],
            axis=1,  # AR lags for test row are 0-padded
        )  # (1, N+p)
        X_test_window = np.concatenate(
            [X_with_lags[-L + 1 :], X_test_feat], axis=0
        )  # (L, N+p)
        X_test_seq = X_test_window[np.newaxis, ...]  # (1, L, N+p)

        return X_seq, X_test_seq
