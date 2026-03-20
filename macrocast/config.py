"""YAML experiment configuration loader.

Loads a YAML config file and returns a fully resolved ExperimentConfig object
that ForecastExperiment can consume directly.

Config file format (experiment.yaml):
---------------------------------------
experiment:
  id: "my_experiment"              # optional; auto-generated if absent
  output_dir: "~/.macrocast/results"

data:
  dataset: "fred_md"              # fred_md | fred_qd | fred_sd
  vintage: null                   # null for current release; "YYYY-MM" for vintage
  target: "INDPRO"                # column name in the dataset
  cache_dir: "~/.macrocast/cache" # optional

features:
  n_factors: 8
  n_lags: 4
  use_factors: true
  standardize_X: true
  standardize_Z: false
  lookback: 12                    # LSTM look-back window (ignored for cross-sectional)

experiment:
  horizons: [1, 3, 6, 12]
  window: "expanding"             # expanding | rolling
  rolling_size: null              # required when window = rolling
  oos_start: "2000-01-01"        # null for default (80th percentile)
  oos_end: null
  n_jobs: -1

models:
  - name: krr
    regularization: factors
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
      gamma_grid: [0.01, 0.1, 1.0]
      cv_folds: 5

  - name: rf
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      n_estimators: 500

  - name: xgboost
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2

  - name: nn
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [64, 128]
      max_epochs: 200
      patience: 20

  - name: lstm
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [32, 64]
      max_epochs: 200
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from macrocast.pipeline.components import (
    CVScheme,
    CVSchemeType,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.pipeline.models import (
    KRRModel,
    LSTMModel,
    NNModel,
    RFModel,
    SVRLinearModel,
    SVRRBFModel,
    XGBoostModel,
)

# ---------------------------------------------------------------------------
# Model registry: name → (class, default Nonlinearity, default Regularization)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, tuple] = {
    "krr": (KRRModel, Nonlinearity.KRR, Regularization.FACTORS),
    "svr_rbf": (SVRRBFModel, Nonlinearity.SVR_RBF, Regularization.FACTORS),
    "svr_linear": (SVRLinearModel, Nonlinearity.SVR_LINEAR, Regularization.NONE),
    "rf": (RFModel, Nonlinearity.RANDOM_FOREST, Regularization.NONE),
    "xgboost": (XGBoostModel, Nonlinearity.XGBOOST, Regularization.NONE),
    "nn": (NNModel, Nonlinearity.NEURAL_NET, Regularization.NONE),
    "lstm": (LSTMModel, Nonlinearity.LSTM, Regularization.NONE),
}

_REGULARIZATION_MAP: dict[str, Regularization] = {r.value: r for r in Regularization}

_LOSS_MAP: dict[str, LossFunction] = {
    "l2": LossFunction.L2,
    "epsilon_insensitive": LossFunction.EPSILON_INSENSITIVE,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    dataset: str = "fred_md"
    vintage: str | None = None
    target: str = "INDPRO"
    cache_dir: str | None = None


@dataclass
class ExperimentConfig:
    """Fully resolved experiment configuration.

    Attributes
    ----------
    experiment_id : str
    output_dir : Path
    data : DataConfig
    feature_spec : FeatureSpec
    model_specs : list of ModelSpec
    horizons : list of int
    window : Window
    rolling_size : int or None
    oos_start : str or None
    oos_end : str or None
    n_jobs : int
    """

    experiment_id: str
    output_dir: Path
    data: DataConfig
    feature_spec: FeatureSpec
    model_specs: list[ModelSpec]
    horizons: list[int]
    window: Window
    rolling_size: int | None
    oos_start: str | None
    oos_end: str | None
    n_jobs: int


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and parse a YAML experiment config file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    return _parse_config(raw)


def load_config_from_dict(raw: dict) -> ExperimentConfig:
    """Parse an already-loaded config dictionary (useful for testing)."""
    return _parse_config(raw)


def _parse_config(raw: dict) -> ExperimentConfig:
    exp_sec = raw.get("experiment", {})
    data_sec = raw.get("data", {})
    feat_sec = raw.get("features", {})
    mod_list = raw.get("models", [])

    # Experiment identity
    experiment_id = exp_sec.get("id") or str(uuid.uuid4())
    output_dir = Path(exp_sec.get("output_dir", "~/.macrocast/results")).expanduser()

    # Data
    data_cfg = DataConfig(
        dataset=data_sec.get("dataset", "fred_md"),
        vintage=data_sec.get("vintage"),
        target=data_sec.get("target", "INDPRO"),
        cache_dir=data_sec.get("cache_dir"),
    )

    # Feature spec
    feature_spec = FeatureSpec(
        n_factors=feat_sec.get("n_factors", 8),
        n_lags=feat_sec.get("n_lags", 4),
        use_factors=feat_sec.get("use_factors", True),
        standardize_X=feat_sec.get("standardize_X", True),
        standardize_Z=feat_sec.get("standardize_Z", False),
        lookback=feat_sec.get("lookback", 12),
    )

    # Outer loop config (may be nested under "experiment" or top-level)
    run_sec = exp_sec  # horizons etc. can live in [experiment] section
    horizons = [int(h) for h in run_sec.get("horizons", [1])]
    window_str = run_sec.get("window", "expanding").lower()
    window = Window.EXPANDING if window_str == "expanding" else Window.ROLLING
    rolling_size = run_sec.get("rolling_size")
    oos_start = run_sec.get("oos_start")
    oos_end = run_sec.get("oos_end")
    n_jobs = int(run_sec.get("n_jobs", 1))

    # Model specs
    model_specs = [_parse_model_spec(m) for m in mod_list]

    return ExperimentConfig(
        experiment_id=experiment_id,
        output_dir=output_dir,
        data=data_cfg,
        feature_spec=feature_spec,
        model_specs=model_specs,
        horizons=horizons,
        window=window,
        rolling_size=rolling_size,
        oos_start=str(oos_start) if oos_start else None,
        oos_end=str(oos_end) if oos_end else None,
        n_jobs=n_jobs,
    )


def _parse_model_spec(m: dict) -> ModelSpec:
    """Parse a single model entry from the YAML models list."""
    name = m.get("name", "").lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Valid names: {list(_MODEL_REGISTRY)}"
        )
    model_cls, default_nonlin, default_reg = _MODEL_REGISTRY[name]

    # Regularization: YAML value overrides default
    reg_str = m.get("regularization", default_reg.value)
    regularization = _REGULARIZATION_MAP.get(reg_str, default_reg)

    # CV scheme
    cv_str = m.get("cv_scheme", "kfold").lower()
    k = int(m.get("kfold_k", 5))
    if cv_str == "kfold":
        cv_scheme: CVSchemeType = CVScheme.KFOLD(k=k)
    elif cv_str == "bic":
        cv_scheme = CVScheme.BIC
    elif cv_str == "poos":
        cv_scheme = CVScheme.POOS
    else:
        raise ValueError(f"Unknown cv_scheme '{cv_str}'. Valid: kfold, bic, poos.")

    # Loss function
    loss_str = m.get("loss_function", "l2").lower()
    loss_function = _LOSS_MAP.get(loss_str, LossFunction.L2)

    # Model kwargs
    kwargs: dict[str, Any] = m.get("kwargs", {}) or {}

    # model_id
    model_id = m.get("model_id")

    return ModelSpec(
        model_cls=model_cls,
        regularization=regularization,
        cv_scheme=cv_scheme,
        loss_function=loss_function,
        model_kwargs=kwargs,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Default config template
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_YAML = """\
experiment:
  id: null                        # auto-generated if null
  output_dir: ~/.macrocast/results
  horizons: [1, 3, 6, 12]
  window: expanding               # expanding | rolling
  rolling_size: null
  oos_start: null                 # null = 80th percentile of sample
  oos_end: null
  n_jobs: -1

data:
  dataset: fred_md                # fred_md | fred_qd | fred_sd
  vintage: null                   # null = current release; YYYY-MM for vintage
  target: INDPRO
  cache_dir: null                 # null = ~/.macrocast/cache

features:
  n_factors: 8
  n_lags: 4
  use_factors: true
  standardize_X: true
  standardize_Z: false
  lookback: 12

models:
  - name: krr
    regularization: factors
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
      gamma_grid: [0.01, 0.1, 1.0]
      cv_folds: 5

  - name: rf
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      n_estimators: 500

  - name: xgboost
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2

  - name: nn
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [64, 128]
      max_epochs: 200
      patience: 20

  - name: lstm
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [32, 64]
      max_epochs: 200
"""
