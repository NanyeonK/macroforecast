# macrocast.pipeline

Full API reference for the pipeline layer modules.

---

## macrocast.pipeline.components

::: macrocast.pipeline.components
    options:
      members:
        - Nonlinearity
        - Regularization
        - CVScheme
        - CVSchemeType
        - LossFunction
        - Window
      show_root_heading: true

---

## macrocast.pipeline.estimator

::: macrocast.pipeline.estimator
    options:
      members:
        - MacrocastEstimator
        - SequenceEstimator
      show_root_heading: true

---

## macrocast.pipeline.features

::: macrocast.pipeline.features
    options:
      members:
        - FeatureBuilder
      show_root_heading: true

---

## macrocast.pipeline.models

::: macrocast.pipeline.models
    options:
      members:
        - KRRModel
        - SVRRBFModel
        - SVRLinearModel
        - RFModel
        - XGBoostModel
        - NNModel
        - LSTMModel
      show_root_heading: true

---

## macrocast.pipeline.experiment

::: macrocast.pipeline.experiment
    options:
      members:
        - ModelSpec
        - FeatureSpec
        - ForecastExperiment
      show_root_heading: true

---

## macrocast.pipeline.results

::: macrocast.pipeline.results
    options:
      members:
        - ForecastRecord
        - ResultSet
      show_root_heading: true
