# API Reference

Full auto-generated API documentation for all three macrocast layers.

---

## Module Structure

| Module | Contents |
|--------|----------|
| `macrocast` | Top-level namespace; re-exports all public symbols |
| `macrocast.data` | FRED-MD/QD/SD loaders, MacroFrame, transforms, missing, vintages |
| `macrocast.pipeline` | ForecastExperiment, model zoo, components, features, results |
| `macrocast.evaluation` | Metrics, decomposition, MCS, DM test, regime, dual weights, PBSV |
| `macrocast.utils.cache` | Cache management utilities |

---

## Top-Level Namespace

Public symbols from all layers are re-exported at the package level:

```python
import macrocast as mc

# Data layer
mc.load_fred_md()
mc.load_fred_qd()
mc.load_fred_sd()
mc.MacroFrame
mc.MacroFrameMetadata
mc.VariableMetadata
mc.TransformCode
mc.apply_tcode
mc.apply_tcodes
mc.classify_missing
mc.handle_missing
mc.list_available_vintages
mc.load_vintage_panel
mc.RealTimePanel

# Pipeline layer
mc.ForecastExperiment
mc.ModelSpec
mc.FeatureSpec
mc.FeatureBuilder
mc.Nonlinearity
mc.Regularization
mc.CVScheme
mc.LossFunction
mc.Window
mc.MacrocastEstimator
mc.SequenceEstimator
mc.KRRModel
mc.SVRRBFModel
mc.SVRLinearModel
mc.RFModel
mc.XGBoostModel
mc.NNModel
mc.LSTMModel
mc.ForecastRecord
mc.ResultSet

# Evaluation layer
mc.msfe
mc.mae
mc.relative_msfe
mc.csfe
mc.oos_r2
mc.decompose_treatment_effects
mc.DecompositionResult
mc.mcs
mc.MCSResult
mc.dm_test
mc.DMResult
mc.regime_conditional_msfe
mc.RegimeResult
mc.krr_dual_weights
mc.tree_dual_weights
mc.nn_dual_weights
mc.effective_history_length
mc.top_analogies
mc.oshapley_vi
mc.compute_pbsv
mc.model_accordance_score
```

---

## Auto-Generated Reference

::: macrocast
    options:
      members:
        - load_fred_md
        - load_fred_qd
        - load_fred_sd
        - MacroFrame
        - MacroFrameMetadata
        - VariableMetadata
        - TransformCode
        - apply_tcode
        - apply_tcodes
        - classify_missing
        - handle_missing
        - list_available_vintages
        - load_vintage_panel
        - RealTimePanel
      show_root_heading: true
      show_source: false
