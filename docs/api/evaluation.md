# macrocast.evaluation

Full API reference for the evaluation layer modules.

---

## macrocast.evaluation.metrics

::: macrocast.evaluation.metrics
    options:
      members:
        - msfe
        - mae
        - relative_msfe
        - csfe
        - oos_r2
      show_root_heading: true

---

## macrocast.evaluation.decomposition

::: macrocast.evaluation.decomposition
    options:
      members:
        - decompose_treatment_effects
        - DecompositionResult
      show_root_heading: true

---

## macrocast.evaluation.mcs

::: macrocast.evaluation.mcs
    options:
      members:
        - mcs
        - MCSResult
      show_root_heading: true

---

## macrocast.evaluation.dm

::: macrocast.evaluation.dm
    options:
      members:
        - dm_test
        - DMResult
      show_root_heading: true

---

## macrocast.evaluation.regime

::: macrocast.evaluation.regime
    options:
      members:
        - regime_conditional_msfe
        - RegimeResult
      show_root_heading: true

---

## macrocast.evaluation.dual

::: macrocast.evaluation.dual
    options:
      members:
        - krr_dual_weights
        - tree_dual_weights
        - nn_dual_weights
        - effective_history_length
        - top_analogies
      show_root_heading: true

---

## macrocast.evaluation.pbsv

::: macrocast.evaluation.pbsv
    options:
      members:
        - oshapley_vi
        - compute_pbsv
        - model_accordance_score
      show_root_heading: true
