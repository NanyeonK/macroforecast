# 4. Detail (code)

Detail docs are the full macrocast layer grammar. Read them when you need exact YAML, runtime artifacts, custom method hooks, or compatibility rules.

Use the layers in order. Earlier layers define the data and representation contract that later layers consume.

## 4.1 Layer 0: Study Design / Execution Grammar

[Open Layer 0](layer0/index.md)

Layer 0 decides the shape of the study before data or models are chosen.

- research design;
- experiment unit;
- failure policy;
- reproducibility mode;
- compute mode.

## 4.2 Layer 1: Data Task / Official Frame

[Open Layer 1](layer1/index.md)

Layer 1 decides the official data task and produces the data frame that later layers are allowed to use.

- dataset and source adapter;
- target structure and variable universe;
- information set, vintage, release lag, and contemporaneous-X rules;
- raw source missing/outlier handling;
- official transform policy.

## 4.3 Layer 2: Representation / Research Preprocessing

[Open Layer 2](layer2/index.md)

Layer 2 turns the Layer 1 official frame into the representation used by forecast generators.

- target construction and target scaling;
- post-official-frame missing/outlier handling;
- lag, factor, rotation, and feature-block construction;
- feature selection and representation sweep choices;
- custom representation hooks.

## 4.4 Layer 3: Forecast Generator

[Open Layer 3](layer3/index.md)

Layer 3 consumes the Layer 2 representation and generates forecasts.

- model family and benchmark choices;
- forecast type and forecast object;
- training window and tuning policy;
- future-X and iterated-forecast paths;
- payload and model-extension contracts.

## 4.5 Layer 4: Evaluation

[Open Layer 4](layer4/index.md)

Layer 4 evaluates forecast artifacts without changing the forecast-generation path.

- metric family;
- comparison scope;
- aggregation and ranking;
- regime and OOS-period choices.

## 4.6 Layer 5: Output / Provenance

[Open Layer 5](layer5/index.md)

Layer 5 decides how artifacts, manifests, and saved objects are written.

- export format;
- artifact granularity;
- saved-object policy;
- manifest and sidecar contracts.

## 4.7 Layer 6: Statistical Tests

[Open Layer 6](layer6/index.md)

Layer 6 applies statistical tests to already generated forecast and evaluation artifacts.

- equal predictive ability tests;
- nested-model tests;
- multiple-model tests;
- density and direction tests;
- multiple-testing correction.

## 4.8 Layer 7: Interpretation / Importance

[Open Layer 7](layer7/index.md)

Layer 7 explains fitted models and forecast paths without changing the forecast itself.

- importance family;
- SHAP and surrogate methods;
- partial dependence;
- grouping, stability, and temporal output.
