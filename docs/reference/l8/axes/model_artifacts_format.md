# `model_artifacts_format`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``model_artifacts_format`` on sub-layer ``L8_B_saved_objects`` (layer ``l8``).

## Sub-layer

**L8_B_saved_objects**

## Axis metadata

- Default: `'pickle'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `joblib`  --  operational

Default sklearn / xgboost serialisation via joblib.

Optimised for numpy-array-heavy estimators (sklearn / xgboost / lightgbm). Smaller and faster than plain pickle for typical sklearn fitted-model graphs.

**When to use**

Default; broad compatibility across sklearn / xgboost / lightgbm.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`pickle`](#pickle), [`onnx`](#onnx), [`pmml`](#pmml)

_Last reviewed 2026-05-05 by macroforecast author._

### `onnx`  --  operational

ONNX export (where supported by the family).

Open Neural Network Exchange format. Cross-language deployment (C++ / C# / Java / JS runtimes) and faster inference than the native sklearn pickle. Supported for sklearn / xgboost / lightgbm / pytorch families; raises if the active L4 family lacks an ONNX exporter.

**When to use**

Cross-language deployment; production inference servers.

**When NOT to use**

Models without ONNX support (BVAR, DFM, MRF, custom callables).

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'
* ONNX specification. <https://onnx.ai/>

**Related options**: [`joblib`](#joblib), [`pmml`](#pmml)

_Last reviewed 2026-05-05 by macroforecast author._

### `pickle`  --  operational

Plain Python pickle (less efficient than joblib).

Compatibility option for older toolchains or non-sklearn estimators that don't benefit from joblib's array optimisation. Larger files but maximally portable across Python versions.

**When to use**

Compatibility with older toolchains.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`joblib`](#joblib), [`onnx`](#onnx), [`pmml`](#pmml)

_Last reviewed 2026-05-05 by macroforecast author._

### `pmml`  --  operational

PMML export (PMML-compatible families only).

Predictive Model Markup Language; XML-based exchange format primarily used in enterprise / Java deployments. Supported for linear / tree-family models via ``sklearn2pmml``.

**When to use**

Enterprise / Java deployment. Selecting ``pmml`` on ``l8.model_artifacts_format`` activates this branch of the layer's runtime.

**When NOT to use**

Modern ML deployment -- ONNX is more widely supported.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'
* PMML 4.4 specification. <https://dmg.org/pmml/v4-4/GeneralStructure.html>

**Related options**: [`joblib`](#joblib), [`onnx`](#onnx)

_Last reviewed 2026-05-05 by macroforecast author._
