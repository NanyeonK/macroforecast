# Installation

## Requirements

- Python 3.10 or later
- pandas, numpy, scikit-learn, statsmodels, PyYAML

## Install from source

```bash
git clone https://github.com/your-org/macroforecast.git
cd macroforecast
pip install -e .
```

## Verify installation

```python
import macrocast
print(f"macrocast imported successfully")
print(f"Available functions: {len(dir(macrocast))}")
```

Run the test suite:

```bash
python -m pytest tests/ -x -q
```

Expected: 291 tests pass in ~3 minutes.

## Optional dependencies

macrocast has several optional dependencies for specific features. Install only what you need:

| Package | Required for | Install |
|---------|-------------|---------|
| `optuna` | Bayesian optimization tuning | `pip install optuna` |
| `shap` | TreeSHAP, KernelSHAP, LinearSHAP importance | `pip install shap` |
| `lime` | LIME local surrogate importance | `pip install lime` |
| `xgboost` | XGBoost model family | `pip install xgboost` |
| `lightgbm` | LightGBM model family | `pip install lightgbm` |
| `catboost` | CatBoost model family | `pip install catboost` |
| `openpyxl` | FRED-SD Excel workbook loading | `pip install openpyxl` |
| `deep` extra (`torch`) | LSTM / GRU / TCN model families | `pip install macrocast[deep]` |

Install all optional dependencies at once:

```bash
pip install "macrocast[all]"
```

All optional dependencies are import-guarded. The package works without them, but the corresponding features will raise `ImportError` with a clear message when invoked.

### The `[deep]` extra

The `lstm`, `gru`, and `tcn` model families ship behind an opt-in `[deep]` extra so core installs stay free of a torch dependency:

```bash
pip install macrocast[deep]
```

Without the extra, referencing `model_family` in {lstm, gru, tcn} at sweep time raises a clear `ExecutionError` with the install hint.

CPU-only torch is sufficient for Phase 5 — GPU / multi-node workflows are a Phase 11 deliverable. For strict reproducibility install the CPU wheel explicitly:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install macrocast[deep]
```


## Core dependencies (automatically installed)

| Package | Purpose |
|---------|---------|
| `pandas` | Data handling and DataFrame output |
| `numpy` | Numerical computation |
| `scikit-learn` | Model families (Ridge, Lasso, RF, etc.), preprocessing, CV |
| `statsmodels` | AR models, statistical tests |
| `PyYAML` | Recipe YAML parsing |

**See also:** [Getting Started: Quickstart](getting_started/quickstart.md)
