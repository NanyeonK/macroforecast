# Installation

## Requirements

- Python 3.10 or later
- pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib, openpyxl, PyYAML

## Install from PyPI

```bash
pip install macroforecast
```

Optional extras:

```bash
pip install 'macroforecast[deep]'
pip install 'macroforecast[xgboost,lightgbm]'
pip install 'macroforecast[tuning]'
pip install 'macroforecast[shap]'
pip install 'macroforecast[all]'
```

Or pin to a tagged GitHub release directly:

```bash
pip install "git+https://github.com/NanyeonK/macroforecast.git@v0.8.6"
```

## Install from source (development)

```bash
git clone https://github.com/NanyeonK/macroforecast.git
cd macroforecast
pip install -e ".[dev]"
```

## Verify installation

```python
import macroforecast
print(f"macroforecast version: {macroforecast.__version__}")     # 0.7.0
```

Run the test suite:

```bash
python -m pytest tests/ -x -q -m "not deep"
```

Expected: 953 tests pass / 12 skipped in ~25 seconds (CI baseline).

## Optional dependencies

macroforecast has several optional dependencies for specific features. Install only what you need:

| Package | Required for | Install |
|---------|-------------|---------|
| `optuna` | Bayesian optimization tuning | `pip install optuna` |
| `shap` | TreeSHAP, KernelSHAP, LinearSHAP importance | `pip install shap` |
| `lime` | LIME local surrogate importance | `pip install lime` |
| `xgboost` | XGBoost model family | `pip install xgboost` |
| `lightgbm` | LightGBM model family | `pip install lightgbm` |
| `catboost` | CatBoost model family | `pip install catboost` |
| `deep` extra (`torch`) | LSTM / GRU / TCN model families | see `[deep]` section below |

(``openpyxl`` is now a core dependency in v0.6.3+ since FRED-SD Excel workbook
loading is a baseline FRED-SD code path.)

Install all optional dependencies at once:

```bash
pip install "macroforecast[all] @ git+https://github.com/NanyeonK/macroforecast.git@v0.8.6"
```

All optional dependencies are import-guarded. The package works without them, but the corresponding features will raise `ImportError` with a clear message when invoked.

### The `[deep]` extra

The `lstm`, `gru`, and `tcn` model families ship behind an opt-in `[deep]` extra so core installs stay free of a torch dependency:

```bash
pip install "macroforecast[deep] @ git+https://github.com/NanyeonK/macroforecast.git@v0.8.6"
```

Without the extra, referencing `model_family` in {lstm, gru, tcn} at sweep time raises a clear `ExecutionError` with the install hint.

For CPU-only torch (sufficient unless a GPU sweep is planned):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install "macroforecast[deep] @ git+https://github.com/NanyeonK/macroforecast.git@v0.8.6"
```


## Core dependencies (automatically installed)

| Package | Purpose |
|---------|---------|
| `pandas` | Data handling and DataFrame output |
| `numpy` | Numerical computation |
| `scikit-learn` | Model families (Ridge, Lasso, RF, etc.), preprocessing, CV |
| `statsmodels` | AR models, statistical tests |
| `scipy` | Statistical routines and numerical utilities |
| `matplotlib` | L7 figure rendering (bar / heatmap / pdp / US choropleth) |
| `openpyxl` | FRED-SD Excel workbook loading |
| `PyYAML` | Recipe YAML parsing |

**See also:** [Researchers: Quickstart](for_researchers/quickstart.md)
