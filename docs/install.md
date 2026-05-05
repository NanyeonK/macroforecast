# Installation

## Requirements

- Python 3.10 or later
- pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib, PyYAML

## PyPI namespace notice

The `macrocast` PyPI namespace is currently held by an unrelated 2017
package (`macrocast 0.0.2` by Amir Sani). `pip install macrocast` will
install **that** package, not this one. Until the namespace is resolved,
install from GitHub.

## Install from a tagged release (recommended)

```bash
pip install "git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
```

Optional extras use the same `package[extra] @ git+url` syntax:

```bash
pip install "macrocast[deep] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
pip install "macrocast[xgboost,lightgbm] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
pip install "macrocast[tuning] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
pip install "macrocast[shap] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
```

## Install from source (development)

```bash
git clone https://github.com/NanyeonK/macrocast.git
cd macrocast
pip install -e ".[dev]"
```

## Verify installation

```python
import macrocast
print(f"macrocast version: {macrocast.__version__}")     # 0.5.1
```

Run the test suite:

```bash
python -m pytest tests/ -x -q -m "not deep"
```

Expected: 953 tests pass / 12 skipped in ~25 seconds (CI baseline).

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
| `deep` extra (`torch`) | LSTM / GRU / TCN model families | see `[deep]` section below |

Install all optional dependencies at once:

```bash
pip install "macrocast[all] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
```

All optional dependencies are import-guarded. The package works without them, but the corresponding features will raise `ImportError` with a clear message when invoked.

### The `[deep]` extra

The `lstm`, `gru`, and `tcn` model families ship behind an opt-in `[deep]` extra so core installs stay free of a torch dependency:

```bash
pip install "macrocast[deep] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
```

Without the extra, referencing `model_family` in {lstm, gru, tcn} at sweep time raises a clear `ExecutionError` with the install hint.

For CPU-only torch (sufficient unless a GPU sweep is planned):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install "macrocast[deep] @ git+https://github.com/NanyeonK/macrocast.git@v0.5.1"
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
