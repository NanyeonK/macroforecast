# Phase 05 — Deep Models (LSTM / GRU / TCN + sequence adapter)

| Field | Value |
|-------|-------|
| Phase ID | phase-05 |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-07 (decomposition gains deep-family variance), phase-08 (paper-ready bundle references deep entries) |
| Version tag target | v0.7 |
| Status | in_progress |

## 1. Goal

Open macrocast's deep-learning slot. Core install stays sklearn-only; an opt-in `[deep]` extra adds PyTorch-based LSTM / GRU / TCN families that plug into the existing sweep runner via the same `_run_<family>_autoreg_executor` contract every other model_family uses.

Phase 5 is the *minimum viable* deep catalog — three architectures chosen because they span the three standard inductive biases used in macro time-series literature (RNN gating, gated-RNN-lite, dilated-causal-conv). Richer deep catalogs (Transformer / NBEATS / TFT) are deferred to the Phase 10 (v1.1) catalog. Classical state-space / TVP_AR / MIDAS are deferred to Phase 11 (v2). VAR / BVAR are **not** included — factor-based multivariate baselines already exist in the core (`factor_augmented_linear`, `pcr`, `pls`, `ardi`, `factor_model`, `factor_model_benchmark`), and small-system VAR adds identity without measurable value in the CLSS-style horse races Phase 7 is designed for.

## 2. Scope

**In scope:**
- 3 new model_family values: `lstm`, `gru`, `tcn` (all require `[deep]` extra)
- `execution/adapters/sequence.py` — flat-series → (n_windows, lookback, n_features) reshape utility
- `execution/models/deep/{_base,lstm,gru,tcn}.py` — adapter modules
- `pyproject.toml` `[deep]` optional-dependency (**done** as of commit `26b7fdf`: `torch>=2.0`; **follow-up will drop pytorch-lightning** — not used by the chosen implementation)
- `docs/conf.py` `autodoc_mock_imports` extension (**done** as of commit `26b7fdf`)
- `execution/models/deep/_import_guard.py` — `ExecutionError` with install hint when `[deep]` absent (**done** as of commit `26b7fdf`)
- `.github/workflows/ci-deep.yml` — installs `.[deep]`, runs `pytest -m deep`
- Per-model tests + sweep-safety test + baseline comparison test + missing-extra test
- Docs additions: `install.md` `[deep]` section, `user_guide/model_catalog.md` extension, `api/models/deep.md` autodoc page

**Out of scope:**
- VAR / BVAR — dropped from Phase 5; identity covered by existing factor-based axes. If a genuine multivariate baseline is later demanded, a Large-BVAR (Banbura-Giannone-Reichlin 2010) will be scoped under Phase 11.
- Transformer / NBEATS / TFT / DFM / FAVAR → Phase 10 (v1.1 scope catalog)
- state_space / TVP_AR / MIDAS → Phase 11 (v2 scope catalog)
- Hyperparameter tuning for deep models — fixed `DeepModelConfig` defaults only. `optuna`-backed HP search for deep families is a Phase 10 deliverable.
- Raw-panel feature input for deep models — Phase 5 ships univariate autoreg only; raw-panel deep executors are Phase 10.
- GPU / multi-node / distributed training — Phase 11.
- `pytorch-lightning` integration — dropped from `[deep]` extra in a follow-up commit on this branch; plain torch is enough for three MSE regression models.
- Probabilistic / quantile deep heads — Phase 11.7.

## 3. Sub-Tasks

| ID | Sub-task | Priority | Est LOC | Files | Gate | Status |
|:---:|---------|:--------:|:-------:|-------|------|:------:|
| 05.1 | `pyproject.toml` `[deep]` extra | P1 | ~10 | `pyproject.toml` | `pip install .[deep]` works | **done** (commit `26b7fdf`) |
| 05.2 | `docs/conf.py` autodoc mock | P1 | ~2 | `docs/conf.py` | core-only RTD build green | **done** (commit `26b7fdf`) |
| 05.3 | `_import_guard.py` | P1 | ~30 | `macrocast/execution/models/deep/_import_guard.py` | `require_torch("lstm")` raises `ExecutionError` without torch | **done** (commit `26b7fdf`) |
| 05.4 | Drop `pytorch-lightning` from `[deep]` | P1 | ~3 | `pyproject.toml` | `pip install .[deep]` installs torch only | pending |
| 05.5 | Sequence adapter | P1 | ~100 | `macrocast/execution/adapters/sequence.py` | `test_sequence_adapter.py` green | pending |
| 05.6 | `DeepModelConfig` + `_BaseDeepModel` | P1 | ~200 | `macrocast/execution/models/deep/_base.py` | base class unit test green | pending |
| 05.7 | `LSTMModel` / `GRUModel` / `TCNModel` | P1 | ~500 | `macrocast/execution/models/deep/{lstm,gru,tcn}.py` | per-model `fit` + `predict` deterministic | pending |
| 05.8 | Registry entries — `lstm`, `gru`, `tcn` | P1 | ~15 | `macrocast/registry/training/model_family.py` | 3 new `EnumRegistryEntry` with `status='operational'`, `priority='A'` | pending |
| 05.9 | Executor dispatch in `build.py` | P1 | ~150 | `macrocast/execution/build.py` | 3 new `_run_{lstm,gru,tcn}_autoreg_executor` functions wired into dispatch | pending |
| 05.10 | `ci-deep.yml` GitHub Actions | P1 | ~80 | `.github/workflows/ci-deep.yml` | deep install + `pytest -m deep` green | pending |
| 05.11 | `pytest` markers | P1 | ~5 | `pyproject.toml` | `deep` registered under `[tool.pytest.ini_options].markers` | pending |
| 05.12 | Tests (7 files) | **P0** | ~700 | see 6 | all green (deep tests run only when torch is present) | pending |
| 05.13 | Docs (3 additions + 1 extension) | P1 | ~400 | see 8 | RTD build green including deep autodoc stubs | pending |

**Progress pointer:** sub-tasks 05.1 / 05.2 / 05.3 landed in commit `26b7fdf` on branch `feat/phase-05-deep-models` (né `feat/phase-05a-deep-tsm`, to be renamed at push time).

## 4. API / Schema Specifications

### 4.1 Install-time selection policy

```bash
pip install macrocast          # core only; lstm/gru/tcn raise ExecutionError if requested
pip install macrocast[deep]    # + torch; all three families become operational
```

When `[deep]` is absent and a recipe requests `model_family in {lstm, gru, tcn}`:

```python
raise ExecutionError(
    f"model_family {model_family!r} requires the [deep] extra. "
    "Install with: pip install macrocast[deep]"
)
```

This is dispatched through `_import_guard.require_torch(model_family)` on the first line of each deep executor.

### 4.2 `pyproject.toml` `[deep]` extra (final)

```toml
[project.optional-dependencies]
deep = ["torch>=2.0"]
```

(`pytorch-lightning` is **not** in the final `[deep]` set — see 05.4.)

### 4.3 Import guard (already landed)

See `macrocast/execution/models/deep/_import_guard.py` (commit `26b7fdf`).

### 4.4 Sequence adapter

```python
# macrocast/execution/adapters/sequence.py
from __future__ import annotations

import numpy as np


def reshape_for_sequence(
    *,
    series: np.ndarray,    # shape (T,) univariate
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat 1-D time series → supervised sequence windows.

    Returns
    -------
    X_seq : ndarray, shape (n_windows, lookback, 1)
        Each row i is series[i : i + lookback], reshaped to have a
        trailing feature axis of size 1 so downstream torch layers see
        (batch, time, features).
    y_seq : ndarray, shape (n_windows,)
        y_seq[i] = series[i + lookback + horizon - 1].

    Windows whose target index is beyond len(series) - 1 are dropped.

    Raises
    ------
    ValueError
        If len(series) < lookback + horizon (no valid window).
    """
```

Minimum valid window: `lookback + horizon`. For horizon=1, lookback=12, need ≥ 13 observations. Executor enforces this with a clear `ExecutionError` if the training series is shorter.

### 4.5 `DeepModelConfig` (fixed defaults — no tuning in Phase 5)

```python
# macrocast/execution/models/deep/_base.py
from dataclasses import dataclass

@dataclass(frozen=True)
class DeepModelConfig:
    lookback: int = 12
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10
    validation_fraction: float = 0.2
    seed: int = 0            # overwritten by current_seed(model_family=...)
```

`validation_fraction` controls the chronological tail of training windows that becomes the validation split for early stopping. If the resulting validation set would have fewer than 5 windows, early stopping is disabled and training runs for `max_epochs` unconditionally.

### 4.6 `_BaseDeepModel` contract

```python
# macrocast/execution/models/deep/_base.py
class _BaseDeepModel:
    """All three Phase-5 deep models inherit this class.

    Subclasses implement `_build_net(self, n_features: int) -> torch.nn.Module`;
    fit / predict / the training loop / early stopping / seeding are shared.
    """

    config: DeepModelConfig
    model_family: str    # 'lstm' | 'gru' | 'tcn'

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray) -> "_BaseDeepModel": ...
    def predict_next(self, history: np.ndarray) -> float:
        """Given the trailing `lookback` values of the series, return the next
        1-step forecast as a Python float."""
        ...
```

Seed handling inside `__init__`:

```python
seed = self.config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

CUDA seeding is intentionally skipped (Phase 5 default runs on CPU; if CUDA is available the executor forces `.to('cpu')` to stay deterministic under the Phase 0 seed policy).

`torch.use_deterministic_algorithms(True)` is **not** set — too many common ops raise under it (e.g., `LSTM` cuDNN paths). Reproducibility target for Phase 5 is "bit-identical on the same machine with the same torch + numpy + Python" — stronger guarantees are a Phase 11 concern.

### 4.7 Training loop

Shared across all 3 families:

- Optimizer: `torch.optim.Adam(lr=config.learning_rate)`
- Loss: `torch.nn.MSELoss()`
- Batching: `DataLoader(TensorDataset(X, y), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed))`
- Validation split: last `validation_fraction` of the (n_windows, ...) tensor **chronologically** — no shuffle on the split itself.
- Early stopping: monitor val MSE each epoch; if no improvement for `early_stopping_patience` consecutive epochs, restore best weights and stop.
- Best-weights snapshot: `copy.deepcopy(net.state_dict())` on every val-MSE improvement.

Deterministic batch shuffling is tied to `config.seed` via `Generator().manual_seed(seed)` — identical recipes produce identical mini-batch orderings.

### 4.8 Per-family architectures

**LSTM (`lstm.py`)**

```python
class LSTMModel(_BaseDeepModel):
    model_family = "lstm"

    def _build_net(self, n_features: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            _Packer("lstm", torch.nn.LSTM(
                input_size=n_features,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.n_layers,
                dropout=self.config.dropout if self.config.n_layers > 1 else 0.0,
                batch_first=True,
            )),
            torch.nn.Linear(self.config.hidden_size, 1),
        )
```

**GRU (`gru.py`)**: identical structure, `torch.nn.GRU` swapped in.

**TCN (`tcn.py`)**: a single-stack causal dilated-conv implementation — no external package dep.

```python
class _TemporalBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        ...
class TCNModel(_BaseDeepModel):
    model_family = "tcn"
    def _build_net(self, n_features):
        channels = [self.config.hidden_size] * self.config.n_layers
        layers = []
        for i, c_out in enumerate(channels):
            c_in = n_features if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(_TemporalBlock(c_in, c_out, kernel_size=3,
                                         dilation=dilation,
                                         dropout=self.config.dropout))
        return torch.nn.Sequential(*layers, _LastTimeStep(),
                                   torch.nn.Linear(channels[-1], 1))
```

`_Packer` / `_LastTimeStep` are private helpers in `_base.py` that unpack `(output, state)` tuples from `nn.LSTM`/`nn.GRU` and select the final timestep respectively.

### 4.9 Registry entries (minimal additions)

Append three rows to `macrocast/registry/training/model_family.py` inside the existing `AXIS_DEFINITION.entries` tuple:

```python
EnumRegistryEntry(id='lstm', description='lstm',  status='operational', priority="A"),
EnumRegistryEntry(id='gru',  description='gru',   status='operational', priority="A"),
EnumRegistryEntry(id='tcn',  description='tcn',   status='operational', priority="A"),
```

No structural change to `EnumRegistryEntry` — the `[deep]` requirement is handled entirely at dispatch time through `require_torch`. A separate metadata mapping (`REQUIRES_EXTRA = {"lstm": "deep", ...}`) is **not** introduced in Phase 5; sweep-time users discover the `[deep]` requirement through the `ExecutionError` when the family is exercised. Phase 10 can promote this to a schema-level flag if the richer catalog warrants it.

### 4.10 Executor dispatch integration

Each family gets one `_run_<family>_autoreg_executor` function appended to `macrocast/execution/build.py` with the same signature every other autoreg executor uses:

```python
def _run_lstm_autoreg_executor(
    train: pd.Series,
    horizon: int,
    recipe: RecipeSpec,
    contract: PreprocessContract,
    raw_frame: pd.DataFrame | None = None,
    origin_idx: int | None = None,
    start_idx: int = 0,
) -> dict[str, float | int]:
    from .models.deep._import_guard import require_torch
    require_torch("lstm")
    from .models.deep.lstm import LSTMModel
    from .models.deep._base import DeepModelConfig
    from .adapters.sequence import reshape_for_sequence

    cfg = DeepModelConfig(seed=current_seed(model_family="lstm"))
    X_seq, y_seq = reshape_for_sequence(
        series=train.to_numpy(dtype=float),
        lookback=cfg.lookback,
        horizon=horizon,
    )
    model = LSTMModel(config=cfg).fit(X_seq, y_seq)
    history = train.to_numpy(dtype=float)[-cfg.lookback:]
    y_pred = model.predict_next(history)
    # Recursive multi-step: for h > 1, iterate
    for _ in range(horizon - 1):
        history = np.concatenate([history[1:], [y_pred]])
        y_pred = model.predict_next(history)
    return {
        "y_pred": float(y_pred),
        "selected_lag": cfg.lookback,
        "selected_bic": math.nan,
        "tuning_payload": {},
    }
```

`_run_gru_autoreg_executor` and `_run_tcn_autoreg_executor` are byte-identical apart from the three module / family name swaps.

The dispatch dict (`_get_model_executor`) gets three new entries inside the `"autoreg"` branch; no new feature_builder. Raw-panel deep executors are **not** added in Phase 5.

### 4.11 pytest marker registration

`pyproject.toml` gains:

```toml
[tool.pytest.ini_options]
markers = [
    "deep: tests that require torch (skipped unless [deep] extra is installed)",
]
```

Deep-model test files start with:

```python
import pytest
pytest.importorskip("torch")
pytestmark = pytest.mark.deep
```

— `importorskip` handles the core-only install case by skipping the whole file, and the marker lets CI scope deep tests into the dedicated workflow.

## 5. File Layout

**New files (implementation):**
- `macrocast/execution/adapters/sequence.py`
- `macrocast/execution/models/deep/_base.py`
- `macrocast/execution/models/deep/lstm.py`
- `macrocast/execution/models/deep/gru.py`
- `macrocast/execution/models/deep/tcn.py`

**New files (tests):**
- `tests/test_sequence_adapter.py`
- `tests/test_model_lstm.py` (deep)
- `tests/test_model_gru.py` (deep)
- `tests/test_model_tcn.py` (deep)
- `tests/test_deep_models_sweep_safety.py` (deep)
- `tests/test_deep_models_vs_baseline.py` (deep)
- `tests/test_deep_missing_extra.py`

**New files (CI/docs):**
- `.github/workflows/ci-deep.yml`
- `docs/api/models/deep.md`

**Modified files:**
- `pyproject.toml` — drop `pytorch-lightning`, register `deep` pytest marker
- `macrocast/registry/training/model_family.py` — 3 entries
- `macrocast/execution/build.py` — 3 executors + dispatch wiring
- `docs/install.md` — `[deep]` section
- `docs/user_guide/model_catalog.md` — 3 new family entries

**Already landed (commit `26b7fdf`):**
- `pyproject.toml` `[deep]` entry
- `docs/conf.py` autodoc mock
- `macrocast/execution/models/{__init__,tsm/__init__,deep/__init__}.py` package markers
- `macrocast/execution/models/deep/_import_guard.py`
- `macrocast/execution/adapters/__init__.py`

(The empty `tsm/__init__.py` stays but no `tsm/` content ships in Phase 5. The package marker is harmless and avoids a separate follow-up commit to remove it; Phase 10 will repurpose `tsm/` when DFM / FAVAR arrive, or the marker can be dropped during Phase 9 docs consolidation.)

## 6. Test Strategy

### `tests/test_sequence_adapter.py`
- (T=20, lookback=12, horizon=1) → shapes `(8, 12, 1)`, `(8,)`
- (T=20, lookback=12, horizon=3) → shapes `(6, 12, 1)`, `(6,)`
- (T=12, lookback=12, horizon=1) → `ValueError` (no valid window)
- y_seq alignment: for deterministic `series = np.arange(20)`, verify y_seq[i] == series[i + lookback + horizon - 1]

### `tests/test_model_{lstm,gru,tcn}.py` — `pytest.mark.deep`
Each file has the same three tests with the family swapped in:
- `test_fit_predict_smoke`: synthetic AR(1) series of 100 points, fit with DeepModelConfig(max_epochs=5) for speed, `predict_next` returns finite float
- `test_seed_reproducibility`: two fits with same seed → identical `predict_next` outcome to float equality after reducing `max_epochs=3`
- `test_seed_divergence`: two fits with different seeds → `predict_next` differs by more than 1e-6 (ensures the seed actually threads through)

### `tests/test_sequence_adapter.py` + `tests/test_deep_models_sweep_safety.py` (deep)
Run a 2-variant sweep on a 60-obs synthetic series, both variants exercise `lstm` with different seeds. Assert:
- no shared-tensor mutation (each variant's predictions depend only on its own seed)
- manifest written, `reproducibility_spec` propagated
- final `y_pred` values differ across variants

### `tests/test_deep_models_vs_baseline.py` (deep)
Synthetic AR(2) series 200 obs, horizon=1: each of `lstm` / `gru` / `tcn` (max_epochs=20) achieves val-MSE within 5× of an AR(2)-BIC baseline's val-MSE on the final 40 points. This is a *sanity* bound, not a performance claim — catches broken architectures without enforcing research-grade superiority.

### `tests/test_deep_missing_extra.py` — **not** marked `deep`
Monkey-patch `sys.modules['torch']` to `None` (simulating the core-only install), call `require_torch('lstm')`, assert `ExecutionError` with the expected install-hint message. Runs in every CI matrix row.

### Sweep-runner smoke in main CI
`tests/test_full_operational_sweep.py` already parametrizes across `model_family`. Add `lstm` to the parametrization *only* when torch is importable (`pytest.importorskip` at the test level). Keeps the main pytest matrix unchanged for core-only installs, but exercises the dispatch wiring when torch is present.

## 7. Acceptance Gate

- [x] `[deep]` extra installable (`pip install -e .[deep]`)
- [ ] `pytorch-lightning` removed from `[deep]` extra
- [ ] `sequence_adapter` unit tests green
- [ ] `lstm`, `gru`, `tcn` registered as `operational` in `model_family`
- [ ] 3 autoreg executors in `build.py` dispatch; invoking `execute_recipe` with `model_family=lstm` (+ torch present) returns a finite forecast
- [ ] `_import_guard` raises `ExecutionError` with install hint when torch is missing (verified by `test_deep_missing_extra.py`)
- [ ] `ci-deep.yml` green on PR
- [ ] All 7 Phase 5 test files green (deep tests under `pytest.mark.deep`)
- [ ] Existing 479-test suite remains green
- [ ] RTD build green including the new `docs/api/models/deep.md`
- [ ] `docs/install.md` documents the `[deep]` install flow
- [ ] `docs/user_guide/model_catalog.md` lists `lstm`, `gru`, `tcn`

## 8. Docs Deliverables

**New:**
- `docs/api/models/deep.md` — autodoc page. Because mock-imports cover `torch`, this renders on RTD without requiring `[deep]` at build time.

**Extended:**
- `docs/install.md` — `[deep]` section: install command, torch CPU / CUDA notes (CPU-only wheel recommended for reproducibility in research contexts), troubleshooting for torch wheel / CUDA mismatches.
- `docs/user_guide/model_catalog.md` — new subsection "Deep models" listing `lstm`, `gru`, `tcn` with the fixed default config. Mentions that HP tuning for deep families lands in v1.1.

## 9. Migration Notes

- Existing recipes that don't reference `lstm` / `gru` / `tcn` are unaffected. No registry removals.
- Core-only installs cannot execute the new families; they get a clear `ExecutionError` at sweep time rather than a silent skip — consistent with `macrocast`'s fail-fast policy.
- `pytorch-lightning` was briefly in `[deep]` (commit `26b7fdf`) but is removed before Phase 5 merges; no downstream consumer can rely on it because Phase 5 is the first commit that exposed the extra.

## 10. Cross-references

- ADR-004 (deep-learning-optional): Phase 5 is the first materialization of ADR-004.
- Phase 0 seed policy: `current_seed(model_family=<family>)` is the sole entry point for deterministic seeds in the deep executors.
- Phase 1 sweep runner: deep executors plug into `_get_model_executor` exactly like sklearn families — no sweep-runner changes needed.
- Phase 7 decomposition: deep families become additional points of variance for decomposition analysis once Phase 7 ships.
- Phase 10 (v1.1 scope catalog): successor deep families (Transformer / NBEATS / TFT / DFM / FAVAR) and HP tuning for deep families land there.
- Phase 11 (v2 scope catalog): classical state-space / TVP_AR / MIDAS; distributed / GPU; probabilistic heads.

## 11. Autonomous Execution Checklist

This plan is designed to be executed end-to-end without further design decisions from the user. Before starting implementation, verify:

- All hyperparameters in 4.5 are fixed and no tuning axis is introduced.
- All file paths in 5 are absolute under the repo root and do not overlap with existing files (checked against `ls macrocast/execution/` on branch `feat/phase-05-deep-models`).
- The dispatch integration in 4.10 adds three keys to the `autoreg` branch of `_get_model_executor` without reshaping the existing dispatch dict.
- Seed handling (4.6) uses the existing `current_seed` contract — no new reproducibility knobs.
- Every acceptance-gate checkbox in 7 has a concrete test file in 6 that flips it.

If any of the above becomes false during implementation (e.g. a test reveals that `torch.nn.LSTM` with `num_layers=1` and `dropout>0` raises — a known warning that requires the `if n_layers > 1` guard already in 4.8), resolve by following the guard rails in the plan and noting the incident in 12 Revision Log, not by pausing for user input.

## 12. Revision Log

- 2026-04-17 (first draft of phase-05a): ultraplan v2.2 Phase 5 extraction (included VAR / BVAR)
- 2026-04-17 (plan revision 1a): axis name mapping to existing registry entries
- 2026-04-17 (**phase-05 consolidation**): Phase 5a / 5b / 5c split retired. VAR / BVAR dropped (factor-based multivariate axes already in core). Transformer / NBEATS / TFT / DFM / FAVAR absorbed into Phase 10 (v1.1) catalog. State-space / TVP_AR / MIDAS absorbed into Phase 11 (v2) catalog. `pytorch-lightning` dropped from `[deep]` extra (not needed for three MSE regression models). Plan rewritten as autonomous-execution-ready — every design choice is pre-committed so impl can run without user check-ins. File renamed `phase_05a_deep_tsm.md` → `phase_05_deep_models.md`.
