# Compatibility patches relative to upstream 1.0.6

Upstream: <https://github.com/RyanLucas3/MacroRandomForest> tag/release `1.0.6`
(2022-07-28). This fork's `1.0.6.post1` applies the following minimal
patches. No algorithmic changes.

## Patch 1 — `MRF.py:_name_translations` (np.where 0-D)

Before:
```python
self.trend_pos = np.where(self.S_pos == self.trend_pos)
```

After:
```python
self.trend_pos = np.where(np.asarray(self.S_pos) == self.trend_pos)
```

**Why:** When `S_pos` is supplied as a Python list, the equality reduces to
a Python `bool`. `np.where(scalar_bool)` calls `nonzero` on a 0-D array,
which numpy 2.x rejects with `ValueError: Calling nonzero on 0d arrays is
not allowed.` Coercing through `np.asarray` ensures the comparison returns
a 1-D mask in every entry pattern (list / tuple / ndarray).

## Patch 2 — `MRF.py:_input_safety_checks` (foolproof 1)

Before:
```python
self.trend_pos = np.where(self.x_pos == self.trend_pos)  # foolproof 1
```

After:
```python
self.trend_pos = np.where(np.asarray(self.x_pos) == self.trend_pos)
```

**Why:** Same 0-D `nonzero` rejection as Patch 1, on the foolproof branch
that fires when `y_pos` overlaps `x_pos`.

## Patch 3 — `MRF.py:_pred_given_tree` (matrix → ndarray)

Before:
```python
fitted_vals = zz_all.T @ ((1-self.HRW) * beta_hat + self.HRW*b0)
for j in range(len(fitted_vals)):
    fitted[ind_all[j]] = fitted_vals[j]
```

(both branches: `len(ind_all) == 1` and `else`)

After:
```python
fitted_vals = np.asarray(
    zz_all.T @ ((1-self.HRW) * beta_hat + self.HRW*b0)
).ravel()
for j in range(len(fitted_vals)):
    fitted[ind_all[j]] = fitted_vals[j]
```

**Why:** `zz_all` is a `np.matrix`, so `zz_all.T @ vector` returns a 2-D
matrix. Indexing with `fitted_vals[j]` returns a 1×1 matrix, and numpy 2.x
rejects assigning a 1×1 matrix to an int-indexed scalar slot
(`ValueError: setting an array element with a sequence`). Flattening to a
1-D ndarray restores scalar-slot compatibility while preserving the
algorithm: the matrix product result is a 1-D vector regardless of the
intermediate dtype.

## Patch 4 — `MRF.py:_ensemble_loop` (set_index → direct index)

Before:
```python
self.avg_pred = pd.DataFrame(self.avg_pred, columns=['Ensembled_Prediction'])
self.avg_pred.set_index(self.oos_pos, inplace=True)
```

After:
```python
self.avg_pred = pd.DataFrame(self.avg_pred, columns=['Ensembled_Prediction'])
self.avg_pred.index = pd.Index(self.oos_pos)
```

**Why:** `DataFrame.set_index(keys, ...)` interprets a list `keys` as column
selectors. Pandas 2.x raises `KeyError: 'None of [...] are in the columns'`
because the integers in `oos_pos` are row positions, not column names.
Direct index assignment via `pd.Index` is the documented modern path.

## Verification

Both serial (`parallelise=False`) and parallel (`parallelise=True, n_cores>1`)
paths produce the expected `{'pred', 'betas', 'pred_ensemble', ...}` output
dict on numpy 2.4 + pandas 3.0 (Python 3.14). Smoke test (B=50, n=200,
5 features, oos_pos=last 50) returns `pred` shape `(50, 1)` indexed by
`oos_pos`, `betas` shape `(200, 6)`, `pred_ensemble` shape `(50, 50)`.
