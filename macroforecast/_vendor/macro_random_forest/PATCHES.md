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

## Patch 5 — `MRF.py:_one_MRF_tree` line 546 (dead-code bayes branch, contiguity safety)

Before:
```python
y = weights[whos_who]*np.matrix(y)
```

After:
```python
y = np.ascontiguousarray(np.atleast_2d(np.asarray(y)))
```

**Why:** The `bayes=True` branch fires only when `min(rando_vec) < 0`, which cannot happen
with any of the four `bootstrap_opt` paths (0–3). The `np.matrix(y)` call on a non-contiguous
pandas Series (from a sliced DataFrame) would raise `ValueError: ndarray is not contiguous`
on numpy 2.x if this branch were reached. The `weights[whos_who]*np.matrix(y)` expression also
has ambiguous matrix-multiply vs broadcast semantics; as dead code, the semantic question is
deferred. The replacement makes the line contiguity-safe.

---

## Patch 6 — `MRF.py:_splitter_mrf` line 705 (root cause of 9 failing tests)

Before:
```python
y = np.matrix(y)
```

After:
```python
y = np.asarray(np.atleast_2d(y))
```

**Why:** `y` is a pandas Series backed by a non-contiguous memory view (result of
`data.iloc[rando_vec, :].iloc[:, y_pos]` where `rando_vec` is a non-sequential list of
sampled indices). `np.matrix()` on a non-contiguous array raises
`ValueError: ndarray is not contiguous` on numpy 2.x. `np.atleast_2d` converts a 1-D Series
to shape `(1, n)` — identical to `np.matrix(Series).shape`. The `.T` transpose downstream
(`z_T @ y.T`) produces `(n, 1)` in both cases. This patch closes all 9 pre-existing MRF
test failures.

---

## Patch 7 — `MRF.py:_pred_given_tree` line 866 (ori_z non-contiguous concat)

Before:
```python
ori_z = np.matrix(self.ori_z)
```

After:
```python
ori_z = np.ascontiguousarray(np.asarray(self.ori_z))
```

**Why:** `self.ori_z` is built by `pd.concat([pd.Series([1]*T), data_ori.iloc[:, z_pos]], axis=1)`.
pd.concat creates a non-contiguous backing array (column-wise concat). `np.matrix()` on this
raises `ValueError: ndarray is not contiguous`. `np.ascontiguousarray(np.asarray(...))` forces
C-order layout. All downstream uses (`ori_z[rows, :]` indexing, `@`-operator matmul) are
semantics-equivalent for ndarray.

---

## Patch 8 — `MRF.py:_pred_given_tree` line 868 (regul_mat defensive)

Before:
```python
regul_mat = np.matrix(rw_regul_dat)
```

After:
```python
regul_mat = np.ascontiguousarray(np.asarray(rw_regul_dat))
```

**Why:** `rw_regul_dat` may be non-contiguous when `z_pos` is non-sequential.
`np.ascontiguousarray` guarantees C-order for safe passing to `_random_walk_regularisation`.

---

## Patch 9 — `MRF.py:_pred_given_tree` line 870 (leafs object-dtype matrix)

Before:
```python
leafs_mat = np.matrix(leafs)
```

After:
```python
leafs_mat = np.asarray(leafs, dtype=object)
```

**Why:** `leafs` is an object-dtype DataFrame (filter strings mixed with float coefficients).
`np.matrix(leafs)` wraps it as object-dtype matrix. `np.asarray(leafs, dtype=object)` produces
an identical 2-D object ndarray. Key difference: `leafs_mat[i, 4:]` now returns a 1-D object
array (shape `(k+1,)`) instead of a `(1, k+1)` matrix row. Consequently `np.transpose(...)` is
a no-op and `b0` becomes 1-D `(k+1,)`. This is consistent with `beta_hat` which is also 1-D
`(k+1,)` after `np.linalg.solve` with a 1-D rhs (confirmed by tracing `yy` through
`_random_walk_regularisation`).

---

## Patch 10 — `MRF.py:_pred_given_tree` line 937 (shape check without np.matrix)

Before:
```python
if np.matrix(np.transpose(zz_all)).shape[0] != 1:
```

After:
```python
if np.asarray(zz_all).T.shape[0] != 1:
```

**Why:** `np.transpose(zz_all)` produces a Fortran-order view (non-contiguous). Wrapping it in
`np.matrix()` raises `ValueError: ndarray is not contiguous` on numpy 2.x. `np.asarray(zz_all).T`
produces the same transposed view without invoking `np.matrix()`, and `.shape[0]` gives the same
result for the orientation check.

---

## Patch 11 — `MRF.py:_random_walk_regularisation` lines 1003–1004 (y_neighbors)

Before:
```python
y_neighbors = np.matrix(
    self.rw_regul*rw_regul_dat[everybody, 0])
```

After:
```python
y_neighbors = np.ravel(
    self.rw_regul*rw_regul_dat[everybody, 0])
```

**Why:** `rw_regul_dat[everybody, 0]` is 1-D ndarray (after Patch 8). `np.matrix(1-D)` = row
matrix `(1, n)`. The sole downstream use is `np.append(np.array(yy), y_neighbors)` which
flattens to 1-D. `np.ravel` produces 1-D directly, avoiding the round-trip through `np.matrix`.

---

## Patch 12 — `MRF.py:_random_walk_regularisation` lines 1005–1006 (z_neighbors)

Before:
```python
z_neighbors = np.matrix(self.rw_regul * np.column_stack(
    [np.repeat(1, repeats=len(everybody)), rw_regul_dat[everybody, 1: ncrd]]))
```

After:
```python
z_neighbors = np.asarray(self.rw_regul * np.column_stack(
    [np.repeat(1, repeats=len(everybody)), rw_regul_dat[everybody, 1: ncrd]]))
```

**Why:** `np.column_stack(...)` already returns a 2-D ndarray of shape `(len(everybody), ncrd)`.
`np.matrix(2-D ndarray)` wraps without reshaping. `np.asarray(2-D ndarray)` = same shape. Both
are valid inputs to `np.vstack` downstream.

---

## Patch 13 — `MRF.py:_random_walk_regularisation` lines 1012–1013 (y_neighbors2)

Before:
```python
y_neighbors2 = np.matrix(
    self.rw_regul ** 2 * rw_regul_dat[everybody2, 0])
```

After:
```python
y_neighbors2 = np.ravel(
    self.rw_regul ** 2 * rw_regul_dat[everybody2, 0])
```

**Why:** Same as Patch 11 applied to `everybody2`.

---

## Patch 14 — `MRF.py:_random_walk_regularisation` lines 1014–1016 (z_neighbors2 with nested matrix)

Before:
```python
z_neighbors2 = np.matrix(self.rw_regul ** 2*np.column_stack(
    [np.repeat(1, repeats=len(everybody2)),
     np.matrix(rw_regul_dat[everybody2, 1: ncrd])]))
```

After:
```python
z_neighbors2 = np.asarray(self.rw_regul ** 2 * np.column_stack(
    [np.repeat(1, repeats=len(everybody2)),
     np.asarray(rw_regul_dat[everybody2, 1: ncrd])]))
```

**Why:** Removes both the inner and outer `np.matrix(...)` calls. The inner
`np.matrix(rw_regul_dat[everybody2, 1:ncrd])` wraps a 2-D ndarray — `np.asarray` is equivalent.
`np.column_stack` accepts ndarrays. The outer `np.asarray` ensures the result is a plain ndarray
for `np.vstack`.

---

## Patch 15 — `MRF.py:plot_res` line 1124 (visualization data matrix)

Before:
```python
data = np.matrix(self.data)
```

After:
```python
data = np.ascontiguousarray(np.asarray(self.data))
```

**Why:** `self.data` is a pandas DataFrame that may be non-contiguous. The visualization function
`plot_res()` uses `data[:, z_pos]` and `data[:, y_pos]` for column indexing and `@`-operator
matmul — all equivalent for C-order ndarray.

---

## Patch 16 — `MRF.py:standard` line 1291 (standardisation keepdims)

Before:
```python
Y = np.matrix(Y)
size = Y.shape
mean_y = Y.mean(axis=0)
sd_y = Y.std(axis=0, ddof=1)
```

After:
```python
Y = np.atleast_2d(np.asarray(Y))
size = Y.shape
mean_y = Y.mean(axis=0, keepdims=True)
sd_y = Y.std(axis=0, ddof=1, keepdims=True)
```

**Why:** `std_stuff["std"][:, y_pos]` and `std_stuff["mean"][:, y_pos]` at lines 451–455 require
mean and std to be 2-D arrays of shape `(1, k)`. With `np.matrix`, `Y.mean(axis=0)` naturally
returns `(1, k)`. With ndarray, `Y.mean(axis=0)` returns `(k,)` — causing `IndexError` on
2-D column indexing. `keepdims=True` restores the `(1, k)` shape while producing identical
values. The `Y0` computation `(Y - np.repeat(mean_y, n, axis=0))` is unchanged.
