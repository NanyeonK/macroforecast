"""Which signal set actually produced each archived Table-2 column?

macroforecast reproduces the design as written to 1e-16, so the remaining ambiguity
is the mapping between the authors' script and their archived forecast paths. Build
every plausible design and cross-match against the five archived columns.
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y = panel[NAMES], panel["mkt_excess"].to_numpy(float)
R, T = 457, len(panel)
P = T - R

def stack(cols):
    return np.column_stack(cols)

def ma_design(windows):
    return stack([econ[v].rolling(L, min_periods=L).mean().to_numpy(float)
                  for L in windows for v in NAMES])

def lag_design(nblocks):
    return stack([econ[v].shift(b).to_numpy(float) for b in range(nblocks) for v in NAMES])

def authors_loop(X):
    fc = np.full(P, np.nan)
    for t in range(1, P + 1):
        n = R + t - 1
        yy, XX, last = y[:n][1:], X[:n][:-1, :], X[:n][-1, :]
        preds = np.empty(X.shape[1])
        for k in range(X.shape[1]):
            xk = XX[:, k]
            ok = np.isfinite(xk) & np.isfinite(yy)
            A = np.column_stack([np.ones(ok.sum()), xk[ok]])
            b, *_ = np.linalg.lstsq(A, yy[ok], rcond=None)
            preds[k] = b[0] + b[1] * last[k]
        fc[t - 1] = preds.mean()
    return fc

CANDIDATES = {
    "lag x1 (current, 14)": lag_design(1),
    "lag x3 (42)":          lag_design(3),
    "lag x6 (84)":          lag_design(6),
    "lag x12 (168)":        lag_design(12),
    "MA{1,3} (28)":         ma_design((1, 3)),
    "MA{1,3,6} (42)":       ma_design((1, 3, 6)),
    "MA{1,3,6,12} (56)":    ma_design((1, 3, 6, 12)),
    "MA{3,6} (28)":         ma_design((3, 6)),
    "MA{1..6} (84)":        ma_design((1, 2, 3, 4, 5, 6)),
    "MA{1..12} (168)":      ma_design((1,2,3,4,5,6,7,8,9,10,11,12)),
    "MA{1..24} (336)":      ma_design(tuple(range(1, 25))),
    "MA{1..36} (504)":      ma_design(tuple(range(1, 37))),
}
tgt = np.load("/tmp/hlz_target_table2.npz", allow_pickle=True)
ref = tgt["fc"]
labels = ["col0 (Current value)", "col1 ('3 lags')", "col2 ('MA6')",
          "col3 ('6 lags')", "col4 ('MA12')"]

paths = {}
for name, X in CANDIDATES.items():
    paths[name] = authors_loop(X)
    print(f"computed {name}", flush=True)

print("\n=== max|Δ| between each candidate design and each archived column ===")
print("| candidate | " + " | ".join(labels) + " |")
print("|---|" + "---|" * 5)
for name, p in paths.items():
    cells = []
    for j in range(5):
        cells.append(f"{np.nanmax(np.abs(p - ref[:, j])):.2e}")
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\n=== best match per archived column ===")
for j, lab in enumerate(labels):
    best = min(paths, key=lambda n: np.nanmax(np.abs(paths[n] - ref[:, j])))
    err = np.nanmax(np.abs(paths[best] - ref[:, j]))
    verdict = "EXACT" if err < 1e-10 else f"closest ({err:.2e})"
    print(f"  {lab}: {best}  -> {verdict}")
