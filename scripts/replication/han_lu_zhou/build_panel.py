"""B5 step 1: turn the Han-Lu-Zhou author archive into a clean monthly panel.

Source: Harvard Dataverse doi:10.7910/DVN/7APSCU (CC0), Codes/Data/WGpredictors2022.mat,
verified MD5 b490831db8e9d266ce5b50eeeb92ed7e. Using the authors' own constructed panel
removes data-reconstruction ambiguity, so the replication tests macroforecast rather than
my data build.
"""
import scipy.io as sio
import numpy as np, pandas as pd

D = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
m = sio.loadmat(f"{D}/WGpredictors2022.mat")

names = [str(x[0]) for x in m["varnames"].ravel()]
raw_date = m["date"].ravel().astype(int)
print("varnames:", names)
print("N_ECON:", int(m["N_ECON"].ravel()[0]))
print("date head/tail:", raw_date[:3], raw_date[-3:])

# date is YYYYMM
idx = pd.PeriodIndex([f"{d//100}-{d%100:02d}" for d in raw_date], freq="M").to_timestamp("M")
idx.name = "date"
print("date range:", idx.min().date(), "->", idx.max().date(), "| n =", len(idx))

econ = pd.DataFrame(m["ECON"], index=idx, columns=names)
y = pd.Series(m["y"].ravel(), index=idx, name="mkt_excess")
rf = pd.Series(m["r_f"].ravel(), index=idx, name="r_f")
ncf = pd.Series(m["ncf_full"].ravel(), index=idx, name="ncf_growth")

panel = pd.concat([y, econ, rf, ncf], axis=1)
print("\npanel:", panel.shape)
print("NaN per column (nonzero only):")
nz = panel.isna().sum()
print(nz[nz > 0].to_string() if (nz > 0).any() else "  none")

# X_ECON: the authors' dense MA ladder, 14 vars x 36 windows
X = np.asarray(m["X_ECON"], dtype=float)
print(f"\nX_ECON: {X.shape}  (= {len(names)} vars x {X.shape[1] // len(names)} windows)")

# Verify the ladder layout against a moving average we compute ourselves, so the
# column ordering is established by evidence rather than assumed.
best = None
for layout in ("var_major", "win_major"):
    errs = []
    for j, v in enumerate(names):
        for L in (1, 3, 6, 12):
            col = j * (X.shape[1] // len(names)) + (L - 1) if layout == "var_major" \
                else (L - 1) * len(names) + j
            mine = econ[v].rolling(L, min_periods=L).mean()
            a, b = X[:, col], mine.to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 50:
                errs.append(np.max(np.abs(a[ok] - b[ok])) / (np.std(b[ok]) + 1e-12))
    score = float(np.median(errs)) if errs else np.inf
    print(f"  layout {layout}: median relative max-err = {score:.3e}")
    if best is None or score < best[1]:
        best = (layout, score)
print(f"  -> X_ECON layout = {best[0]} (MA windows 1..{X.shape[1] // len(names)})")

panel.to_parquet("/tmp/hlz_panel.parquet")
np.save("/tmp/hlz_X_ECON.npy", X)
print("\nwrote /tmp/hlz_panel.parquet and /tmp/hlz_X_ECON.npy")
print(panel.head(3).to_string())
