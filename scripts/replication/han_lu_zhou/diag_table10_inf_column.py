"""Which side owns the (panel A, control=INF) deviation?

That cell is degenerate: the target is inflation and the control IS inflation, so
the k=1 regression carries the same column twice. Port the authors' regression
faithfully in numpy and see whether it lands on the printed value or on ours.
"""
import numpy as np, scipy.io as sio, pandas as pd

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
m = sio.loadmat(f"{ARCH}/Data_macro_indicator.mat")
ECON = m["ECON"]; T = ECON.shape[0]; R = (1964 - 1947 + 1) * 4; P = T - R
LADDER = [1, 2, 4, 8, 12]

def fc_paths(tcol, z, solver):
    y = ECON[:, tcol]
    XMA = np.full((T, len(LADDER)), np.nan)
    for j, L in enumerate(LADDER):
        for t in range(L - 1, T):
            XMA[t, j] = y[t] if L == 1 else np.nanmean(y[t - L + 1 : t + 1])
    actual = np.full(P, np.nan); FC = np.full((P, len(LADDER)), np.nan)
    for t in range(P):
        actual[t] = y[R + t]
        n = R + t
        yy = y[1:n]
        for j in range(len(LADDER)):
            Xd = np.column_stack([np.ones(n - 1), XMA[: n - 1, j], z[: n - 1]])
            ok = np.isfinite(Xd).all(1) & np.isfinite(yy)
            b = solver(Xd[ok], yy[ok])
            FC[t, j] = np.array([1.0, XMA[n - 1, j], z[n - 1]]) @ b
    return actual, FC

def lstsq(X, yy):                       # numpy min-norm (what sklearn does)
    return np.linalg.lstsq(X, yy, rcond=None)[0]

def basic(X, yy):                       # MATLAB regress: zero the redundant column
    q, r, piv = __import__("scipy.linalg", fromlist=["qr"]).qr(X, mode="economic", pivoting=True)
    tol = max(X.shape) * np.finfo(float).eps * abs(r[0, 0])
    k = int((np.abs(np.diag(r)) > tol).sum())
    b = np.zeros(X.shape[1])
    b[piv[:k]] = np.linalg.solve(r[:k, :k], (q[:, :k].T @ yy))
    return b

def r2(actual, fbench, f):
    return 100.0 * (1.0 - np.sum((actual - f) ** 2) / np.sum((actual - fbench) ** 2))

PRINTED_A_INF = [1.04, 7.94, 7.34, 8.39]
SETS = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
for label, solver in (("min-norm (lstsq)", lstsq), ("basic soln (MATLAB regress)", basic)):
    a, FC = fc_paths(0, m["INF"].ravel(), solver)
    bench = FC[:, 0]
    vals = [r2(a, bench, FC[:, s].mean(1)) for s in SETS]
    print(f"{label:30s} " + "  ".join(f"{v:6.3f}" for v in vals))
print(f"{'printed':30s} " + "  ".join(f"{v:6.3f}" for v in PRINTED_A_INF))
print(f"{'macroforecast (ours)':30s}  1.210   8.120   7.560   8.610")

# sanity: a NON-degenerate column must agree with print under either solver
print()
a, FC = fc_paths(0, m["UNE"].ravel(), lstsq)
print("control=UNE (non-degenerate), min-norm: " +
      "  ".join(f"{r2(a, FC[:,0], FC[:,s].mean(1)):6.3f}" for s in SETS) +
      "   printed:  2.230  10.100  10.340  12.140")
