"""B5 / Table 5 — neural networks NN1-NN5.

Structure from the paper and Internet Appendix:
  NN1..NN5 are one to five hidden layers, (2), (4,2), (8,4,2), (16,8,4,2), (32,16,8,4,2),
  ReLU activations, Adam; hyper-parameters otherwise follow Gu, Kelly & Xiu (2020)
  (IA §IA1.3). The reported forecast averages the FIVE BEST of many random seeds, ranked
  by R2_OS over a validation period.

Unlike Tables 2-4 the appendix does not say the predictors are grouped -- the networks take
the whole 14 x L trend block at once, which is the point of the exercise ("networks excel
when handling a large number of trend signals and their complex nonlinear interactions").
So each cell is ONE arm over all signals, not a pooled set.

TRAINING PROTOCOL. The paper gives architectures, ReLU, Adam and the "average the five
best seeds in validation" rule, and nothing else -- no epochs, batch size, learning rate,
regularization, early stopping, seed count or validation length appears anywhere in the
article or its Internet Appendix. Table 5 is therefore not identified by its own
specification. Rather than tune toward the printed values, the gaps are filled from Gu,
Kelly & Xiu (2020), whom the paper cites (NBER w25398, hyperparameter table, NN1-NN5
column): Epochs = 100, Patience = 5, LR in {0.001, 0.01}, Adam defaults, Ensemble = 10.

`[ASSUMPTION]` GKX set Batch Size = 10000 for a panel of millions of stock-months; this
sample has 500-1100 training rows, so that value would make every batch the full sample.
128 is used instead and stated here.
`[ASSUMPTION]` GKX do not state the validation split for early stopping; a 20% tail of
the fit window is used, split by time rather than at random.
`[ASSUMPTION]` The paper says "five best of many seeds" without giving the count; GKX's
Ensemble = 10 is used, so ten seeds are fit and the best five averaged.
`[ASSUMPTION]` Inputs are standardized -- unstandardized trend levels span orders of
magnitude and a ReLU network on raw scales is not what any of the cited work does.
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.models import get_model

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DESIGNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L1", "L6", "L12"]
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10   # GKX: Ensemble = 10
TOP_K = 5                 # HLZ: average the five best in validation
VAL_MONTHS = 120          # tail of the fit window used to rank seeds
MAX_EPOCHS = 100          # GKX: Epochs = 100
PATIENCE = 5              # GKX: Patience = 5
LEARNING_RATE = 0.001     # GKX: LR in {0.001, 0.01}; the lower end
VAL_FRACTION = 0.2        # [ASSUMPTION] GKX does not state the split
BATCH_SIZE = 128          # [ASSUMPTION] GKX uses 10000, which exceeds this sample

ARCH = {"NN1": (2,), "NN2": (4, 2), "NN3": (8, 4, 2),
        "NN4": (16, 8, 4, 2), "NN5": (32, 16, 8, 4, 2)}
# printed Table 5 (RAPS 16(2), p. 259)
PAPER = {
    "L1":  {"NN1": -0.11, "NN2": 0.03, "NN3": 0.41, "NN4": -2.07, "NN5": -1.69},
    "L6":  {"NN1": 0.67, "NN2": 0.23, "NN3": 1.77, "NN4": 0.41, "NN5": -0.41},
    "L12": {"NN1": 1.23, "NN2": 1.52, "NN3": 0.78, "NN4": 1.28, "NN5": 0.01},
}
ARCH_COL = {"L1": 0, "L6": 1, "L12": 2, "L24": 3, "L36": 4}
LMAP = {"L1": 1, "L6": 6, "L12": 12, "L24": 24, "L36": 36}

import scipy.io as sio
_d = sio.loadmat("/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data/Data_trend.mat")
ARCHIVE = {k: np.asarray(_d[f"FC_{k}"], float) for k in ARCH}
ACTUAL = _d["actual"].ravel()
HA = _d["FC_HA"].ravel()

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index
R = 456                                    # first origin -> first target 1965-01
poos = idx[(idx >= idx[R + 1]) & (idx <= pd.Timestamp("2022-12-31"))]
nn = get_model("nn")


from joblib import Parallel, delayed


def seed_ensemble_forecast(Xtr, ytr, Xte, arch):
    """Fit N_SEEDS networks, rank on a held-out tail, average the best TOP_K."""
    n = len(Xtr)
    cut = max(n - VAL_MONTHS, int(0.6 * n))
    Xf, yf = Xtr.iloc[:cut], ytr.iloc[:cut]
    Xv, yv = Xtr.iloc[cut:], ytr.iloc[cut:]
    scored = []
    for s in range(N_SEEDS):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = nn(Xf, yf, hidden_layer_sizes=arch, activation="relu",
                         optimizer="adam", max_epochs=MAX_EPOCHS,
                         learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                         validation_fraction=VAL_FRACTION,
                         early_stopping_patience=PATIENCE, random_state=s)
                pv = np.asarray(fit.predict(Xv), dtype=float).ravel()
                mse = float(np.mean((pv - yv.to_numpy(float)) ** 2))
                # refit on the full training window with the same seed
                fit_full = nn(Xtr, ytr, hidden_layer_sizes=arch, activation="relu",
                              optimizer="adam", max_epochs=MAX_EPOCHS,
                              learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                              validation_fraction=VAL_FRACTION,
                              early_stopping_patience=PATIENCE, random_state=s)
                pt = np.asarray(fit_full.predict(Xte), dtype=float).ravel()
            scored.append((mse, pt))
        except Exception:
            continue
    if not scored:
        return np.full(len(Xte), np.nan)
    scored.sort(key=lambda t: t[0])
    best = [p for _, p in scored[:TOP_K]]
    return np.mean(np.column_stack(best), axis=1)


def r2_os(actual, ha, fc):
    ok = np.isfinite(fc) & np.isfinite(actual) & np.isfinite(ha)
    return 100.0 * (1.0 - np.mean((actual[ok] - fc[ok]) ** 2) / np.mean((actual[ok] - ha[ok]) ** 2))


results = {}
for design in DESIGNS:
    L = LMAP[design]
    cols = {f"{v}_MA{l}": econ[v].rolling(l, min_periods=l).mean()
            for l in range(1, L + 1) for v in NAMES}
    X = pd.DataFrame(cols, index=idx)
    # standardize on the training window at each origin (done inside the loop below)
    tgt = y.shift(-1)
    print(f"\n########## design={design} (L={L})  signals={X.shape[1]}  "
          f"seeds={N_SEEDS} top{TOP_K} ##########", flush=True)
    for name, arch in ARCH.items():
        t0 = time.time()

        def one_origin(d, arch=arch):
            i = idx.get_loc(d)
            # tgt[s] is y[s+1], so a training row indexed s carries the pair
            # (X[s], y[s+1]). The forecast uses X[i-1] to predict y[i], so s must stop
            # at i-2: including s = i-1 would put the very observation being predicted
            # into the training set. The authors' loop ends at the same place --
            # regress y_t(2:end) on X_t(1:end-1).
            tr = idx[: max(i - 1, 0)]
            ok = X.loc[tr].notna().all(axis=1) & tgt.reindex(tr).notna()
            tr = tr[ok.to_numpy()]
            if len(tr) < 200:
                return np.nan
            Xtr, ytr = X.loc[tr], tgt.loc[tr]
            mu, sd = Xtr.mean(), Xtr.std(ddof=0).replace(0.0, 1.0)
            Xte = X.iloc[[i - 1]]
            p = seed_ensemble_forecast((Xtr - mu) / sd, ytr, (Xte - mu) / sd, arch)
            return float(p[0])

        preds = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(one_origin)(d) for d in poos
        )
        fc = np.asarray(preds, dtype=float)
        n = min(len(fc), len(ACTUAL))
        mine = r2_os(ACTUAL[-n:], HA[-n:], fc[-n:])
        pap = PAPER[design][name]
        ref = ARCHIVE[name][:, ARCH_COL[design]]
        md = float(np.nanmax(np.abs(fc[-n:] - ref[-n:])))
        results[(design, name)] = (mine, pap, md)
        print(f"  {name}  mine={mine:7.3f}%  paper={pap:6.2f}%  Δ={mine-pap:+7.3f}pp   "
              f"path max|Δ|={md:.3e}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        np.save(f"/tmp/hlz_t5_{design}_{name}.npy", fc)

print("\n########## TABLE 5 — neural networks ##########")
print("| design | model | mine % | paper % | Δ pp | path max|Δ| |")
print("|---|---|---|---|---|---|")
for (design, name), (mine, pap, md) in results.items():
    print(f"| {design} | {name} | {mine:.3f} | {pap:.2f} | {mine-pap:+.3f} | {md:.2e} |")
print("T5_DONE")
