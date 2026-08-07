"""Per-origin |POOS-BIC| and |KF-BIC| from the G2 store, over time, for INDPRO h=1.
If they diverge at late origins, the pipeline CV is fine; if identical throughout, the
pipeline POOS/KF is systematically collapsing to BIC (unlike the isolated select_params)."""
import glob
import numpy as np
import pandas as pd

cells = glob.glob("runs/gcls_b4_stage1/_result_store_indpro/cells/*.parquet")
frames = []
for f in cells:
    df = pd.read_parquet(f, columns=["origin", "horizon", "arm", "prediction", "actual"])
    if df["horizon"].iloc[0] == 1:
        frames.append(df)
M = pd.concat(frames, ignore_index=True)
piv = M.pivot_table(index="origin", columns="arm", values="prediction", aggfunc="first").sort_index()
for base in ("AR", "ARDI"):
    d_poos = (piv[f"{base},POOS"] - piv[f"{base},BIC"]).abs()
    d_kf = (piv[f"{base},KF"] - piv[f"{base},BIC"]).abs()
    d_aic = (piv[f"{base},AIC"] - piv[f"{base},BIC"]).abs()
    nz_poos = (d_poos > 1e-10).sum()
    nz_kf = (d_kf > 1e-10).sum()
    print(f"{base}: origins={len(piv)}  POOS!=BIC at {nz_poos} origins (max|d|={d_poos.max():.2e})  "
          f"KF!=BIC at {nz_kf}  AIC!=BIC max={d_aic.max():.2e}")
    # show a few late-origin diffs
    late = piv.index[-6:]
    for o in late:
        print(f"   {str(o.date())}: POOS-BIC={(piv.loc[o, f'{base},POOS']-piv.loc[o, f'{base},BIC']):+.3e}  "
              f"AIC-BIC={(piv.loc[o, f'{base},AIC']-piv.loc[o, f'{base},BIC']):+.3e}")
print("OK perorigin")
