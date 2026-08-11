import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/tmp/mf-b4-kf')
from macroforecast.feature_engineering.targets import average_target

idx = pd.date_range('2000-01-31', periods=80, freq='ME')
rng = np.random.default_rng(0)
logY = pd.Series(np.cumsum(rng.normal(0.01, 0.02, 80)), index=idx)   # log level
obj  = logY.diff()                                                    # one-period object = dlog Y

for h in (1, 3, 9, 24):
    got = average_target(pd.DataFrame({'y': obj}), target='y', horizon=h).iloc[:, 0]
    paper = pd.concat([obj.shift(-k) for k in range(1, h+1)], axis=1).mean(axis=1)   # (1/h) sum obj_{t+h'}
    alt   = (logY.shift(-h) - logY) / h                                              # same thing from the LOG LEVEL
    c = got.dropna().index.intersection(paper.dropna().index)
    print('h=%-3d  corr(got, paper)=%+.4f   max|got-paper|=%.4f   |  corr(paper, from-log-level)=%+.6f' % (
        h, got.loc[c].corr(paper.loc[c]), (got.loc[c]-paper.loc[c]).abs().max(),
        paper.dropna().corr(alt.reindex(paper.dropna().index))))
