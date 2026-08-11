import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/tmp/mf-b4-kf'); sys.path.insert(0,'/tmp/mf-b4-kf/scripts/replication/gcls_2022_pipeline')
import data as D
bundle, preds = D.augmented_bundle()
p = bundle.panel if hasattr(bundle,'panel') else bundle
raw = p['INDPRO']; yobj = p['YOBJ__INDPRO']
d = pd.DataFrame({'raw': raw, 'yobj': yobj}).dropna()
print('YOBJ__INDPRO first 4:', d.yobj.head(4).round(6).tolist())
print('dlog(INDPRO) first 4 :', (np.log(d.raw) - np.log(d.raw.shift(1))).dropna().head(4).round(6).tolist())
print('log(INDPRO)  first 4 :', np.log(d.raw).head(4).round(4).tolist())
dlog = (np.log(d.raw) - np.log(d.raw.shift(1)))
print()
print('corr(YOBJ, dlog) = %.6f   <- 1.0 means YOBJ is the ONE-PERIOD OBJECT' % d.yobj.corr(dlog))
print('corr(YOBJ, log ) = %.6f   <- 1.0 would mean YOBJ is the LOG LEVEL' % d.yobj.corr(np.log(d.raw)))
