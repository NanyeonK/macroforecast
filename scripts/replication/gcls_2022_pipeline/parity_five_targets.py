import math, sys
import pandas as pd
sys.path.insert(0, '/tmp/mf-b4-kf')
sys.path.insert(0, '/tmp/mf-b4-kf/scripts/replication/gcls_2022_pipeline')

RUN = '/home/nanyeon99/project/mf-b4-gcls2022/runs/gcls_b4_stage1'

# --- arm -> family, from the registry's own tags (not reconstructed from names) ---
import registry
arms = registry.build_gcls2022_arms()
if arms is None:
    for name in dir(registry):
        obj = getattr(registry, name)
        if callable(obj) and 'arm' in name.lower():
            print('candidate builder:', name)
    raise SystemExit('no build_arms()')

def family(tags):
    if tags['LF'] == 'eps':
        return 'SVR'
    if tags['NL'] == 1 and tags['SH'] == 'ridge':
        return 'KRR'
    if tags['NL'] == 1:
        return 'RF'
    if tags['SH'] != 'none':
        return 'Shrinkage'
    return 'Linear' if tags['CV'] in ('bic', 'aic') else 'CV-selected'

fam = {a.name: family(a.tags) for a in arms}
print('arms:', len(fam))
print(pd.Series(list(fam.values())).value_counts().to_string())

# --- mine: rel_rmspe = sqrt(relative_mse), the same convention g2_finalize uses ---
def load_mine(path):
    d = pd.read_csv(path)
    d = d[~d['is_benchmark']].copy()
    d['target'] = d['target'].str.replace('YOBJ__', '', regex=False)
    d['rel_rmspe_mine'] = d['relative_mse'].apply(lambda v: math.sqrt(v) if pd.notna(v) and v >= 0 else float('nan'))
    return d[['target', 'horizon', 'contender', 'rel_rmspe_mine']].rename(columns={'contender': 'arm'})

# INDPRO comes from the SEEDED re-run (kfseed, 2026-08-09), not from xlag0_full
# (2026-08-07), which predates the #515 k-fold seed fix that landed 2026-08-08 06:24.
# Mixing a pre-seed target with four post-seed targets in one table would compare
# runs that differ in more than the target.
mine = pd.concat([load_mine(f'{RUN}/rest4_accuracy.csv'), load_mine(f'{RUN}/kfseed_accuracy.csv')])
mine_unseeded_indpro = load_mine(f'{RUN}/xlag0_full_accuracy.csv')
print('\nmine cells:', len(mine), '| targets:', sorted(mine.target.unique()))

gold = pd.read_csv(f'{RUN}/gcls_tableA1_4tgt_gold.csv')
gold5 = pd.read_csv(f'{RUN}/gcls_tableA1_indpro_gold.csv')
assert 'target' not in gold5.columns  # INDPRO gold leaves the target implicit
gold5['target'] = 'INDPRO'
gold = pd.concat([gold, gold5])
gold = gold[gold['sample'] == 'full'][['target', 'arm', 'horizon', 'rel_rmspe']].rename(columns={'rel_rmspe': 'rel_rmspe_paper'})
print('gold rows (full sample):', len(gold), '| targets:', sorted(gold.target.unique()))

j = mine.merge(gold, on=['target', 'arm', 'horizon'], how='inner')
print('matched cells:', len(j))
unmatched = len(mine) - len(j)
if unmatched:
    miss = mine.merge(gold, on=['target','arm','horizon'], how='left')
    miss = miss[miss.rel_rmspe_paper.isna()]
    print('UNMATCHED', unmatched, '->', sorted(miss.arm.unique())[:6])

j['delta'] = j['rel_rmspe_mine'] - j['rel_rmspe_paper']
j['abs_delta'] = j['delta'].abs()
j['family'] = j['arm'].map(fam)

print('\n=== 1c: median |delta| by TARGET ===')
t = j.groupby('target')['abs_delta'].agg(['median', 'count']).sort_values('median')
print(t.to_string())
print('\n=== 1b: median |delta| by FAMILY ===')
f = j.groupby('family')['abs_delta'].agg(['median', 'count']).sort_values('median')
print(f.to_string())
print('\nALL contenders: median |delta| = %.4f over %d cells' % (j['abs_delta'].median(), len(j)))
j.to_csv('/tmp/rest4_parity_cells.csv', index=False)

u = mine_unseeded_indpro.merge(gold, on=['target', 'arm', 'horizon'], how='inner')
u['abs_delta'] = (u['rel_rmspe_mine'] - u['rel_rmspe_paper']).abs()
s = j[j.target == 'INDPRO']
print()
print('=== INDPRO: seeded vs unseeded, same cells ===')
print('unseeded (xlag0_full, 08-07): median |delta| = %.4f over %d cells' % (u.abs_delta.median(), len(u)))
print('seeded   (kfseed,     08-09): median |delta| = %.4f over %d cells' % (s.abs_delta.median(), len(s)))
m = s.merge(u[['target', 'arm', 'horizon', 'abs_delta']], on=['target', 'arm', 'horizon'], suffixes=('_seed', '_unseed'))
print('closer under seeding: %d | farther: %d | unchanged: %d' % (
    int((m.abs_delta_seed < m.abs_delta_unseed).sum()),
    int((m.abs_delta_seed > m.abs_delta_unseed).sum()),
    int((m.abs_delta_seed == m.abs_delta_unseed).sum())))
