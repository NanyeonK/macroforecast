import pandas as pd
c = pd.read_csv('/tmp/rest4_parity_cells.csv')
print('cells:', len(c))
c['mine_beats'] = c.rel_rmspe_mine < 1.0
c['paper_beats'] = c.rel_rmspe_paper < 1.0
g = c.groupby('target').agg(
    mine=('mine_beats','sum'), paper=('paper_beats','sum'), n=('mine_beats','size'),
    med_mine=('rel_rmspe_mine','median'), med_paper=('rel_rmspe_paper','median'))
g['agree'] = c.groupby('target').apply(lambda d: int((d.mine_beats == d.paper_beats).sum()), include_groups=False)
print()
print('=== beats AR,BIC (rel-RMSPE < 1): mine vs paper ===')
print(g.to_string())
print()
tot = c.mine_beats.sum(); totp = c.paper_beats.sum()
print('ALL: mine %d/%d  paper %d/%d  same verdict on %d cells (%.1f%%)' % (
    tot, len(c), totp, len(c), (c.mine_beats==c.paper_beats).sum(),
    100*(c.mine_beats==c.paper_beats).mean()))
print()
print('=== best arm per target (mine vs paper) ===')
for t, d in c.groupby('target'):
    bm = d.loc[d.rel_rmspe_mine.idxmin()]; bp = d.loc[d.rel_rmspe_paper.idxmin()]
    print('%-9s mine: %-18s h=%-2d %.3f   |  paper: %-18s h=%-2d %.3f' % (
        t, bm.arm, bm.horizon, bm.rel_rmspe_mine, bp.arm, bp.horizon, bp.rel_rmspe_paper))
print()
print('=== median |delta| by horizon ===')
print(c.groupby('horizon')['abs_delta'].agg(['median','count']).to_string())
