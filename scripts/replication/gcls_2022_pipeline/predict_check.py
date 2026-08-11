import pandas as pd
c = pd.read_csv('/tmp/rest4_parity_cells.csv')
avg = c[c.target != 'T10YFFM']    # the four direct_average targets
lvl = c[c.target == 'T10YFFM']    # the one direct (level) target -- immune to the defect
for name, d in (('direct_average (4 targets)', avg), ('direct / level (T10YFFM)', lvl)):
    print('===', name)
    g = d.groupby('horizon').apply(lambda x: pd.Series({
        'mine_dist_from_1': (x.rel_rmspe_mine - 1).abs().median(),
        'paper_dist_from_1': (x.rel_rmspe_paper - 1).abs().median(),
        'mine_beats_%': 100*(x.rel_rmspe_mine < 1).mean(),
        'paper_beats_%': 100*(x.rel_rmspe_paper < 1).mean(),
    }), include_groups=False)
    print(g.round(3).to_string()); print()
