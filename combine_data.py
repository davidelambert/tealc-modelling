from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.resolve()
charts = ROOT/'charts'

g = pd.read_csv('{}/{}.csv'.format(str(charts), 'ps'))
g['material'] = 'ps'
g['wound'] = 0
g['brand2'] = 0

d = pd.read_csv('{}/{}_D.csv'.format(str(charts), 'ps'))
d['material'] = 'ps'
d['wound'] = 0
d['brand2'] = 1

data = pd.concat([g, d])

for mat in ['nps', 'pb', '8020', '8515', 'ss', 'pn', 'fw',
            'bnps', 'bss', 'bfw']:
    g = pd.read_csv('{}/{}.csv'.format(str(charts), mat))
    g['material'] = mat
    g['wound'] = 1
    g['brand2'] = 0

    if Path(charts/'{}_D.csv'.format(mat)).exists():
        d = pd.read_csv('{}/{}_D.csv'.format(str(charts), mat))
        d['material'] = mat
        d['wound'] = 1
        d['brand2'] = 1
        data = pd.concat([data, g, d])
    else:
        data = pd.concat([data, g])

data.to_csv(ROOT/'data.csv', index=False)
