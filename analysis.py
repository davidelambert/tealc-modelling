# %%
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.api import OLS
import plotly

ROOT = Path(__file__).parent.resolve()
d = pd.read_csv(ROOT/'data.csv')

# %%
uw_means = d.groupby(['gauge', 'material'], as_index=False).mean()

# %%
means_only = uw_means.loc[uw_means['brand2'] == 0.5,
                          ['material', 'gauge', 'unit_weight']]
means_only = means_only.rename(columns={'unit_weight': 'uw_mean'})

# %%
diffmeans = d.merge(means_only)
diffmeans['abs_diff'] = (diffmeans['unit_weight']
                         - diffmeans['uw_mean'])
diffmeans['prop_diff'] = (np.log(diffmeans['unit_weight'])
                          - np.log(diffmeans['uw_mean']))

# %%
pp5 = len(diffmeans.loc[np.abs(diffmeans['prop_diff']) > 0.05]) // 2
pp10 = len(diffmeans.loc[np.abs(diffmeans['prop_diff']) > 0.1]) // 2
n_gauges = len(d['gauge'].unique())
print("Gauges with > 5 percent difference from GHS/D'Addario mean: {}/{}"
      .format(pp5, n_gauges))
print("Gauges with > 10 percent difference from GHS/D'Addario mean: {}/{}"
      .format(pp10, n_gauges))


# %%
output = uw_means.loc[:, ['material', 'gauge', 'unit_weight']]
materials = list(output['material'].unique())
if Path(ROOT/'unit_weights.json').exists():
    Path(ROOT/'unit_weights.json').unlink()
with open(ROOT/'unit_weights.json', 'a') as f:
    f.write('{')
    for m in materials:
        key = (output.loc[output['material'] == m, ['gauge', 'unit_weight']]
               .rename(columns={'unit_weight': m})
               .set_index('gauge')
               .to_json(indent=2))
        if m != materials[-1]:
            f.write(key[1:-2] + ',\n')
        else:
            f.write(key[1:-1])
    f.write('}')
# %%
