## String data analysis and modelling for `tenscalc` 

This repository contains a basic
analysis of data from string tension guides published by two major US string
manufacturers. This analysis informs the development of the
[tenscalc](http://github.com/davidelambert/tenscalc) Python package.

What follows is a slightly condensed display version of the code contained in
`analysis.py` and `analysis.ipynb`. `analysis.py` is the principal analysis script,
written in a format compatible with Jupyter notebook-friendly editors likw VSCode or
PyCharm Professional Edition (i.e. with pseudo-cells prepended by `# %%`).
`analysis.ipynb` is a traditional Jupyter notebook produced from `analysis.py`.

The version below contain static image scatterplots under the **Visualization** heading.
These plots are available as interactive [plotly](https://plotly.com/graphing-libraries/)
plots in `analysis.py` and `analysis.ipynb`.

## Setup
```python
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf

ROOT = Path.cwd()
d = pd.read_csv(ROOT/'data.csv')
```

## Mean unit weights
For gauge/material items with unit weight data from both GHS and D'Addario,
take the simple mean unit weight of both brands. Also include gauge/material
items produced by only one brand. Format these data as JSON, keyed by
material, to use as a unit weight lookup table when possible.


```python
uw_means = d.groupby(['gauge', 'material'], as_index=False).mean()
```


```python
means_only = uw_means.loc[uw_means['brand2'] == 0.5,
                          ['material', 'gauge', 'unit_weight']]
means_only = means_only.rename(columns={'unit_weight': 'uw_mean'})
```


```python
diffmeans = d.merge(means_only)
diffmeans['abs_diff'] = (diffmeans['unit_weight']
                         - diffmeans['uw_mean'])
diffmeans['prop_diff'] = (np.log(diffmeans['unit_weight'])
                          - np.log(diffmeans['uw_mean']))
```


```python
pp5 = len(diffmeans.loc[np.abs(diffmeans['prop_diff']) > 0.05]) // 2
pp10 = len(diffmeans.loc[np.abs(diffmeans['prop_diff']) > 0.1]) // 2
n_gauges = len(d['gauge'].unique())
print("Gauges with > 5 percent difference from GHS/D'Addario mean: {}/{}"
      .format(pp5, n_gauges))
print("Gauges with > 10 percent difference from GHS/D'Addario mean: {}/{}"
      .format(pp10, n_gauges))
```

```
Gauges with > 5 percent difference from GHS/D'Addario mean: 21/88
Gauges with > 10 percent difference from GHS/D'Addario mean: 4/88
```

```python
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
```

## Visualization

```python
mat_map = {
    'ps': 'Plain Steel',
    'nps': 'Nickel-Plated Steel Roundwound',
    'ss': 'Stainless Steel Roundwound',
    'pn': 'Pure Nickel Roundwound',
    'fw': 'Stainless Steel Flatwound',
    'pb': 'Phosphor Bronze Roundwound',
    '8020': '80/20 Bronze Roundwound',
    '8515': '85/15 Bronze Roundwound',
    'bnps': 'Nickel-Plated Steel Roundwound',
    'bss': 'Stainless Steel Roundwound',
    'bfw': 'Stainless Steel Flatwound'
}

d['gauge_sq'] = np.power(d['gauge'], 2)
d['brand'] = d['brand2'].map({0: 'G', 1: 'D'})
d['mat_long'] = d['material'].map(mat_map)
```


```python
p_ps = px.scatter(d[d['material'] == 'ps'], x='gauge_sq', y='unit_weight',
                  trendline='ols', title='Plain Steel Strings',
                  hover_data={'gauge': True, 'gauge_sq': False, 'brand': True},
                  width=600, height=400)
p_ps.show()
```
![plain steel strings scatterplot](/images/output_9_1.png)
```python
elec_mat = ['nps', 'ss', 'pn', 'fw', ]
elec_filter = [mat in elec_mat for mat in d['material']]
p_elec = px.scatter(d.loc[elec_filter], x='gauge_sq', y='unit_weight',
                    trendline='ols', title='Electric Wound Strings',
                    hover_data={'gauge': True, 'gauge_sq': False, 'brand': True},
                    width=600, height=400,
                    color='material', color_discrete_map={'nps': '#008080',
                                                          'ss': '#800080',
                                                          'pn': '#40E0D0',
                                                          'fw': '#EE82EE'})
p_elec.show()
```

![electric strings scatterplot](/images/output_10_0.png)

```python
acou_mat = ['pb', '8020', '8515', ]
acou_filter = [mat in acou_mat for mat in d['material']]
p_acou = px.scatter(d.loc[acou_filter], x='gauge_sq', y='unit_weight',
                    trendline='ols', title='Acoustic Wound Strings',
                    hover_data={'gauge': True, 'gauge_sq': False, 'brand': True},
                    width=600, height=400,
                    color='material', color_discrete_map={'pb': '#8B4513',
                                                          '8020': '#FFD700',
                                                          '8515': '#D2691E'})
p_acou.show()
```

![acoustic strings scatterplot](/images/output_11_0.png)

```python
bass_mat = ['bnps', 'bss', 'bfw', ]
bass_filter = [mat in bass_mat for mat in d['material']]
p_bass = px.scatter(d.loc[bass_filter], x='gauge_sq', y='unit_weight',
                    trendline='ols', title='Bass Strings',
                    hover_data={'gauge': True, 'gauge_sq': False, 'brand': True},
                    width=600, height=400,
                    color='material', color_discrete_map={'bnps': '#008080',
                                                          'bss': '#800080',
                                                          'bfw': '#EE82EE'})
p_bass.show()
```

![bass strings scatterplot](/images/output_12_0.png)

## OLS modelling
Build unit weight models of the form
<img src="https://render.githubusercontent.com/render/math?UW_{i,m} = \alpha_m + \beta_m * gauge_{i}^{2}">

Build OLS models for each string material separately for ease of use and
interpretation. Write constants and coefficients for each material to JSON for use
when unit weights are not available from the lookup table created in the **Mean unit
weights** section.


```python
def short_table(fitted_model, title=None):
    """Abbreviated single-variable OLS results table"""

    coef = round(fitted_model.params['gauge_sq'], ndigits=4)
    se = round(fitted_model.bse['gauge_sq'], ndigits=4)
    t = round(fitted_model.tvalues['gauge_sq'], ndigits=2)
    p = round(fitted_model.pvalues['gauge_sq'], ndigits=4)
    r2 = round(fitted_model.rsquared, ndigits=4)

    cw = 10
    if title is not None:
        title = f' {title} '.center(5 * cw, '=')
    else:
        title = '=' * (5 * cw)
    chead = ''.join([ch.rjust(cw) for ch in ['Coef.', 'SE', 't', 'Pr>|t|', 'R^2']])
    cval = ''.join([str(v).rjust(cw) for v in [coef, se, t, p, r2]])
    table = '{}\n{}\n{}'.format(title, chead, cval)

    return table
```


```python
mat_list = d['material'].unique()
tension_models = dict()
for mat in mat_list:
    mod = smf.ols('unit_weight ~ gauge_sq', data=d[d['material'] == mat]).fit()
    print(short_table(mod, title=mat_map[mat]), '\n')
    tension_models[mat] = {'const': mod.params.loc['Intercept'],
                           'coef': mod.params.loc['gauge_sq']}
```

    ================== Plain Steel ===================
         Coef.        SE         t    Pr>|t|       R^2
        0.2269    0.0027     84.05       0.0    0.9939 
    
    ========= Nickel-Plated Steel Roundwound =========
         Coef.        SE         t    Pr>|t|       R^2
        0.1926    0.0034     55.93       0.0      0.98 
    
    =========== Phosphor Bronze Roundwound ===========
         Coef.        SE         t    Pr>|t|       R^2
        0.1987    0.0013    158.85       0.0    0.9981 
    
    ============ 80/20 Bronze Roundwound =============
         Coef.        SE         t    Pr>|t|       R^2
         0.193    0.0011    170.88       0.0    0.9986 
    
    ============ 85/15 Bronze Roundwound =============
         Coef.        SE         t    Pr>|t|       R^2
        0.1985    0.0016    126.26       0.0    0.9979 
    
    =========== Stainless Steel Roundwound ===========
         Coef.        SE         t    Pr>|t|       R^2
        0.1876    0.0033     57.16       0.0      0.99 
    
    ============= Pure Nickel Roundwound =============
         Coef.        SE         t    Pr>|t|       R^2
        0.2155    0.0032      67.8       0.0    0.9961 
    
    =========== Stainless Steel Flatwound ============
         Coef.        SE         t    Pr>|t|       R^2
        0.1831    0.0034     53.21       0.0    0.9895 
    
    ========= Nickel-Plated Steel Roundwound =========
         Coef.        SE         t    Pr>|t|       R^2
        0.1805    0.0019     94.15       0.0    0.9943 
    
    =========== Stainless Steel Roundwound ===========
         Coef.        SE         t    Pr>|t|       R^2
        0.1691    0.0033      51.6       0.0    0.9867 
    
    =========== Stainless Steel Flatwound ============
         Coef.        SE         t    Pr>|t|       R^2
        0.1845    0.0019     98.31       0.0    0.9972 
    



```python
with open(ROOT/'tension_models.json', 'w') as f:
    json.dump(tension_models, f, indent=2)
```
