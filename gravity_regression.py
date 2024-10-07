# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS, compare
import matplotlib.pyplot as plt

# %%
gravity = pd.read_csv('~/Documents/python/data/gravity80.csv')
#gravity = gravity[gravity['year']>1990]
df = gravity
df['trade'] = df['tradeflow_comtrade_o'] + df['tradeflow_comtrade_d']
# df['rhsg'] = (df['gdp_o'] * df['gdp_d'])/ (df['dist'])**2
df['gatt'] = df['gatt_o'] * df['gatt_d']
df['wto'] = df['wto_o'] * df['wto_d']
df['eu'] = df['eu_o'] * df['eu_d']
df = df.drop(columns=['gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
# df = df.drop(columns=['dist', 'gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'gdp_o', 'gdp_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
df['log_trade'] = np.log(df['trade'])
df['log_gdp_o'] = np.log(df['gdp_o'])
df['log_gdp_d'] = np.log(df['gdp_d'])
df['log_dist'] = np.log(df['dist'])
df.head()

# %%
year = pd.Categorical(df['year'])
df = df.set_index(['iso3_o', 'year'])
df["year"] = year

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'gatt', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PooledOLS(df.log_trade, exog)
pooled_res = mod.fit()
print(pooled_res)

# %%
mod = RandomEffects(df.log_trade, exog)
re_res = mod.fit()
print(re_res)

# %%
re_res.variance_decomposition
# %%
re_res.theta.head
# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'gatt', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = BetweenOLS(df.log_trade, exog)
be_res = mod.fit()
print(be_res)

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'gatt', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.log_trade, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'gatt', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.log_trade, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)

# %%
print(compare({"BE": be_res, "RE": re_res, "Pooled": pooled_res}))
