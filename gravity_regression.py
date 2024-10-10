# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS, compare
import matplotlib.pyplot as plt
from scipy import stats

# %%
gravity = pd.read_csv('~/Documents/python/data/gravity80.csv')
#gravity = gravity[gravity['year']>1990]
df = gravity
df['trade'] = df['tradeflow_comtrade_o'] + df['tradeflow_comtrade_d']
df['tradeflow'] = df['tradeflow_comtrade_o']
# df['rhsg'] = (df['gdp_o'] * df['gdp_d'])/ (df['dist'])**2
df['gatt'] = df['gatt_o'] * df['gatt_d']
df['wto'] = df['wto_o'] * df['wto_d']
df['eu'] = df['eu_o'] * df['eu_d']
df = df.drop(columns=['gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
# df = df.drop(columns=['dist', 'gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'gdp_o', 'gdp_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
df['log_trade'] = np.log(df['trade'])
df['log_tradeflow'] = np.log(df['tradeflow'])
df['log_gdp_o'] = np.log(df['gdp_o'])
df['log_gdp_d'] = np.log(df['gdp_d'])
df['log_dist'] = np.log(df['dist'])
df.head()

# %%
year = pd.Categorical(df['year'])
df = df.set_index(['iso3_o', 'year'])
df["year"] = year

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PooledOLS(df.log_tradeflow, exog)
pooled_res = mod.fit()
print(pooled_res)

# %%
mod = RandomEffects(df.log_tradeflow, exog)
re_res = mod.fit()
print(re_res)

# %%
re_res.variance_decomposition
# %%
re_res.theta.head
# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = BetweenOLS(df.log_tradeflow, exog)
be_res = mod.fit()
print(be_res)

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.log_tradeflow, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# %%
exog_vars = ['log_gdp_o', 'log_gdp_d', 'log_dist', 'wto', 'eu', 'fta_wto']
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.log_tradeflow, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)

# %%
print(compare({"FE": fe_res, "RE": re_res, "Pooled": pooled_res}))

# %%
# Perform Hausman test
# Extract coefficients and covariance matrices
b_fe = fe_res.params
b_re = re_res.params

# Covariance matrices
cov_fe = fe_res.cov
cov_re = re_res.cov

# Compute the difference between the coefficients
b_diff = b_fe - b_re

# Compute the covariance of the difference
cov_diff = cov_fe - cov_re

# Compute Hausman test statistic
hausman_statistic = np.dot(np.dot(b_diff.T, np.linalg.inv(cov_diff)), b_diff)

# Degrees of freedom (number of coefficients)
df = b_diff.shape[0]

# Calculate p-value from the chi-squared distribution
p_value = 1 - stats.chi2.cdf(hausman_statistic, df)

# Print results
print("Hausman test statistic:", hausman_statistic)
print("p-value:", p_value)

# Interpret the result
if p_value < 0.05:
    print("Reject null hypothesis: Fixed effects model is preferred.")
else:
    print("Fail to reject null hypothesis: Random effects model is preferred.")

# %%
# # Calculate fitted values manually
m = smf.ols("log_tradeflow ~ log_gdp_o", data=df).fit()
plt.scatter(df['log_gdp_o'], df['log_tradeflow'], color='blue', label='Data Points')
plt.plot(df['log_gdp_o'], m.fittedvalues, color="red", label="Regression Line")
plt.xlabel("Logarithm of GDP")
plt.ylabel("Logarithm of Trade Flow")
plt.title("Pooled OLS Model")
plt.legend()
plt.show()
