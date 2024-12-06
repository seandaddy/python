# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS, compare
import matplotlib.pyplot as plt
from scipy import stats

# %%
gravity = pd.read_csv("~/Documents/python/data/gravity80.csv")
# gravity = gravity[gravity['year']>1990]
df = gravity
df = df.assign(
    trade=df["tradeflow_comtrade_o"] + df["tradeflow_comtrade_d"],
    tradeflow=df["tradeflow_comtrade_o"],
    gatt=df["gatt_o"] * df["gatt_d"],
    wto=df["wto_o"] * df["wto_d"],
    eu=df["eu_o"] * df["eu_d"],
)

# Apply logarithms in one go using np.log for multiple columns
df[["log_trade", "log_tradeflow", "log_gdp_o", "log_gdp_d", "log_dist"]] = np.log(
    df[["trade", "tradeflow", "gdp_o", "gdp_d", "dist"]]
)

df = df.drop(
    columns=[
        "gatt_o",
        "gatt_d",
        "wto_o",
        "wto_d",
        "eu_o",
        "eu_d",
        "tradeflow_comtrade_o",
        "tradeflow_comtrade_d",
    ]
)

# %%
year = pd.Categorical(df["year"])
df = df.set_index(["iso3_o", "year"])
df["year"] = year

# %%
exog_vars = ["log_gdp_o", "log_gdp_d", "log_dist", "wto", "eu", "fta_wto"]
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
exog_vars = ["log_gdp_o", "log_gdp_d", "log_dist", "wto", "eu", "fta_wto"]
exog = sm.add_constant(df[exog_vars])
mod = BetweenOLS(df.log_tradeflow, exog)
be_res = mod.fit()
print(be_res)

# %%
exog_vars = ["log_gdp_o", "log_gdp_d", "log_dist", "wto", "eu", "fta_wto"]
exog = sm.add_constant(df[exog_vars])
mod = PanelOLS(df.log_tradeflow, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# %%
exog_vars = ["log_gdp_o", "log_gdp_d", "log_dist", "wto", "eu", "fta_wto"]
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

gravity = pd.read_csv("~/Documents/python/data/gravity80.csv")
# gravity = gravity[gravity['year']>1990]
df = gravity
df = df.assign(
    trade=df["tradeflow_comtrade_o"] + df["tradeflow_comtrade_d"],
    tradeflow=df["tradeflow_comtrade_o"],
    gatt=df["gatt_o"] * df["gatt_d"],
    wto=df["wto_o"] * df["wto_d"],
    eu=df["eu_o"] * df["eu_d"],
)

# Apply logarithms in one go using np.log for multiple columns
df[["log_trade", "log_tradeflow", "log_gdp_o", "log_gdp_d", "log_dist"]] = np.log(
    df[["trade", "tradeflow", "gdp_o", "gdp_d", "dist"]]
)

model_ols = smf.ols("log_tradeflow ~ log_gdp_o", data=df).fit(
    cov_type="HC3"
)  # HC3 gives robust standard errors
print(model_ols.summary())

# 2. OLS with country fixed effects and clustered standard errors
model_ols_fe = smf.ols("log_tradeflow ~ log_gdp_o + C(iso3_o)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["iso3_o"]}
)
print(model_ols_fe.summary())

# 3. Predicted values
df["log_tradeflow_hat"] = model_ols_fe.predict(df)

# 4. Separate predictions by country
countries = df["iso3_o"].unique()
country_preds = {
    country: df[df["iso3_o"] == country]["log_tradeflow_hat"] for country in countries
}

# 5. Plot the results
plt.figure(figsize=(10, 6))

# First group (1-99 countries)
for country in countries[:99]:
    plt.plot(
        df[df["iso3_o"] == country]["log_gdp_o"],
        country_preds[country],
        label=f"{country}",
        color="blue",
        alpha=0.5,
    )

# Second group (100-126 countries)
for country in countries[99:]:
    plt.plot(
        df[df["iso3_o"] == country]["log_gdp_o"],
        country_preds[country],
        label=f"{country}",
        color="green",
        alpha=0.5,
    )

# Add a linear fit line
plt.plot(
    df["log_gdp_o"], model_ols.predict(df), color="red", linewidth=2, label="Linear Fit"
)

# Final plot settings
plt.xlabel("Logarithm of GDP")
plt.ylabel("Predicted Logarithm of Trade Flow")
plt.title("OLS with and without Fixed Effects by Country")
plt.show()
