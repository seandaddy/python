# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS, compare
from scipy import stats

# %%
gravity = pd.read_csv("~/Documents/python/data/gravity80.csv")
gravity = gravity[gravity["year"] > 1995]
df = gravity
df = df[df["eu_o"] == 1]
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
# %%
# Export results to LaTeX
with open("results.tex", "w") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\begin{document}\n")

    f.write("\\section*{Pooled OLS Results}\n")
    f.write(pooled_res.summary.as_latex())

    f.write("\\section*{Random Effects Results}\n")
    f.write(re_res.summary.as_latex())

    f.write("\\section*{Between OLS Results}\n")
    f.write(be_res.summary.as_latex())

    f.write("\\section*{Fixed Effects Results}\n")
    f.write(fe_res.summary.as_latex())

    f.write("\\section*{Fixed Effects with Time Effects Results}\n")
    f.write(fe_te_res.summary.as_latex())

    f.write("\\section*{Model Comparison}\n")
    f.write(
        compare({"FE": fe_res, "RE": re_res, "Pooled": pooled_res}).summary.as_latex()
    )

    f.write("\\section*{Hausman Test Results}\n")
    f.write(f"Hausman test statistic: {hausman_statistic:.4f}\\\\\n")
    f.write(f"p-value: {p_value:.4f}\\\\\n")
    if p_value < 0.05:
        f.write("Reject null hypothesis: Fixed effects model is preferred.\n")
    else:
        f.write("Fail to reject null hypothesis: Random effects model is preferred.\n")

    f.write("\\end{document}\n")
