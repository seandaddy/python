# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from scipy import stats

# %%
df = pd.read_excel("/Users/eer/Documents/python/data/DAIZIHAN  DATA.xlsx")
semdf = df.dropna()

# %%
# Perform VIF test on "GDP POP WGI patent TTS science open DIS RTA TC"
X = semdf[["GDP", "POP", "WGI", "patent", "TTS", "science", "open", "DIS", "RTA", "TC"]]
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# calculating VIF for each feature
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]
print(vif_data)

# %%
# Perform the correlation analysis on X
X = semdf[
    [
        "sem",
        "IC",
        "DOS",
        "GDP",
        "POP",
        "WGI",
        "patent",
        "TTS",
        "science",
        "open",
        "DIS",
        "RTA",
        "TC",
    ]
]
correlation_matrix = X.corr()
print(correlation_matrix)

# %%
# Run Panel regression the variable,
# sem on TC, TTS, GDP, DIS, POP, open, science, WGI, patent, RTA, oecd
# with fixed and random effect, as well as pooled ols method
# Create a panel data structure with 'Country' and 'Year' as index
years = pd.to_datetime(semdf["years"], errors="coerce")
ID = pd.Categorical(semdf["ID"])
df = semdf.set_index(["ID", "years"])
df["years"] = years
df["ID"] = ID
exog_vars = [
    "TC",
    "TTS",
    "GDP",
    "DIS",
    "POP",
    "open",
    "science",
    "WGI",
    "patent",
    "RTA",
]
exog = sm.add_constant(df[exog_vars])
# Random Effects Regression
re_model = RandomEffects(df["sem"], exog)
re_results = re_model.fit()
# print(re_results)
# Pooled OLS Regression
pooled_model = PooledOLS(df["sem"], exog)
pooled_results = pooled_model.fit()
# print(pooled_results)
# Fixed Effects (Within) Regression
fe_model = PanelOLS(df["sem"], exog, entity_effects=True, drop_absorbed=True)
fe_results = fe_model.fit()
# print(fe_results)
# Compare the models using linearmodels.panel.compare
print(compare({"FE": fe_results, "RE": re_results, "Pooled": pooled_results}))

# %%
# Perform Hausman test
# Extract coefficients and covariance matrices
b_fe = fe_results.params
b_re = re_results.params
# Covariance matrices
cov_fe = fe_results.cov
cov_re = re_results.cov
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

# Export results to LaTeX
# with open("china_results.tex", "w") as f:
#     f.write("\\documentclass{article}\n")
#     f.write("\\usepackage{booktabs}\n")
#     f.write("\\begin{document}\n")
#     f.write("\\section*{Pooled OLS Results}\n")
#     f.write(pooled_results.summary.as_latex())
#     f.write("\\section*{Random Effects Results}\n")
#     f.write(re_results.summary.as_latex())
#     f.write("\\section*{Fixed Effects resultsults}\n")
#     f.write(fe_results.summary.as_latex())
#     f.write("\\section*{Model Comparison}\n")
#     f.write(
#         compare(
#             {"FE": fe_results, "RE": re_results, "Pooled": pooled_results}
#         ).summary.as_latex()
#     )
#     f.write("\\section*{Hausman Test Results}\n")
#     f.write(f"Hausman test statistic: {hausman_statistic:.4f}\\\\\n")
#     f.write(f"p-value: {p_value:.4f}\\\\\n")
#     if p_value < 0.05:
#         f.write("Reject null hypothesis: Fixed effects model is preferred.\n")
#     else:
#         f.write("Fail to reject null hypothesis: Random effects model is preferred.\n")
#     f.write("\\end{document}\n")

# %%
# Run Panel regression the variable,
# IC on TC, TTS, GDP, DIS, POP, open, science, WGI, patent, RTA, oecd
# with fixed and random effect, as well as pooled ols method
# Create a panel data structure with 'Country' and 'Year' as index
years = pd.to_datetime(semdf["years"], errors="coerce")
ID = pd.Categorical(semdf["ID"])
df = semdf.set_index(["ID", "years"])
df["years"] = years
df["ID"] = ID
exog_vars = [
    "TC",
    "TTS",
    "GDP",
    "DIS",
    "POP",
    "open",
    "science",
    "WGI",
    "patent",
    "RTA",
]
exog = sm.add_constant(df[exog_vars])
# Random Effects Regression
re_model = RandomEffects(df["IC"], exog)
re_results = re_model.fit()
# print(re_results)
# Pooled OLS Regression
pooled_model = PooledOLS(df["IC"], exog)
pooled_results = pooled_model.fit()
# print(pooled_results)
# Fixed Effects (Within) Regression
fe_model = PanelOLS(df["IC"], exog, entity_effects=True, drop_absorbed=True)
fe_results = fe_model.fit()
# print(fe_results)
# Compare the models using linearmodels.panel.compare
print(compare({"FE": fe_results, "RE": re_results, "Pooled": pooled_results}))

# %%
# Run Panel regression the variable,
# IC on TC, TTS, GDP, DIS, POP, open, science, WGI, patent, RTA, oecd
# with fixed and random effect, as well as pooled ols method
# Create a panel data structure with 'Country' and 'Year' as index
years = pd.to_datetime(semdf["years"], errors="coerce")
ID = pd.Categorical(semdf["ID"])
df = semdf.set_index(["ID", "years"])
df["years"] = years
df["ID"] = ID
exog_vars = [
    "TC",
    "TTS",
    "GDP",
    "DIS",
    "POP",
    "open",
    "science",
    "WGI",
    "patent",
    "RTA",
]
exog = sm.add_constant(df[exog_vars])
# Random Effects Regression
re_model = RandomEffects(df["DOS"], exog)
re_results = re_model.fit()
# print(re_results)
# Pooled OLS Regression
pooled_model = PooledOLS(df["DOS"], exog)
pooled_results = pooled_model.fit()
# print(pooled_results)
# Fixed Effects (Within) Regression
fe_model = PanelOLS(df["DOS"], exog, entity_effects=True, drop_absorbed=True)
fe_results = fe_model.fit()
# print(fe_results)
# Compare the models using linearmodels.panel.compare
print(compare({"FE": fe_results, "RE": re_results, "Pooled": pooled_results}))

# %%
lm_model = smf.mixedlm(
    "DOS ~ TC + TTS + GDP + DIS + POP + open + science + WGI + patent + RTA",
    data=semdf,
    groups=semdf["ID"],
    re_formula="~1",
)
lm_results = lm_model.fit()
print(lm_results.summary())
