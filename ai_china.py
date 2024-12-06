# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from scipy import stats

# %%
df = pd.read_excel("/Users/eer/Documents/python/data/AI_data1.xlsx")
# List of columns to fill with moving average
columns_to_fill = ['Top1', 'Duality', 'LnSub']

# Fill missing values with moving average for each column
for column in columns_to_fill:
    # Compute the moving average with a window size of 2
    df[f'{column}_Moving_Avg'] = df.groupby('code')[column].transform(lambda x: x.rolling(window=2, min_periods=1).mean())

    # Fill missing values with the moving average
    df[column] = df[column].fillna(df[f'{column}_Moving_Avg'])

    # Drop the 'Moving_Avg' column as it's no longer needed
    df.drop(columns=[f'{column}_Moving_Avg'], inplace=True)

# %%
# Drop rows with missing values
# df = df.dropna()
df.describe()
# semdf = df.dropna()

# %%
# Describe the data
description = df.describe().transpose()

# Select only the required statistics
description = description[['count', 'mean', 'std', 'min', 'max']]
description = description.applymap(lambda x: f"{x:.3f}")

# Convert to LaTeX format
latex_description = description.to_latex()

# Print the LaTeX formatted description
print(latex_description)

# %%
# # Set the index to be a MultiIndex (code, year)
df1 = df.set_index(['code', 'year'])

# Define the dependent variable and independent variables
y = df1['Roa']
X = df1[['LnTax', 'LnSub', 'Cash', 'Lnsize', 'Age', 'Lev', 'Top1', 'CSR']]

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fixed Effects Model
fixed_effects_model = PanelOLS(y, X, entity_effects=True)
fixed_effects_results = fixed_effects_model.fit()
print(fixed_effects_results)

# Random Effects Model
random_effects_model = RandomEffects(y, X)
random_effects_results = random_effects_model.fit()
print(random_effects_results)

# %%
# Run Panel regression the variable,
# Create a panel data structure with 'code' and 'year' as index
year = pd.to_datetime(df["year"], errors="coerce")
code = pd.Categorical(df["code"])
df = df.set_index(["code", "year"])
df["year"] = year
df["code"] = code
exog_vars = [
    'LnTax',
    'LnSub',
    'Cash',
    'Lnsize',
    'Age',
    'Lev',
    'Top1',
    'CSR',
    'Efficiency',
]
exog = sm.add_constant(df[exog_vars])
# Random Effects Regression
re_model = RandomEffects(df["Roa"], exog)
re_results = re_model.fit()
# print(re_results)
# Pooled OLS Regression
pooled_model = PooledOLS(df["Roa"], exog)
pooled_results = pooled_model.fit()
# print(pooled_results)
# Fixed Effects (Within) Regression
fe_model = PanelOLS(df["Roa"], exog, entity_effects=True, drop_absorbed=True)
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
