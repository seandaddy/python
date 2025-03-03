# %%
import pandas as pd
import numpy as np
from linearmodels import RandomEffects
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_excel("/Users/drsyoh/Documents/python/data/gravity91.xlsx")
df.columns
# %%
countries = ["China", "United States", "Japan"]
filtered_data = df[df["Country"].isin(countries)]

# Calculate growth rate for Export1
filtered_data["Export1_growth"] = (
    filtered_data.groupby("Country")["Export1"].pct_change() * 100
)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Export1
for country in countries:
    country_data = filtered_data[filtered_data["Country"] == country]
    plt.plot(
        country_data["quarter"],
        country_data["Export1"],
        label=f"{country} Export",
        marker="o",
    )

lines1, labels1 = ax1.get_legend_handles_labels()

# Create a second y-axis for growth rate
ax2 = ax1.twinx()
for country in countries:
    country_data = filtered_data[filtered_data["Country"] == country]
    ax2.plot(
        country_data["quarter"],
        country_data["Export1_growth"],
        label=f"{country} Growth Rate",
        linestyle="--",
        marker="s",
    )

lines2, labels2 = ax2.get_legend_handles_labels()

# Get handles and labels from the second plot
lines2, labels2 = ax2.get_legend_handles_labels()

# Combine legends from both y-axes
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Customize the plot
# plt.title('Export1 and Growth Rate for China and United States')
plt.xlabel("Quarter")
plt.ylabel("Export")
ax2.set_ylabel("Growth Rate (%)")

plt.grid(True)
plt.savefig("./figure/export_growth_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
df = pd.read_excel("/Users/drsyoh/Documents/python/data/gravity91.xlsx")
# First, create log transformations
df["ln_Export"] = np.log(df["Export1"])
df["ln_GDP"] = np.log(df["GDP1"])
df["ln_Dist"] = np.log(df["Dist1"])

# Convert quarter to datetime or numeric format
# Option 1: If quarter is in format like "2020Q1", "2020Q2", etc.
df["quarter"] = pd.to_datetime(df["quarter"].astype(str).str.replace("q", "-"))
# Set up panel data structure
# %%
# df = df.set_index(["Country", "quarter"])
# %%
oecd_countries = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
    'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg',
    'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic',
    'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States'
]

# Separate the data based on OECD membership
oecd_data = df[df['Country'].isin(oecd_countries)]
non_oecd_data = df[~df['Country'].isin(oecd_countries)]
# %%
df = df.set_index(['Country', 'quarter'])
oecd_data = oecd_data.set_index(['Country', 'quarter'])
non_oecd_data = non_oecd_data.set_index(['Country', 'quarter'])

# %%
# Add a constant to the datasets
df = sm.add_constant(df)
oecd_data = sm.add_constant(oecd_data)
non_oecd_data = sm.add_constant(non_oecd_data)

# %%
# Define the dependent and independent variables
dependent_var = 'ln_Export'
independent_vars = ['const', 'ln_GDP', 'ln_Dist', 'Rta', 'Strict1', 'Cases1', 'fiscal1']

# Function to run random effects model
def run_random_effects(data, dependent_var, independent_vars):
    formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
    model = RandomEffects.from_formula(formula, data)
    result = model.fit()
    return result

# Run the model on each dataset
result_df = run_random_effects(df, dependent_var, independent_vars)
result_oecd = run_random_effects(oecd_data, dependent_var, independent_vars)
result_non_oecd = run_random_effects(non_oecd_data, dependent_var, independent_vars)

# Print the summary of results
print("Results for the full dataset:")
print(result_df.summary)

print("\nResults for the OECD dataset:")
print(result_oecd.summary)

print("\nResults for the non-OECD dataset:")
print(result_non_oecd.summary)
# print("Variance Decomposition for Full Dataset:")
# print(result_df.variance_decomposition)

# print("\nVariance Decomposition for OECD Dataset:")
# print(result_oecd.variance_decomposition)

# print("\nVariance Decomposition for Non-OECD Dataset:")
# print(result_non_oecd.variance_decomposition)

# %%
# Extract sigma_u, sigma_e, and rho
def extract_model_stats(result):
    sigma_u = result.variance_decomposition['Effects']
    sigma_e = result.variance_decomposition['Residual']
    rho = result.variance_decomposition['Percent due to Effects']
    return sigma_u, sigma_e, rho

sigma_u_df, sigma_e_df, rho_df = extract_model_stats(result_df)
sigma_u_oecd, sigma_e_oecd, rho_oecd = extract_model_stats(result_oecd)
sigma_u_non_oecd, sigma_e_non_oecd, rho_non_oecd = extract_model_stats(result_non_oecd)

# # Create a summary table
summary_table = pd.DataFrame({
    'Dataset': ['Full', 'OECD', 'Non-OECD'],
    'Sigma_u': [sigma_u_df, sigma_u_oecd, sigma_u_non_oecd],
    'Sigma_e': [sigma_e_df, sigma_e_oecd, sigma_e_non_oecd],
    'Rho': [rho_df, rho_oecd, rho_non_oecd]
})

print(summary_table)
# %%
# Extract model coefficients and statistics
def extract_model_stats(result):
    stats = {
        'Coef.': result.params,
        'Std.Err.': result.std_errors,
        'P>|t|': result.pvalues
    }
    return pd.DataFrame(stats)

# Create summary tables for each dataset
summary_df = extract_model_stats(result_df)
summary_oecd = extract_model_stats(result_oecd)
summary_non_oecd = extract_model_stats(result_non_oecd)

# Add sigma_u, sigma_e, and rho to the summary tables
def add_variance_decomposition(summary, result):
    variance_decomp = result.variance_decomposition
    summary.loc['sigma_u'] = [variance_decomp['Effects'], None, None]
    summary.loc['sigma_e'] = [variance_decomp['Residual'], None, None]
    summary.loc['rho'] = [variance_decomp['Percent due to Effects'], None, None]
    return summary

summary_df = add_variance_decomposition(summary_df, result_df)
summary_oecd = add_variance_decomposition(summary_oecd, result_oecd)
summary_non_oecd = add_variance_decomposition(summary_non_oecd, result_non_oecd)

# Format the results with asterisks for significance levels and standard errors in parentheses
def format_results(summary):
    formatted = summary.copy()
    for index, row in summary.iterrows():
        if pd.notnull(row['P>|t|']):
            stars = ''
            if row['P>|t|'] < 0.01:
                stars = '***'
            elif row['P>|t|'] < 0.05:
                stars = '**'
            elif row['P>|t|'] < 0.1:
                stars = '*'
            formatted.at[index, 'Coef.'] = f"{row['Coef.']:.4f}{stars}"
            formatted.at[index, 'Std.Err.'] = f"({row['Std.Err.']:.4f})"
        else:
            formatted.at[index, 'Coef.'] = f"{row['Coef.']:.4f}"
            formatted.at[index, 'Std.Err.'] = ''
    return formatted[['Coef.', 'Std.Err.']]

formatted_df = format_results(summary_df)
formatted_oecd = format_results(summary_oecd)
formatted_non_oecd = format_results(summary_non_oecd)

# Combine the formatted summary tables into one table
combined_summary = pd.concat([formatted_df, formatted_oecd, formatted_non_oecd], axis=1, keys=['Full', 'OECD', 'Non-OECD'])

# Export the combined summary table to LaTeX
latex_table = tabulate(combined_summary, headers='keys', tablefmt='latex')
with open('random_effects_summary.tex', 'w') as f:
    f.write(latex_table)

print("Summary table exported to random_effects_summary.tex")
