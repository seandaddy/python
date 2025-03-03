# %%
import pandas as pd

# %%
df = pd.read_csv("/Users/eer/Downloads/gini.csv")
print(df.head())
# %%
df.rename(columns={"Country Name": "Country", "Country Code": "code"}, inplace=True)

# %%
df.set_index("Country", inplace=True)
print(df.head())
# %%
df_long = df.reset_index().melt(
    id_vars=["Country", "code"], var_name="Year", value_name="gini"
)

# Display the first few rows of the long format DataFrame
print(df_long.head())
# %%

# Drop rows with NaN values in 'Military expenditure (% of GDP)'
df_long.dropna(subset=["gini"], inplace=True)

# Display the first few rows of the cleaned long format DataFrame
# print(df_long.head())
# %%
df_long.to_csv("/Users/eer/Downloads/gini_ex_long.csv", index=False)
