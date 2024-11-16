# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %%
df = pd.read_csv("~/Documents/python/data/hospitalvisits.csv")

# %%
# Step 1: Filter the DataFrame to get the patients in the ER department of the EAST hospital branch
east_er_patients = df[(df['Hospital Branch'] == 'East') & (df['Department'] == 'ER')]

# Step 2: Count the number of patients in the filtered DataFrame
east_er_count = east_er_patients['Number of Patient Visits'].sum()

# Step 3: Count the total number of patients in the DataFrame
total_patients_count = df['Number of Patient Visits'].sum()

# Step 4: Calculate the percentage
percentage_east_er = (east_er_count / total_patients_count) * 100

print(f"The percentage of total count of patients in the ER department of the EAST hospital branch is {percentage_east_er:.2f}%")
# %%
# Step 1: Filter the DataFrame to get the patients in the ER department of the EAST hospital branch
south_er_icu_patients = df[(df['Hospital Branch'] == 'South') & (df['Department'].isin(['ER', 'ICU']))]

# Step 2: Count the number of patients in the filtered DataFrame
south_er_icu_count = south_er_icu_patients['Number of Patient Visits'].sum()

# Step 3: Calculate the percentage
percentage_south_er_icu = (south_er_icu_count / total_patients_count) * 100

print(f"The percentage of total count of patients in the ER department of the EAST hospital branch is {percentage_south_er_icu:.2f}%")
# %%
# Filter for cardiology and group by branch
cardiology_by_branch = df[df['Department'] == 'Cardiology'].groupby('Hospital Branch')['Number of Patient Visits'].sum()

print("Patient visits to Cardiology by branch:")
print(cardiology_by_branch)
# %%
# Group by department and calculate percentage
dept_percentages = (df.groupby('Department')['Number of Patient Visits'].sum() / total_patients_count * 100)

print("Percentage of total visits by department:")
print(dept_percentages)

# %%
df = pd.read_excel("~/Documents/python/data/superstore.xls")

# Filter for furniture category
furniture_data = df[df['Product Category'] == 'Furniture']

# Prepare the data
# Assuming we're using Quantity as independent variable and Sales as dependent variable
X = furniture_data['Profit'].values.reshape(-1,1)
y = furniture_data['Sales'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared
y_pred = model.predict(X)
r_squared = r2_score(y, y_pred)

print(f"R-squared value for Furniture category: {r_squared:.4f}")

# %%
# Filter data for East region and Office Supplies
east_office = df[(df['Region'] == 'East') &
                 (df['Product Category'] == 'Office Supplies')]

# Calculate correlation
correlation = east_office['Sales'].corr(east_office['Profit'])
print(f"Correlation between Sales and Profit: {correlation:.4f}")

# %%
south_furniture = df[(df['Region'] == 'South') &
                 (df['Product Category'] == 'Furniture')]

# Calculate correlation
correlation = south_furniture['Sales'].corr(south_furniture['Profit'])
print(f"Correlation between Sales and Profit: {correlation:.4f}")
