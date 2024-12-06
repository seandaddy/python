# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import plotly.graph_objects as go
import plotly.io as pio

# Set the renderer for Plotly
pio.renderers.default = 'browser'

df = pd.read_csv("./data/flight.csv")
df.columns
# %%

# Count the number of flights per airline
flights_per_airline = df['Airline'].value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
flights_per_airline.plot(kind='bar')
plt.title('Number of Flights per Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.show()

### 2. Pie Chart: Proportion of Cancelled Flights

# %%
cancelled_flights = df['Cancelled?'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
cancelled_flights.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Cancelled Flights')
plt.ylabel('')
plt.show()

### 3. Box Plot: Distribution of Arrival Delays by Airline


# Plot the box plot
# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Airline', y='Arrival Delay (Min)', data=df)
plt.title('Distribution of Arrival Delays by Airline')
plt.xlabel('Airline')
plt.ylabel('Arrival Delay (Min)')
plt.xticks(rotation=90)
plt.show()

### 4. Tree Map: Number of Flights by Origin State

# Count the number of flights per origin state
# %%
flights_per_state = df['Origin State'].value_counts().reset_index()
flights_per_state.columns = ['State', 'Number of Flights']

# Plot the tree map
fig = px.treemap(flights_per_state, path=['State'], values='Number of Flights',
                 title='Number of Flights by Origin State')
fig.show()
