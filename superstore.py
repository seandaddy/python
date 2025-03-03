# %%
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

df = pd.read_csv("./data/superstore.csv")
df.columns
# %%
# Calculate profit margin
df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    [
        html.H1(
            "Superstore Profitability Analysis Dashboard",
            style={"textAlign": "center", "margin": "20px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Category:"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[
                                {"label": x, "value": x}
                                for x in df["Category"].unique()
                            ],
                            value=None,
                            multi=True,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Select Time Period:"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=df["Order Date"].min(),
                            end_date=df["Order Date"].max(),
                        ),
                    ],
                    width=4,
                ),
            ],
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [html.H3("Sales vs Profit Analysis"), dcc.Graph(id="scatter-plot")],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3("Bottom 10 Products by Profit"),
                        dcc.Graph(id="bottom-products"),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Profit Margin by Category"),
                        dcc.Graph(id="category-profit"),
                    ],
                    width=12,
                )
            ]
        ),
    ]
)


# Callbacks
@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("bottom-products", "figure"),
        Output("category-profit", "figure"),
    ],
    [
        Input("category-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_graphs(categories, start_date, end_date):
    # Filter data based on inputs
    filtered_df = df.copy()
    if categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(categories)]
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df["Order Date"] >= start_date)
            & (filtered_df["Order Date"] <= end_date)
        ]

    # Scatter plot
    scatter_fig = px.scatter(
        filtered_df,
        x="Sales",
        y="Profit",
        color="Category",
        size="Order Quantity",
        hover_data=["Item"],
        title="Sales vs Profit Analysis",
    )

    # Bottom 10 products
    bottom_10 = (
        filtered_df.groupby("Item")["Profit"].sum().sort_values().head(10).reset_index()
    )
    bottom_fig = px.bar(
        bottom_10, x="Item", y="Profit", title="Bottom 10 Products by Profit"
    )
    bottom_fig.update_xaxes(tickangle=45)

    # Category profit margins
    category_profits = filtered_df.groupby("Category")[["Profit", "Sales"]].sum()
    category_profits["Profit_Margin"] = (
        category_profits["Profit"] / category_profits["Sales"]
    ) * 100
    category_fig = px.bar(
        category_profits.reset_index(),
        x="Category",
        y="Profit_Margin",
        title="Profit Margin by Category",
    )

    return scatter_fig, bottom_fig, category_fig


# Add this analysis to help answer the management's question
def analyze_profitability():
    # Group by Item and calculate various metrics
    item_analysis = (
        df.groupby("Item")
        .agg(
            {
                "Profit": ["mean", "std", "count"],
                "Sales": "sum",
                "Order Quantity": "sum",
            }
        )
        .reset_index()
    )

    # Find items that are consistently unprofitable
    consistently_unprofitable = item_analysis[
        (item_analysis[("Profit", "mean")] < 0)
        & (item_analysis[("Profit", "count")] > 5)  # More than 5 orders
    ]

    # Calculate variation in profitability
    item_analysis["Profit_Volatility"] = item_analysis[("Profit", "std")] / abs(
        item_analysis[("Profit", "mean")]
    )

    return consistently_unprofitable, item_analysis


if __name__ == "__main__":
    # Perform analysis
    consistently_unprofitable, item_analysis = analyze_profitability()

    print("\nAnalysis Findings:")
    print(
        f"Number of consistently unprofitable items: {len(consistently_unprofitable)}"
    )
    print("\nTop 5 most consistently unprofitable items:")
    print(consistently_unprofitable.head())

    app.run_server(debug=True)


# This dashboard and analysis will help answer the management's question by:
# 1. **Visualizing the relationship** between sales and profits to identify if higher sales volume correlates with better profitability
# 2. **Identifying consistently unprofitable items** through the bottom 10 products chart and the detailed analysis
# 3. **Showing profit margins by category** to identify if certain categories are more prone to low profitability
# 4. **Allowing interactive filtering** to analyze profitability patterns across different time periods and categories

# Key insights that can be derived:
# 1. Whether unprofitable items are consistently unprofitable or just temporarily underperforming
# 2. If certain categories have systematic profitability issues
# 3. The relationship between sales volume and profitability
# 4. Seasonal patterns in item profitability

# To use this code:
# 1. Make sure you have the required libraries installed (pandas, plotly, dash, dash-bootstrap-components)
# 2. Update the data path in the code to point to your superstore dataset
# 3. Run the code to launch the interactive dashboard
# 4. Use the filters to explore different aspects of profitability

# The analysis will help management make informed decisions about:
# - Which items to actually remove from inventory
# - Which items might need pricing strategy adjustments
# - Which items might be worth keeping despite low profitability due to other strategic reasons
