# %%
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# %%
# Load the CSV data
df = pd.read_csv("./data/us_chn_expimp.csv")

# Preprocess columns
df["date"] = df["date"].astype(int)
df["iHS2"] = df["iHS2"].astype(int)
df["Import"] = df["Import"].astype(float)
df["itext"] = df["itext"].astype(str)
df["rank"] = df["rank"].astype(int)
df["iall"] = df["iall"].astype(float)
df["ishare"] = df["ishare"].astype(float)
df["eHS2"] = df["eHS2"].astype(int)
df["Export"] = df["Export"].astype(float)
df["etext"] = df["etext"].astype(str)
df["eall"] = df["eall"].astype(float)
df["eshare"] = df["eshare"].astype(float)
df["ename"] = df["ename"].astype(str)
df["iname"] = df["iname"].astype(str)
# %%
# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    [
        html.H1("Economic Complexity and Trade Diversification", style={"textAlign": "center", "margin-top": "20px", "margin-bottom": "20px", "margin-left": "30px", "margin-right": "30px"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Export or Import:"),
                        dcc.Dropdown(
                            id="data-type-dropdown",
                            options=[
                                {"label": "Export", "value": "Export"},
                                {"label": "Import", "value": "Import"},
                            ],
                            value="Export",  # Default value
                            clearable=False,
                        ),
                    ],
                    width=2,
                ),

                dbc.Col(
                    [
                        html.Label("Select HS Codes:"),
                        dcc.Dropdown(
                            id="hs-code-filter",
                            options=[{"label": str(hs_code), "value": hs_code} for hs_code in df["eHS2"].unique()],
                            value=None,
                            multi=True,
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Select Start Year:"),
                                    dcc.Dropdown(
                                        id="start-year-dropdown",
                                        options=[{"label": str(year), "value": year} for year in sorted(df["date"].unique())],
                                        value=df["date"].min(),  # Default value: minimum year
                                        clearable=False,
                                    ),
                                ],
                                style={"width": "48%", "padding-right": "10px"}  # 48% width for the first dropdown, with some space on the right
                            ),


                            html.Div(
                                [
                                    html.Label("Select End Year:"),
                                    dcc.Dropdown(
                                        id="end-year-dropdown",
                                        options=[{"label": str(year), "value": year} for year in sorted(df["date"].unique())],
                                        value=df["date"].max(),  # Default value: maximum year
                                        clearable=False,
                                    ),
                                ],
                                style={"width": "48%"}  # 48% width for the second dropdown
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row", "justifyContent": "space-between"}  # Align the divs side by side
                    ),
                    ],
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="bar-chart")], width=6),
                dbc.Col([dcc.Graph(id="line-chart")], width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="stacked-area-chart")], width=12),
            ]
        ),

        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="scatter-plot")], width=12),
            ]
        ),
    ]
)
# %%
# Callbacks
@app.callback(
    [
        Output("stacked-area-chart", "figure"),
        Output("line-chart", "figure"),
        Output("bar-chart", "figure"),
        Output("scatter-plot", "figure"),
    ],
    [
        Input("hs-code-filter", "value"),
        Input("start-year-dropdown", "value"),
        Input("end-year-dropdown", "value"),
        Input("data-type-dropdown", "value"),  # Added input for selecting data type (Export/Import)
    ],
)
def update_graphs(selected_hs_codes, start_year, end_year, selected_data_type):
    # Filter data based on selected years
    filtered_df = df[
        (df["eHS2"].isin(selected_hs_codes) if selected_hs_codes else True) &
        (df["date"] >= start_year) &
        (df["date"] <= end_year)
    ]
    filtered_df = filtered_df[filtered_df['rank'] != 99]

    import matplotlib.pyplot as plt
    # Define the color palette from matplotlib
    color_palette = plt.cm.tab10  # You can also use plt.cm.Paired, plt.cm.Set2, etc.

    # Create the color mapping for the categories
    unique_categories = filtered_df["ename"].unique()
    color_mapping = {category: f"rgb({int(color_palette(i % 10)[0]*255)}, {int(color_palette(i % 10)[1]*255)}, {int(color_palette(i % 10)[2]*255)})"
                    for i, category in enumerate(unique_categories)}

    # Visualization 2: Stacked Area Chart (Export/Import Categories by Rank Over Time)
    top_categories = (
        filtered_df.groupby(["date", "ename"])[selected_data_type]  # Dynamically use Export/Import based on dropdown
        .sum()
        .reset_index()
        .groupby("date")
        .head(10)  # Top 10 categories
    )
    stacked_area_chart = px.area(
        top_categories,
        x="date",
        y=selected_data_type,  # Dynamically use Export/Import
        color="ename",
        title=f"Top 10 {selected_data_type} Categories by Rank Over Time",
        labels={selected_data_type: f"{selected_data_type} Value (Million USD)", "ename": "Category"},
        color_discrete_map=color_mapping,  # Apply the color mapping here
    )
    stacked_area_chart.update_layout(
        legend=dict(
            title="Category",  # Optional legend title
            itemwidth=100,  # Set the maximum width for each legend item
            itemsizing="constant",  # Ensures consistent item sizing
            font=dict(size=12),  # Set font size for legend text
        )
    )

    # Visualization 2: Line Chart (Export/Import & Trade Balance Over Time)
    trade_balance = (
        filtered_df.groupby("date")[["Import", "Export"]]
        .sum()
        .reset_index()
    )
    trade_balance["Trade Balance"] = trade_balance["Export"] - trade_balance["Import"]
    line_chart = px.line(
        trade_balance,
        x="date",
        y=[selected_data_type, "Trade Balance"],  # Dynamically use Export/Import
        title=f"{selected_data_type} & Trade Balance Over Time",
        labels={selected_data_type: f"{selected_data_type} Value (Million USD)", "Trade Balance": "Trade Balance (Million USD)", "date": "Year"},
    )

    # Visualization 1: Bar Chart (Top 10 Export/Import Categories by Value Sorted by )
    top_categories = (
        filtered_df.groupby("ename")[selected_data_type]  # Dynamically use Export/Import
        .sum()
        .reset_index()
        .sort_values(by=selected_data_type, ascending=False)
        .head(10)
    )
    bar_chart = px.bar(
        top_categories,
        x="ename",
        y=selected_data_type,  # Dynamically use Export/Import
        title=f"Overall Top 10 {selected_data_type} Categories by Value",
        labels={selected_data_type: f"{selected_data_type} Value (Million USD)", "ename": "Category"},
        color_discrete_map=color_mapping,  # Apply the color mapping here
    )

    # Visualization 4: Scatter Plot (Export Share vs Export Value or Import Share vs Import Value)
    scatter_plot = px.scatter(
        filtered_df,
        x="eshare" if selected_data_type == "Export" else "ishare",  # Dynamically use Export/Import Share
        y=selected_data_type,  # Dynamically use Export/Import
        color="ename",
        size=selected_data_type,  # Dynamically use Export/Import
        title=f"{selected_data_type} Share vs {selected_data_type} Value",
        labels={"eshare": "Export Share (%)", "ishare": "Import Share (%)", selected_data_type: f"{selected_data_type} Value (Million USD)"},
        color_discrete_map=color_mapping,  # Apply the color mapping here
    )

    return stacked_area_chart, line_chart, bar_chart, scatter_plot

# %%
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
# http://127.0.0.1:8050
