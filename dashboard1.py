# %%
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt

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

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    [
        html.H1("USA and China Terms of Trade", style={"textAlign": "center", "margin-top": "20px", "margin-bottom": "20px", "margin-left": "30px", "margin-right": "30px"}),
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
                            value="Export",
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
                            value=None,
                            multi=True,
                            options = [{"label": f"{str(hs_code)} - {ename}", "value": hs_code} for hs_code, ename in zip(df["eHS2"].unique(), df["ename"])],
                            clearable=True,
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
                                        value=df["date"].min(),
                                        clearable=False,
                                    ),
                                ],
                                style={"width": "48%", "padding-right": "10px"}
                            ),


                            html.Div(
                                [
                                    html.Label("Select End Year:"),
                                    dcc.Dropdown(
                                        id="end-year-dropdown",
                                        options=[{"label": str(year), "value": year} for year in sorted(df["date"].unique())],
                                        value=df["date"].max(),
                                        clearable=False,
                                    ),
                                ],
                                style={"width": "48%"}
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
        Input("data-type-dropdown", "value"),
    ],
)
def update_graphs(selected_hs_codes, start_year, end_year, selected_data_type):
    # Determine whether to use Export or Import columns based on selection
    hs_code_column = "eHS2" if selected_data_type == "Export" else "iHS2"
    category_column = "ename" if selected_data_type == "Export" else "iname"
    value_column = selected_data_type  # 'Export' or 'Import'

    # Filter data based on selected HS codes and years
    filtered_df = df[
        (df[hs_code_column].isin(selected_hs_codes) if selected_hs_codes else True) &
        (df["date"] >= start_year) &
        (df["date"] <= end_year)
    ]
    filtered_df = filtered_df[filtered_df['rank'] != 99]

    # Define the color palette from matplotlib
    color_palette = plt.cm.tab10
    unique_categories = filtered_df[category_column].unique()
    color_mapping = {category: f"rgb({int(color_palette(i % 10)[0]*255)}, {int(color_palette(i % 10)[1]*255)}, {int(color_palette(i % 10)[2]*255)})"
                     for i, category in enumerate(unique_categories)}

    # Visualization 3: Stacked Area Chart (Export/Import Categories by Rank Over Time)
    top_categories = (
        filtered_df.groupby(["date", category_column])[value_column]
        .sum()
        .reset_index()
        .groupby("date")
        .head(10)
    )
    stacked_area_chart = px.area(
        top_categories,
        x="date",
        y=value_column,
        color=category_column,
        title=f"Top 10 {selected_data_type} Categories by Rank Over Time",
        labels={value_column: f"{selected_data_type} Value (Million USD)", category_column: "Category", "date": "Year"},
        color_discrete_map=color_mapping,
    )
    stacked_area_chart.update_layout(
        legend=dict(
            title="Category",
            itemwidth=100,
            itemsizing="constant",
            font=dict(size=12),
        )
    )

    # Visualization 2: Line Chart (Export/Import & Trade Balance Over Time)
    trade_balance = (
        filtered_df.groupby("date")[["Import", "Export"]]
        .sum()
        .reset_index()
    )
    trade_balance["Trade Balance"] = trade_balance["Export"] - trade_balance["Import"]

    max_value = max(trade_balance["Export"].abs().max(), trade_balance["Import"].abs().max(), trade_balance["Trade Balance"].abs().max())+10

    line_chart = px.line(
        trade_balance,
        x="date",
        y=["Export", "Import", "Trade Balance"],
        title=f"Export, Import & Trade Balance Over Time",
        labels={"Export": "Export Value (Million USD)", "Import": "Import Value (Million USD)", "Trade Balance": "Trade Balance (Million USD)", "date": "Year"},
    )
    line_chart.update_layout(
        yaxis_title="Value (Million USD)",
        yaxis=dict(
            range=[-max_value, max_value]
        )
    )

    # Visualization 1: Bar Chart (Top 10 Export/Import Categories by Value)
    top_categories = (
        filtered_df.groupby(category_column)[value_column]  # Dynamically use Export/Import
        .sum()
        .reset_index()
        .sort_values(by=value_column, ascending=False)
        .head(10)
    )
    bar_chart = px.bar(
        top_categories,
        x=category_column,
        y=value_column,
        title=f"Overall Top 10 {selected_data_type} Categories by Value",
        labels={value_column: f"{selected_data_type} Value (Million USD)", category_column: "Category"},
        color_discrete_map=color_mapping,  # Apply the color mapping here
    )

    # Visualization 4: Scatter Plot (Export Share vs Export Value or Import Share vs Import Value)
    scatter_plot = px.scatter(
        filtered_df,
        x="eshare" if selected_data_type == "Export" else "ishare",
        y=value_column,
        color=category_column,
        title=f"{selected_data_type} Share vs {selected_data_type} Value",
        labels={"eshare": "Export Share (%)", "ishare": "Import Share (%)", value_column: f"{selected_data_type} Value (Million USD)"},
        color_discrete_map=color_mapping,
    )

    return stacked_area_chart, line_chart, bar_chart, scatter_plot


@app.callback(
    Output("hs-code-filter", "options"),
    Input("data-type-dropdown", "value"),
)
def update_hs_code_options(selected_data_type):
    if selected_data_type == "Export":
        options = [{"label": f"{str(hs_code)} - {ename}", "value": hs_code} for hs_code, ename in zip(df["eHS2"].unique(), df["ename"])]
    else:
        options = [{"label": f"{str(hs_code)} - {iname}", "value": hs_code} for hs_code, iname in zip(df["iHS2"].unique(), df["iname"])]

    return options

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
