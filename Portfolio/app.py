#! /Users/glebsokolov/opt/miniconda3/envs/default/bin/python
import dash
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.dash_table import DataTable

prod = 1

if prod:
    nyse = pd.read_csv("nyse.csv")
else:
    nyse = pd.read_csv("nyse-data/nyse.csv")

nyse["Period Ending"] = nyse["Period Ending"].apply(pd.to_datetime)
nyse["Year"] = nyse["Period Ending"].apply(lambda x: x.year)


def budget():
    t = nyse.groupby(["GICS Sector"])[numerical_sum].sum().reset_index()


def filter_q(col):
    return nyse[[col]][nyse[col] < nyse[col].quantile(0.95)]


def table_temp(cols, table, style_cond=[]):
    table = table[cols]
    columns = [{"name": x, "id": x} for x in table.columns]
    return DataTable(
        style_table={"overflowX": "auto"},
        style_cell={
            "height": "auto",
            "textAlign": "center",
            # all three widths are needed
            # 'minWidth': '15px', 'width': '15px', 'maxWidth': '15px',
            "whiteSpace": "normal",
        },
        columns=columns,
        data=table.to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=8,
        page_current=0,
        page_action="native",
        style_cell_conditional=style_cond,
    )


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(external_stylesheets=external_stylesheets)
server = app.server

numerical = [
    "Total Revenue",
    "Cost of Goods Sold",
    "Sales, General and Admin.",
    "Research and Development",
    "Other Operating Items",
]
numerical_sum = [i + "_sum" for i in numerical]

total_revenue_hist = ff.create_distplot(
    hist_data=[
        filter_q("Total Revenue").values.reshape(
            -1,
        )
    ],
    group_labels=["Distribution of Total Revenues"],
    bin_size=5,
)
total_revenue_hist.update_layout(template="ggplot2")

scatter_general = px.scatter(
    data_frame=nyse,
    title="Total Sales by Country and Month",
    y="Total Revenue",
    x="Research and Development",
    color="GICS Sector",
    template="ggplot2",
)
heatmap = px.imshow(
    nyse.drop("Unnamed: 0", axis=1).corr(),
    aspect="auto",
    height=600,
    template="ggplot2",
)
heatmap.update_layout(font_size=8)

scatter_matrix = px.scatter_matrix(nyse[numerical], height=600)
scatter_matrix.update_layout(font_size=6, template="ggplot2")

total_sales = px.bar(
    data_frame=nyse[["GICS Sector", "Cost of Goods Sold", "Years"]]
    .groupby(["GICS Sector", "Years"])
    .agg(total_sales=("Cost of Goods Sold", "sum"))
    .reset_index()
    .rename(columns={"total_sales": "Total Sales"}),
    y="Total Sales",
    x="GICS Sector",
    color="Years",
    barmode="group",
    title="Sum of Total Sales for each Sector, yearly",
    template="ggplot2",
)

counts = (
    nyse.groupby(["GICS Sub Industry"])[["Unnamed: 0"]]
    .count()
    .rename(columns={"Unnamed: 0": "count"})
    .reset_index()
)

d_columns = [{"name": x, "id": x} for x in counts.columns]

count_table_conds = [
    {
        "if": {"column_id": "GICS Sub Industry"},
        "minWidth": "100px",
        "width": "100px",
        "maxWidth": "100px",
        "whiteSpace": "normal",
    },
    {
        "if": {"column_id": "count"},
        "minWidth": "50px",
        "width": "50px",
        "maxWidth": "50px",
        "whiteSpace": "normal",
        "textAlign": "center",
    },
]
pd.set_option("display.precision", 4)
overview_table_conds = [
    {"if": {"column_id": col}, "width": "50px", "maxWidth": "50px", "minWidth": "50px"}
    for col in numerical
]

overview = table_temp(
    ["Ticker Symbol", "Year", "GICS Sector", "GICS Sub Industry"] + numerical,
    nyse.drop("Unnamed: 0", axis=1),
    style_cond=[],
)

temp = nyse.drop("Unnamed: 0", axis=1).describe().reset_index().round(3)
summary = table_temp(["index"] + numerical, temp)

d_table = DataTable(
    columns=d_columns,
    data=counts.to_dict("records"),
    sort_action="native",
    filter_action="native",
    page_size=10,
    page_current=0,
    page_action="native",
    style_cell_conditional=count_table_conds,
)
means_by_cat = px.histogram(
    data_frame=nyse.sort_values(by="Total Revenue", ascending=False),
    x="GICS Sector",
    template="ggplot2",
)

# grouped = nyse.drop("Unnamed: 0", axis=1).groupby("GICS Sector")


selector_sector = lambda x: html.Div(
    dcc.Dropdown(
        nyse["GICS Sector"].unique(),
        "Industrials",
        id=x,
    ),
)

selector = lambda x: html.Div(
    dcc.Dropdown(
        [
            "Total Revenue",
            "Cost of Goods Sold",
            "Sales, General and Admin.",
            "Research and Development",
            "Other Operating Items",
        ],
        "Total Revenue",
        id=x,
    ),
    style={"width": "30%"},
)


def breaks(num):
    return [html.Br()] * num


app.layout = html.Div(
    children=[
        # html.Div(
        #     [
        #         html.Div(
        #             [
        #                 html.H3("Column 1"),
        #                 overview,
        #             ],
        #             className="six columns",
        #         ),
        #         html.Div(
        #             [
        #                 html.H3("Column 2"),
        #                 summary,
        #             ],
        #             className="six columns",
        #         ),
        #     ],
        #     className="container",
        # ),
        html.H1("Lets view some graphs"),
        html.Span(
            "First, let's have a look on the data itself (you can sort and filter each column's contents). \nNote that all numerical values expressed in $ billions "
        ),
        *breaks(3),
        html.Div(
            [
                html.Div(
                    [overview],
                    # style={"width": "50%"},
                    className="eight columns",
                ),
                # html.Span("The most informative data is contained in this brief summary"),
                # *breaks(3),
                html.Div(
                    [summary],
                    # style={"width": "30%"},
                    className="four columns",
                ),
            ],
            className="container",
        ),
        *breaks(1),
        html.Span(
            "Obviously, we are mostly interested in total revenues and total sales and how are they impacted by other variables. So, first, let's see how is this most important value  distributed"
        ),
        html.Div(
            dcc.Graph(figure=total_revenue_hist, id="revenue_hist"),
            style={"display": "inline-block", "width": "100%"},
        ),
        *breaks(1),
        html.Div(
            dcc.Graph(figure=scatter_general, id="scatter_general"),
            style={"display": "inline-block", "width": "100%"},
        ),
        html.Span(
            "In a dropdown list here you can choose an industrial sector and view scatterplot of a relation between the relation of research and developement amount of a company and its total sales"
        ),
        *breaks(2),
        html.Div(
            dcc.Dropdown(
                options=nyse["GICS Sector"].unique(),
                value="Industrials",
                id="sector-dd",
            ),
            style={"width": "30%"},
        ),
        *breaks(1),
        html.Div(
            dcc.Graph(id="scatter-graph"),
        ),
        *breaks(1),
        html.Div(
            dcc.Graph(figure=total_sales, id="sales-comparison"),
        ),
        html.Span(
            "As we can see here, there are industries where the impact of the research and developement amount is obvious (i.e. Industrials) and where it doesn't play any role (Consumer Discretionary)"
        ),
        *breaks(2),
        html.Span("Here you can choose a variable and view its distribution"),
        *breaks(2),
        selector("var-dd"),
        dcc.Graph(id="hist"),
        html.Br(),
        html.Span("Number of tickers by Industry (table is sortable)"),
        *breaks(3),
        html.Div(
            d_table,
            style={"display": "inline-block", "width": "20%"},
        ),
        html.Div(
            dcc.Graph(id="mean-num", figure=means_by_cat),
            style={"display": "inline-block", "width": "80%"},
        ),
        *breaks(2),
        html.Span(
            "Here you can choose two different sectors and compare their key metrics (revenue, expences, etc.)\
        Note that you can scale the picture removing outliers (top 5 % tickers with max metric value)"
        ),
        *breaks(1),
        html.Span("The Table below contains ticker symbols for outlier values"),
        *breaks(2),
        html.Div(
            dcc.Dropdown(
                nyse["GICS Sector"].unique(),
                "Industrials",
                id="select-sector",
            ),
            style={"display": "inline-block", "width": "30%"},
        ),
        html.Div(
            dcc.Dropdown(
                nyse["GICS Sector"].unique(),
                "Industrials",
                id="select-sector2",
            ),
            style={"display": "inline-block", "width": "30%"},
        ),
        html.Div(
            dcc.Dropdown(
                numerical,
                "Total Revenue",
                id="select2",
            ),
            style={"display": "inline-block", "width": "30%"},
        ),
        *breaks(2),
        dcc.RadioItems(
            ["with outliers", "without outliers"], "with outliers", id="outliers"
        ),
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="boxpl"),
                    style={"display": "inline-block", "width": "75%"},
                    className="eight columns",
                ),
                html.Div(
                    DataTable(
                        sort_action="native",
                        filter_action="native",
                        page_size=5,
                        page_current=0,
                        page_action="native",
                        id="outliers_table",
                        columns=[
                            {"name": "Total Revenue", "id": "Total Revenue"},
                            {"name": "Ticker Symbol", "id": "Ticker Symbol"},
                        ],
                    ),
                    style={
                        "display": "inline-block",
                        "width": "15%",
                        "margin-top": "100px",
                    },
                    className="four columns",
                ),
            ],
            className="container",
        ),
        *breaks(3),
        html.Span(
            "Let's calculate total sum for all the years present in the table for each ticker (these values will have postfix _sum)\
            and then, let's have a look at the heatmap, depicting correlation table for the numerical variables"
        ),
        *breaks(1),
        html.Div(
            dcc.Graph(figure=heatmap),
        ),
        # *breaks(2),
        # html.Span(
        # "Now let's have a look at the scatter matrix for the numerical variables of the dataset"
        # ),
        *breaks(1),
        html.Div(
            dcc.Graph(figure=scatter_matrix),
        ),
    ],
    style={"font-family": "verdana", "margin-left": "100px", "margin-right": "100px"},
)


@app.callback(
    Output(component_id="scatter-graph", component_property="figure"),
    Input(component_id="sector-dd", component_property="value"),
)
def update_plot(selection):
    data = nyse.copy(deep=True)
    if selection:
        sector = selection
        choice = data[data["GICS Sector"] == sector]

        return px.scatter(
            data_frame=choice,
            title="Total Sales by Country and Month",
            y="Total Revenue",
            x="Research and Development",
            template="ggplot2",
        )


@app.callback(
    Output(component_id="hist", component_property="figure"),
    Input(component_id="var-dd", component_property="value"),
)
def update_hist(selection):
    data = nyse.copy(deep=True)
    if selection:
        variable = selection
        choice = data.iloc[filter_q("Total Revenue").index][[variable]]
        summary = choice[variable].describe()
        summary["mode"] = choice[variable].mode()[0]
        # summary["skew"] = stats.skew(choice)
        fig = go.Figure()
        f = data.iloc[filter_q("Total Revenue").index][[variable]].values.reshape(
            -1,
        )
        fig = fig.add_trace(go.Histogram(x=f))
        text = "<br>".join(
            [f"{list(summary.index)[i]}:{round(list(summary)[i], 2)}" for i in range(9)]
        )
        fig.add_annotation(text=text, showarrow=False)
        annotation = {"xref": "paper", "yref": "paper", "x": 1, "y": 1, "align": "left"}
        fig.update_annotations(annotation)
        fig.update_layout(
            title=f"Histogram of Distribution of {selection}",
            xaxis_title=selection,
            yaxis_title="Count",
            template="ggplot2",
        )
        return fig


@app.callback(
    Output(component_id="boxpl", component_property="figure"),
    Output(component_id="outliers_table", component_property="data"),
    Input(component_id="select-sector", component_property="value"),
    Input(component_id="select-sector2", component_property="value"),
    Input(component_id="outliers", component_property="value"),
    Input(component_id="select2", component_property="value"),
)
def update_means(selection1, selection2, outliers, metric):
    data = nyse.copy(deep=True).rename({"GICS Sector": "GICS_Sector"}, axis=1)
    if selection1 and selection2 and outliers and metric:
        choice = data.query(
            f"GICS_Sector=='{selection1}' or GICS_Sector=='{selection2}'"
        )
        outl = choice[choice[metric] > choice[metric].quantile(0.95)]
        if outliers == "with outliers":
            pass
        else:
            choice = choice[choice[metric] < choice[metric].quantile(0.95)]
        return (
            px.box(
                data_frame=choice,
                title=f"Comparison of {metric} in {selection1} and <br> {selection2} sectors, yearly",
                x="Year",
                y=metric,
                color="GICS_Sector",
                template="ggplot2",
            ),
            outl.to_dict("records"),
        )


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)
