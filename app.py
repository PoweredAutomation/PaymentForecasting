import pickle
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc  # Dash componetents
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
import matplotlib
matplotlib.use('Agg')
import json
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

app.config.suppress_callback_exceptions = True

app.title = 'Expected Collection Forecasting'

server = app.server


def load_model_monthly():
    # model variable refers to the global variable
    model_monthly = pickle.load( open( "models/forecast_model_monthly.pckl", 'rb' ) )
    return model_monthly


def load_model_weekly():
    # model variable refers to the global variable
    model_weekly = pickle.load( open( r"models/forecast_model_weekly.pckl", 'rb' ) )
    return model_weekly





def future_dates_test(df,periods, freq, model, forecast_type):

    future_dates = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future_dates)
    print(forecast)
    df['closedate'] = pd.DatetimeIndex(df['closedate'])
    df['closedate'] = pd.to_datetime(df['closedate']) - pd.to_timedelta(7, unit='d')
    new_df = pd.DataFrame()
    if forecast_type == "Weekly":
        df_weekly = df.groupby([pd.Grouper(key='closedate', freq='W-MON')])[
            'NC'].sum().reset_index().sort_values('closedate')
        df_weekly = df_weekly.rename(columns={'closedate': 'ds',
                                              'NC': 'y'})
        new_df = pd.merge(df_weekly, forecast, on='ds')
    elif forecast_type == "Monthly":
        df_monthly = df.groupby([pd.Grouper(key='closedate', freq='M')])['NC'].sum().reset_index().sort_values(
            'closedate')
        df_monthly = df_monthly.rename(columns={'closedate': 'ds',
                                                'NC': 'y'})
        new_df = pd.merge(df_monthly, forecast, on='ds')

    last_index_df = len(new_df.index) - 1
    final_df = new_df.loc[:, ["ds", "y", "yhat"]]
    return last_index_df, final_df, forecast


def plot_graph(last_index_df, final_df, forecast, graphtype, oldrow_parameter, pred_type):
    graph_df = forecast.loc[last_index_df + 1:, ["ds", "yhat"]]
    dtick_val = 0
    tickformat_val = ''
    if pred_type == "Weekly":
        dtick_val = 7 * 86400000.0
        tickformat_val = "\n%d\n%b\n%Y"
    elif pred_type == "Monthly":
        dtick_val = 30 * 86400000.0
        tickformat_val = "%b\n\n\n\n\n\n\n\n\n%Y"

    if oldrow_parameter == "All":
        oldrow = last_index_df
    else:
        oldrow = oldrow_parameter
    graph_df_old = final_df.loc[last_index_df - oldrow:, ["ds", "y", "yhat"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=graph_df_old["ds"], y=graph_df_old["y"],
                             mode=graphtype,
                             name='Actual'))
    fig.add_trace(go.Scatter(x=graph_df_old["ds"], y=graph_df_old["yhat"],
                             mode=graphtype,
                             name='Predicted'))
    fig.add_trace(go.Scatter(x=graph_df["ds"], y=graph_df["yhat"],
                             mode=graphtype,
                             name='Forecast'))

    fig.update_layout(title='Time Series Forecasting')

    fig.update_xaxes(dict(tickfont=dict(
        family='Ariel', size=7, color='black')),dtick="M6", tickformat=tickformat_val)

    fig.update_xaxes(dict(tickfont=dict(family='Ariel', size=10, color='black')),dtick=dtick_val)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list( [
                dict( count=1, label="1m", step="month", stepmode="backward" ),
                dict( count=6, label="6m", step="month", stepmode="backward" ),
                dict( count=1, label="YTD", step="year", stepmode="todate" ),
                dict( count=1, label="1y", step="year", stepmode="backward" ),
                dict( step="all" )
            ] )
        )
    )
    return fig


# m2.plot(forecast,uncertainty=True)

app.layout = html.Div(
    [
        dbc.Navbar(

            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Expected Collection Forecasting")

                            ]
                        )
                    ]
                )
            ],
            color="White",
            dark=False,
            fixed='top',
            style={
                'textAlign': 'center',
                'marginBottom': '10px'
            }
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H6("Practice List"),
                                                        dcc.Dropdown(
                                                            id='practiceList',
                                                            # multi=True,
                                                            options=[
                                                                {'label': 'Clinic1', 'value': 'OSD01'},
                                                                {'label': 'Clinic2', 'value': 'MHP-BIOTECH01'}
                                                            ],
                                                            value='',
                                                            placeholder='Practice Name',
                                                        ),
                                                    ],
                                                    # width=2.4
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Type of Period"),
                                                        dcc.Dropdown(
                                                            id='periodList',
                                                            # multi=True,
                                                            options=[
                                                                {'label': 'Weekly', 'value': 'Weekly'},
                                                                {'label': 'Monthly', 'value': 'Monthly'}
                                                            ],
                                                            value='',
                                                            placeholder='Period',
                                                        ),
                                                    ],
                                                    # width=2.4
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Forecast Weeks/Months"),
                                                        dcc.Dropdown(
                                                            id='timeFrame',
                                                            value='',
                                                            placeholder='Time Frame',
                                                        )
                                                    ],
                                                    # width=2.4
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("No. of Records"),
                                                        dcc.Dropdown(
                                                            id='Row_no',
                                                            value='',
                                                            placeholder='No. of Records',
                                                        )
                                                    ],
                                                    # width=2.4
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.H6("Type of graph"),
                                                        dcc.Dropdown(
                                                            id='graph_type',
                                                            # multi=True,
                                                            options=[
                                                                {'label': 'Line Chart', 'value': 'lines'},
                                                                {'label': 'Scatter Plot', 'value': 'markers'}
                                                            ],
                                                            value='',
                                                            placeholder='Type of graph',
                                                        ),
                                                    ],
                                                    # width=2.4
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Forecast",
                                                            id="ShowForecasting"
                                                        )
                                                    ],
                                                    # width=2.4
                                                ),
                                            ]
                                        )
                                    ],
                                    style={
                                        "padding": "10px"
                                    }
                                )
                            ]
                        )
                    ],
                    style={
                        'marginTop': '80px'
                    }
                ),
                html.Div(id="forecasting-graph"),
                html.Div(id="forecasting-components"),
                html.Div(dcc.Graph(id='lineplot'))


            ]
        )
    ]
)


@app.callback(
    [Output('timeFrame', 'options'),
     Output('Row_no', 'options')],
    [
        Input("practiceList", "value"),
        Input("periodList", "value")
    ]
)
def timeFrameList(practiceList, periodList):
    if periodList == "Weekly":
        options = [{'label': val, 'value': val} for val in range(55) if val % 5 == 0]
        options_2 = [{'label': val, 'value': val} for val in range(300) if val % 50 == 0]
        options_2.append({'label': "All", 'value': "All"})
    elif periodList == "Monthly":
        options = [{'label': val, 'value': val} for val in range(1, 25)]
        options_2 = [{'label': val, 'value': val} for val in range(200) if val % 50 == 0]
        options_2.append({'label': "All", 'value': "All"})
    else:
        options = [{'label': val, 'value': val} for val in range(1, 51)]
        options_2 = [{'label': val, 'value': val} for val in range(300) if val % 50 == 0]
        options_2.append({'label': "All", 'value': "All"})
    return options, options_2


@app.callback(
    [Output('forecasting-graph', 'children'),
    Output('forecasting-components', 'children'),
     ],
    [
        Input("practiceList", "value"),
        Input("periodList", "value"),
        Input("timeFrame", "value"),
        Input("ShowForecasting", "n_clicks"),
        Input("Row_no", "value"),
        Input("graph_type", "value")
    ]
)
def show_pred(practiceList, periodList, timeFrame, ShowForecasting, Record_no, graphtype):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "ShowForecasting" in changed_id:
        if practiceList == "OSD01" and periodList == "Weekly" and timeFrame in range(1, 51):
            df = pd.read_csv( r"data/forecastdata.csv" )
            last_index_df, final_df, forecast = future_dates_test(df,timeFrame, 'w', load_model_weekly(), "Weekly")
            print(forecast)
            plotly_fig_components = tls.mpl_to_plotly(load_model_monthly().plot_components(forecast, figsize=(10, 10)))
            # fig=go.Figure()
            # fig.add_trace(load_model_weekly().plot_components(forecast, figsize=(10, 10),))
            # fig.show()


            figData = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Forecasting Graph")),
                                    # dcc.Graph(figure=plot_graph(last_index_df, final_df, forecast, graphtype, Record_no, "Weekly")
                                    dcc.Graph(figure=plot_graph(last_index_df, final_df, forecast, graphtype, Record_no, "Weekly")

                                              )
                                ]
                            )
                        ]
                    )
                ],
                style={
                    "marginTop": "20px"
                }
            )

            fig_components = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Forecasting Components")),
                                    dcc.Graph(figure=plotly_fig_components)
                                    #
                                    #           )
                                ]
                            )
                        ]
                    )
                ],
                style={
                    "marginTop": "20px"
                }
            )
            return figData, fig_components

        elif practiceList == "OSD01" and periodList == "Monthly" and timeFrame in range(1, 25):
            df = pd.read_csv( r"data/forecastdata.csv" )
            last_index_df, final_df, forecast = future_dates_test(df,timeFrame, 'm', load_model_monthly(), "Monthly")
            plotly_fig_components = tls.mpl_to_plotly(load_model_monthly().plot_components(forecast, figsize=(10, 10)))
            figData = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Forecasting Graph")),
                                    dcc.Graph(figure=plot_graph(last_index_df, final_df, forecast, graphtype, Record_no, "Monthly")

                                              )
                                ]
                            )
                        ]
                    )
                ],
                style={
                    "marginTop": "20px"
                }
            )
            fig_components = dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Forecasting Components")),
                                    dcc.Graph(figure=plotly_fig_components)
                                    #
                                    #           )
                                ]
                            )
                        ]
                    )
                ],
                style={
                    "marginTop": "20px"
                }
            )
            return figData, fig_components

        # elif practiceList == "MHP-BIOTECH01" and periodList == "Weekly" and timeFrame in range(1, 51):
        #     df = pd.read_csv(r"MHPBIOTECH01.csv")
        #     last_index_df, final_df, forecast = future_dates_test(df, timeFrame, 'w', load_model_weekly(), "Weekly")
        #     plotly_fig_components = tls.mpl_to_plotly(load_model_weekly().plot_components(forecast, figsize=(10, 10), ))
        #
        #     figData = dbc.Row(
        #         [
        #             dbc.Col(
        #                 [
        #                     dbc.Card(
        #                         [
        #                             dbc.CardHeader(html.H5("Forecasting Graph")),
        #                             dcc.Graph(figure=plot_graph(last_index_df, final_df, forecast, graphtype, Record_no,
        #                                                         "Weekly")
        #
        #                                       )
        #                         ]
        #                     )
        #                 ]
        #             )
        #         ],
        #         style={
        #             "marginTop": "20px"
        #         }
        #     )
        #
        #     fig_components = dbc.Row(
        #         [
        #             dbc.Col(
        #                 [
        #                     dbc.Card(
        #                         [
        #                             dbc.CardHeader(html.H5("Forecasting Components")),
        #                             dcc.Graph(figure=plotly_fig_components
        #
        #                                       )
        #                         ]
        #                     )
        #                 ]
        #             )
        #         ],
        #         style={
        #             "marginTop": "20px"
        #         }
        #     )
        #     return figData, fig_components
        # elif practiceList == "MHP-BIOTECH01" and periodList == "Monthly" and timeFrame in range(1, 25):
        #     df = pd.read_csv(r"MHPBIOTECH01.csv")
        #     last_index_df, final_df, forecast = future_dates_test(df, timeFrame, 'm', load_model_monthly(), "Monthly")
        #     plotly_fig_components = tls.mpl_to_plotly(load_model_monthly().plot_components(forecast, figsize=(10, 10)))
        #     figData = dbc.Row(
        #         [
        #             dbc.Col(
        #                 [
        #                     dbc.Card(
        #                         [
        #                             dbc.CardHeader(html.H5("Forecasting Graph")),
        #                             dcc.Graph(figure=plot_graph(last_index_df, final_df, forecast, graphtype, Record_no,
        #                                                         "Monthly")
        #
        #                                       )
        #                         ]
        #                     )
        #                 ]
        #             )
        #         ],
        #         style={
        #             "marginTop": "20px"
        #         }
        #     )
        #     fig_components = dbc.Row(
        #         [
        #             dbc.Col(
        #                 [
        #                     dbc.Card(
        #                         [
        #                             dbc.CardHeader(html.H5("Forecasting Components")),
        #                             dcc.Graph(figure=plotly_fig_components
        #
        #                                       )
        #                         ]
        #                     )
        #                 ]
        #             )
        #         ],
        #         style={
        #             "marginTop": "20px"
        #         }
        #     )
        #     return figData, fig_components
    else:
        return None, None



# @app.callback(
#     Output('lineplot','figure'),
#     Input("ShowForecasting", "n_clicks"),
# )
#
# def updateplot(showbutton):
#     changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
#
#     if "ShowForecasting" in changed_id:
#         df=pd.read_csv('data/forecastdata.csv')
#         df['closedate'] = pd.DatetimeIndex( df['closedate'] )
#         fig = px.histogram( df, x="closedate", y="NC", histfunc="avg", title="Histogram on Date Axes" )
#         fig.update_traces( xbins_size="M1" )
#         fig.update_xaxes( showgrid=True, ticklabelmode="period", dtick="M1", tickformat="%b\n%Y" )
#         fig.update_layout( bargap=0.1 )
#         fig.add_trace( go.Scatter( mode="markers", x=df["closedate"], y=df["NC"], name="daily" ) )
#         return fig




if __name__ == '__main__':
    app.run_server(debug=True,port=2020)
