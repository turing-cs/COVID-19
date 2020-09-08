import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from SIR_methods import SIR_modelling

import plotly.graph_objects as go
from scipy import optimize
from scipy import integrate

import os
print(os.getcwd())
df_analyse = pd.read_csv('../../data/processed/COVID_final_set.csv', sep = ';')

fig = go.Figure()
app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''

    # Implmentation of SIR model for Germany

    '''),


    dcc.Dropdown(
        id = 'country_drop_down',
        options=[ {'label': each,'value':each} for each in df_analyse['country'].unique()],
        value= 'Germany', # which are pre-selected
        multi=False),

    dcc.Graph(figure = fig, id = 'SIR_graph')
    ])

def SIR(countries):

    SIR_modelling()


@app.callback(
    Output('SIR_graph', 'figure'),
    [Input('country_drop_down', 'value')])

def update_SIR_figure(country_drop_down):

    traces = []

    df_plot = df_analyse[df_analyse['country'] == country_drop_down]
    df_plot = df_plot[['state', 'country', 'confirmed', 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
    df_plot.sort_values('date', ascending = True).head()
    df_plot = df_plot.confirmed[35:]

    t, fitted = SIR_modelling(df_plot)

    traces.append(dict (x = t,
                        y = fitted,
                        mode = 'markers',
                        opacity = 0.9,
                        name = 'SIR-fit')
                  )

    traces.append(dict (x = t,
                        y = df_plot,
                        mode = 'lines',
                        opacity = 0.9,
                        name = 'Original Data')
                  )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,
                title = 'SIR model fitting',

                xaxis= {'title':'Days',
                       'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis={'title': "Infected population"}
        )
    }


if __name__ == '__main__':
    app.run_server(debug = True, use_reloader = False)
