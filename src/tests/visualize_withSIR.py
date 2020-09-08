import pandas as pd
import numpy as np
import subprocess
import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)


from scipy import signal
import os

from get_data import get_johns_hopkins
from process_JH_data import store_relational_JH_data
from get_pd_large import pd_result_large
from build_features import *


get_johns_hopkins()
store_relational_JH_data()
pd_result_large()

print(os.getcwd())
df_input_large=pd.read_csv('/home/mitesh/COVID19/EDS_COVID/data/processed/COVID_final_set.csv',sep=';')


fig = go.Figure()
figSIR = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Enterprise Data Science on John hopkins COVID-19 data

    Content and Summary of this project are mentioned below

    1. Automated data gathering
    2. Data transformations
    3. Filtering and machine learning to approximating the doubling time
    4. Deployment of responsive dashboard

    '''),

    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany','Italy'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),
html.Details(open=False,children=[
			html.Summary("Part 2: SIR model analysis and prediction"),
			html.Div(className="GUI_settings_div",children=[
				html.H2("Settings for Part 2"),
				html.Label('Please specify the observation window:',htmlFor='GUI_SIR_obs_DatesPicker'),
				dcc.DatePickerRange(
					id='GUI_SIR_obs_DatesPicker',
					start_date = GUI_buf['obsWinStart'],#dt.strptime(GUI_buf['obsWinStart'],"%Y-%m-%d").date(),
					end_date = GUI_buf['obsWinEnd'],#dt.strptime(GUI_buf['obsWinEnd'],"%Y-%m-%d").date(),
					min_date_allowed=dates_in_DataSets[0],#dt.strptime(dates_in_DataSets[0],"%Y-%m-%d"),
					max_date_allowed=dates_in_DataSets[-1],#dt.strptime(dates_in_DataSets[-1],"%Y-%m-%d"),
					minimum_nights = 6,
					display_format='MMM DD, YYYY',
					with_portal=True,
				),
				html.Details([
					html.Summary(["Settings specific to the SIR plot below"]),

					# html.Label('Display options:',htmlFor='GUI_SIR_show_opt_chkbox'),
					dcc.Checklist(
						options=[
							{'label': "content: Show SIR simultation from observation start to prediction end", 'value': "showSmooth_then_Pred"},
							{'label': "content: Show the derived Recovered/ Removed population/ Death (from measurement).", 'value': "show_SIR_R"},
							# {'label': "content: SIR prediction from observation end to prediction end", 'value': "showPredict_from_ObsEnd"},
							{'label': "Y axis: Show in percentage relative to nominal population", 'value':"showInPercent"},
							{'label': "X axis: Prevent auto resizing of X-axis", 'value':"stopAutoResizeX"}
						],
						id = "GUI_SIR_show_opt_chkbox",
						value=["showInPercent","showSmooth_then_Pred"], #
    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])


def update_figure(country_list,show_doubling):


    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }

# %load ../src/features/build_features.py




def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        Doubling rate: double
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           1)
    df_result[str(column+'_filtered')]=result
    return df_result

def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)



    return result




def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()

    #print('--+++ after group by apply')
    #print(pd_filtered_result[pd_filtered_result['country']=='Germany'].tail())

    #df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    #print(df_output[df_output['country']=='Germany'].tail())
    return df_output.copy()





def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output



if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)
