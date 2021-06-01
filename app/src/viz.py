import numpy as np
import pandas as pd

import re
import html
import requests

from datetime import datetime, date
import plotly.express as px
import plotly.graph_objs as go



def above_med(row):
    if (row['median_sal'] == 1.0) or (row['median_sal_prediction'] == 1.0):
        return 1.0
    else:
        return 0.0
    
def iqr(row):
    if (row['iqr'] == 1.0) or (row['iqr_prediction'] == 1.0):
        return 1.0
    else:
        return 0.0

def Q1(row):
    if (row['fin_above_med_sal'] == 0.0) and (row['fin_iqr'] == 0.0):
        return 1.0
    else:
        return 0.0

def Q2(row):
    if (row['fin_above_med_sal'] == 0.0) and (row['fin_iqr'] == 1.0):
        return 1.0
    else:
        return 0.0
    
def Q3(row):
    if (row['fin_above_med_sal'] == 1.0) and (row['fin_iqr'] == 1.0):
        return 1.0
    else:
        return 0.0

def Q4(row):
    if (row['fin_above_med_sal'] == 1.0) and (row['fin_iqr'] == 0.0):
        return 1.0
    else:
        return 0.0

def out(row):
    if row['Q1'] + row['Q2'] + row['Q3'] + row['Q4'] == 0.0:
        return 1.0
    else:
        return 0.0
    
def count_check(row):
    return row['Q1'] + row['Q2'] + row['Q3'] + row['Q4'] + row['out']


def buildout():
    """[Provides a final set of features showing which salary range applies to each data.]

    Returns:
        [csv]: [a table containing our original original, predicted, and bracket columns]
    """
    df = pd.read_csv('app/data/ml.csv', index_col=0)
    df['DatePosted'] = df['DatePosted'].astype('datetime64')
    df = df.sort_values(by="DatePosted")


    df.replace(np.nan, 0, inplace = True)
    df['fin_iqr'] = df.apply( lambda row : iqr(row), axis = 1)
    df['fin_above_med_sal'] = df.apply( lambda row : above_med(row), axis = 1)
    df['Q1'] = df.apply( lambda row : Q1(row), axis = 1)
    df['Q2'] = df.apply( lambda row : Q2(row), axis = 1)
    df['Q3'] = df.apply( lambda row : Q3(row), axis = 1)
    df['Q4'] = df.apply( lambda row : Q4(row), axis = 1)
    df['out'] = df.apply( lambda row : out(row), axis = 1)
    #df['check'] = df.apply( lambda row : count_check(row), axis = 1)
        #buildout(df)['check'].mean()
    #df.to_csv('graph_ready.csv')

    return df



def create_cumsum_plot():
    df = buildout()
    low_iqr = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q2'][1:]
    high_iqr = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q3'][1:]
    #count = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['count'][1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=low_iqr.index, y=low_iqr.values, name="Quartile Below Median",
                         mode='lines', marker=dict(size=3, color='#ffc107'), opacity=.7))
   
    fig.add_trace(go.Scatter(x=high_iqr.index, y=high_iqr.values, name="Quartile Above Median",
                         mode='lines', marker=dict(size=3, color='#000000'), opacity=.7))

    #fig.add_trace(go.Scatter(x=count.index, y=count.values, name="Number of All Postings",
                         ##mode='lines', marker=dict(size=4, color='#7f7f7f'), opacity=.7))

    fig.write_html('app/static/images/cumsum.html',
                   full_html=False,
                   include_plotlyjs='cdn'
                   )


def bar():
    df = buildout()
    Q1_ = df.groupby(["State"])["Q1"].sum()[1:]
    Q2_ = df.groupby(["State"])["Q2"].sum()[1:]
    Q3_ = df.groupby(["State"])["Q3"].sum()[1:]
    Q4_ = df.groupby(["State"])["Q4"].sum()[1:]
    t_bar = pd.DataFrame([Q1_, Q2_, Q3_, Q4_]).T

    
    Q1 = go.Bar(
    x = t_bar.index,
    y = t_bar.Q1.values,
    name='Q1')

    Q2 = go.Bar(
    x = t_bar.index,
    y = t_bar.Q2.values,
    name='Q2')

    Q3 = go.Bar(
    x = t_bar.index,
    y = t_bar.Q3.values,
    name='Q3')

    Q4 = go.Bar(
    x = t_bar.index,
    y = t_bar.Q4.values,
    name='Q4')
    
    data = [Q1, Q2, Q3, Q4]

    layout = go.Layout(title='Postings Within +/- 25% Median Salary', barmode='stack')
    figure = go.Figure(data=data, layout=layout)
    figure.write_html('app/static/images/bar.html',
                   full_html=False,
                   include_plotlyjs='cdn'
                   )