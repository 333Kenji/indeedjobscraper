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
    ### changed from app/
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

def state_totals():
    df = buildout()
    totals = df.groupby(["State"])[["Q1",'Q2','Q3','Q4']].sum()[1:]
    def count_check(row):
        return row[0:].sum()
    totals['count'] = totals.apply( lambda row : count_check(row), axis = 1)
    r = totals.loc['Remote']
    us = totals.loc['United States']
    
    nonstate = [r, us]
    totals.drop(index=['Remote','United States'], inplace=True)
    totals = totals.append(nonstate)
    
    def Q1p(row):
        return str(int(round(row[0]/row['count'],2)*100))+'%'
    totals['Q1p'] = totals.apply( lambda row : Q1p(row), axis = 1)
    
    def Q2p(row):
        return str(int(round(row[1]/row['count'],2)*100))+'%'
    totals['Q2p'] = totals.apply( lambda row : Q2p(row), axis = 1)

    def Q3p(row):
        return str(int(round(row[2]/row['count'],2)*100))+'%'
    totals['Q3p'] = totals.apply( lambda row : Q3p(row), axis = 1)
    
    def Q4p(row):
        return str(int(round(row[3]/row['count'],2)*100))+'%'
    totals['Q4p'] = totals.apply( lambda row : Q4p(row), axis = 1)
    
    
    return totals

def q(row):
    if row['Q1'] == 1.0:
        return 'Q1 <br> 0 - 25%'
    if row['Q2'] == 1.0:
        return 'Q2 <br> 26% - 50%'
    if row['Q3'] == 1.0:
        return 'Q3 <br> 51% - 75%'
    if row['Q4'] == 1.0:
        return 'Q4 <br> 76% - 100%'

def numeriQ(row):
    if row['Q1'] == 1.0:
        return 1
    if row['Q2'] == 1.0:
        return 1
    if row['Q3'] == 1.0:
        return 1
    if row['Q4'] == 1.0:
        return 1
    

def create_cumsum_plot():
    df = buildout()
    Q1 = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q1'][1:]
    Q2 = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q2'][1:]
    Q3 = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q3'][1:]
    Q4 = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['Q4'][1:]
    #count = df.groupby(pd.Grouper(key="DatePosted", freq="1d")).sum()['count'][1:]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=Q1.index, y=Q1.values, name="Q1",
                         mode='lines', marker=dict(size=3, color='#ffc107'), opacity=.7))
    
    fig.add_trace(go.Scatter(x=Q2.index, y=Q2.values, name="Q2",
                         mode='lines', marker=dict(size=3, color='#ffc107'), opacity=.7))
   
    fig.add_trace(go.Scatter(x=Q3.index, y=Q3.values, name="Q3",
                         mode='lines', marker=dict(size=3, color='#000000'), opacity=.7))
    
    fig.add_trace(go.Scatter(x=Q4.index, y=Q4.values, name="Q4",
                         mode='lines', marker=dict(size=3, color='#ffc107'), opacity=.7))
   

    #fig.add_trace(go.Scatter(x=count.index, y=count.values, name="Number of All Postings",
                         ##mode='lines', marker=dict(size=4, color='#7f7f7f'), opacity=.7))

    fig.write_html('app/static/images/cumsum.html',
                   full_html=False,
                   include_plotlyjs='cdn'
                   )
    
def create_bar():
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

    layout = go.Layout(title='Indeed.com Postings By State Broken Down By Quartile', barmode='stack')
    figure = go.Figure(data=data, layout=layout)
    figure.write_html('app/static/images/bar.html',
                   full_html=False,
                   include_plotlyjs='cdn'
                   )
    
    
    
def create_usmap():
    df = state_totals()
    for col in df.columns:
        df[col] = df[col].astype(str)

    df['text'] = 'Q1 ' + df['Q1p'] + '<br>' + \
        'Q2 ' + df['Q2p'] + '<br>' + \
        'Q3 ' + df['Q3p'] + '<br>' + \
        'Q4 ' + df['Q4p'] + '<br>' + \
        'total ' + df['count']

    fig = go.Figure(data=go.Choropleth(
        locations=df.index,
        z=df['count'].astype(float),
        locationmode='USA-states',
        colorscale='Blues',
        autocolorscale=False,
        text=df['text'], # hover text
        marker_line_color='white', # line markers between states
        colorbar_title="Total Postings"
    ))

    fig.update_layout(
        title_text='April-May, 2021 Indeed.com Job Postings By State <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=go.layout.geo.Projection(type = 'albers usa'),
            showlakes=True, # lakes
            lakecolor='rgb(255, 255, 255)'),
    )

    fig.write_html('app/static/images/map.html',
                    full_html=False,
                    include_plotlyjs='cdn'
                    )

def create_sunburst():
    df = buildout()
    df['Q'] = df.apply( lambda row : q(row), axis = 1)
    df['nQ'] = df.apply( lambda row : numeriQ(row), axis = 1)
    fig = px.sunburst(df, path=['Q','State'], values='nQ', title='Salary Range Compositions')
    fig.write_html('app/static/images/sunburst.html',
                    full_html=False,
                    include_plotlyjs='cdn'
                    )
def create_table():
    df = buildout()
    df['links'] = '[n clicked](' + str(df['JobUrl']) + ')'
    df['Q'] = df.apply( lambda row : q(row), axis = 1)
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(['DatePosted','State','City','JobTitle', 'Q', 'Job URL']),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.DatePosted, df.State, df.City, df.JobTitle, df.Q, df.JobUrl],
               fill_color='lavender',
               align='left'))
    ])
    fig.write_html('app/static/images/table.html',
                    full_html=False,
                    include_plotlyjs='cdn'
                    )