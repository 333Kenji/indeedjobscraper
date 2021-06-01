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
    df = pd.read_csv('../data/ml.csv', index_col=0)
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
    df.to_csv('../data/graph_ready.csv')

    return df