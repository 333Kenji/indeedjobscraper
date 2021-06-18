import numpy as np
import pandas as pd

import re
import html
import requests

from datetime import datetime, date
import plotly.express as px
import plotly.graph_objs as go



def Q1(row):
    if (row['Q1'] == 1.0) or (row['Q1_Pred'] == 1.0):
        return 1.0
    else:
        return 0.0
    
def Q2(row):
    if (row['Q2'] == 1.0) or (row['Q2_Pred'] == 1.0):
        return 1.0
    else:
        return 0.0

def Q3(row):
    if (row['Q3'] == 1.0) or (row['Q3_Pred'] == 1.0):
        return 1.0
    else:
        return 0.0
    
def Q4(row):
    if (row['Q4'] == 1.0) or (row['Q4_Pred'] == 1.0):
        return 1.0
    else:
        return 0.0




def q(row):
    if row['Q1_fin'] == 1.0:
        return 'Q1'
    if row['Q2_fin'] == 1.0:
        return 'Q2'
    if row['Q3_fin'] == 1.0:
        return 'Q3'
    if row['Q4_fin'] == 1.0:
        return 'Q4'
    else:
        return 'unk'

def p(row):
    if row['Q1_fin'] == 1.0:
        if row['Q1_Prob'] > 0.0:
            return row['Q1_Prob']
        else:
            return 'Given'
    if row['Q2_fin'] == 1.0:
        if row['Q2_Prob'] > 0.0:
            return row['Q2_Prob']
        else:
            return 'Given'
    if row['Q3_fin'] == 1.0:
        if row['Q3_Prob'] > 0.0:
            return row['Q3_Prob']
        else:
            return 'Given'
    if row['Q4_fin'] == 1.0:
        if row['Q4_Prob'] > 0.0:
            return row['Q4_Prob']
        else:
            return 'Given'
    else:
        return 'unk'

def unk(row):
    if row['Q'] == 'unk':
        return 1.0
    else:
        return 0.0


def buildout():
    """[Provides a final set of features showing which salary range applies to each data.]

    Returns:
        [csv]: [a table containing our original original, predicted, and bracket columns]
    """
    ### changed from app/
    df = pd.read_csv('../data/ml.csv')
    df['DatePosted'] = df['DatePosted'].astype('datetime64')
    df = df.sort_values(by="DatePosted")


    df.replace(np.nan, 0, inplace = True)
    df['Q1_fin'] = df.apply( lambda row : Q1(row), axis = 1)
    df['Q2_fin'] = df.apply( lambda row : Q2(row), axis = 1)
    df['Q3_fin'] = df.apply( lambda row : Q3(row), axis = 1)
    df['Q4_fin'] = df.apply( lambda row : Q4(row), axis = 1)
    df['Q'] = df.apply( lambda row : q(row), axis = 1)
    df['P'] = df.apply( lambda row : p(row), axis = 1)
    df['unk'] = df.apply( lambda row : unk(row), axis = 1)
    
    df.to_csv(f'../data/graph_ready.csv', index=False)
    return df