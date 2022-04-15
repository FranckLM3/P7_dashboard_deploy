import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import gc 

import plotly.graph_objects as go
import pickle
import shap
shap.initjs()

#----------------------------------------------------------------------------------#
#                                 PREPROCESSING                                    #
#----------------------------------------------------------------------------------#

def plot_gauge(prediction_default):
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = prediction_default,
    mode = "gauge+number",
    title = {'text': "Risk of default (%)"},
    gauge = {'axis': {'range': [None, 100],
                    'tick0': 0,
                    'dtick':10},
            'bar': {'color': 'blue',
                    'thickness': 0.3,
                    'line': {'color': 'black'}},
            'steps' : [{'range': [0, 29.8], 'color': "green"},
                        {'range': [30.2, 49.8], 'color': "orange"},
                        {'range': [50.2, 100], 'color': "red"}]}        
                        ))

    fig_gauge.update_layout(width=600, 
                            height=400,
                            margin= {'l': 20, 'r': 20, 'b': 20, 't':20})

    return fig_gauge