
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: team 0                                                                                      -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://deepnote.com/workspace/pgomezman-fb44-564aea80-b395-487c-9bd5-0c594e3cbf9f/project/MyST-Lab3-6769db9b-c501-49f1-b3ff-df4df228bebc/%2Ffunctions.py                                                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import functions as fn
import plotly.express as px
import plotly.graph_objects as go

# --------------- Gráfica 1 -------------------
def fn_graph_rank(df):
    '''
    Piechart of a dataframe with largest value pulled out, input=dataframe
    '''
    # Ordenamos el dataframe de mayor a menor
    df=df.sort_values(by=['Rank'], ascending=False)
    
    # Calculamos el pull del sector con mejor ranking
    l=np.zeros(len(df))
    l[0]=0.2
    fig = go.Figure(data=[go.Pie(labels=df['Symbol'], values=df['Rank'], pull=l, title='Gráfica 1: Ranking')])
    return fig

# --------------- Gráfica 2 -------------------
def fn_capital_plot(p2,p4, C):
    '''
    Plot of capital evolution through time with DrawUp and DrawDown
    '''
    p2['Timestamp']= pd.to_datetime(p2['Timestamp'])
    # Data for lines
    # Drawdown
    xa=[2,3]
    datesdrawdown=pd.to_datetime(p4.iloc[xa,2])
    val_drawdown=C+float(p4.iloc[4,2])
    # Drawup
    xb=[5,6]
    datesdrawup=pd.to_datetime(p4.iloc[xb,2])
    val_drawup=C+float(p4.iloc[7,2])

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=p2['Timestamp'], y=p2['Profit_acm_d'], name='Capital',
                            line=dict(color='black', width=4)))
    fig.add_trace(go.Scatter(x=datesdrawdown, y=[val_drawdown, val_drawdown], name = 'Draw Down',
                            line=dict(color='red', width=4, dash='dash')))

    fig.add_trace(go.Scatter(x=datesdrawup, y=[val_drawup, val_drawup], name = 'Draw Up',
                            line=dict(color='green', width=4, dash='dash')))


    # Edit the layout
    fig.update_layout(title='Gráfica 2: DrawDown y DrawUp',
                    xaxis_title='Fecha',
                    yaxis_title='Capital')
    return fig

 #--------------- Gráfica 3 -------------------
    '''
    Plot of occurences of disposition effects
    '''

def fn_disposition(p3):
    #Transformaciones del dataframe
    d = {'status_quo': p3['status_quo'], 'aversion_perdida': p3['aversion_perdida'], 'sensibilidad_decreciente':0/p3['ocurrencias']}
    df = pd.DataFrame(data=d)
    d1=df.transpose()

    return px.bar(d1, x=d1.index, y=d1.iloc[:,0].values, color=['red', 'blue', 'green'],  title="Disposition effect",labels=dict(index="Effect", value="% of ocurrences"))