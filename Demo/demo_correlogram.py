import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import copy
import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')

df_table = ff.create_table(df.head())
py.iplot(df_table, filename='iris-data-head')

classes=np.unique(df['class'].values).tolist()
# classes

class_code={classes[k]: k for k in range(3)}
# class_code

color_vals=[class_code[cl] for cl in df['class']]

pl_colorscale=[[0.0, '#19d3f3'],
               [0.333, '#19d3f3'],
               [0.333, '#e763fa'],
               [0.666, '#e763fa'],
               [0.666, '#636efa'],
               [1, '#636efa']]

text=[df.loc[ k, 'class'] for k in range(len(df))]


trace1 = go.Splom(dimensions=[dict(label='sepal length',
                                 values=df['sepal length']),
                            dict(label='sepal width',
                                 values=df['sepal width']),
                            dict(label='petal length',
                                 values=df['petal length']),
                            dict(label='petal width',
                                 values=df['petal width'])],
                text=text,
                #default axes name assignment :
                #xaxes= ['x1','x2',  'x3'],
                #yaxes=  ['y1', 'y2', 'y3'], 
                marker=dict(color=color_vals,
                            size=7,
                            colorscale=pl_colorscale,
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
                )

axis = dict(showline=True,
          zeroline=False,
          gridcolor='#fff',
          ticklen=4)

layout = go.Layout(
    title='Iris Data set',
    dragmode='select',
    width=600,
    height=600,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    xaxis4=dict(axis),
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis),
    yaxis4=dict(axis)
)

fig1 = dict(data=[trace1], layout=layout)
py.iplot(fig1, filename='splom-iris1')                               






trace3 = copy.deepcopy(trace1)
trace3['showupperhalf']=False

fig3 = dict(data=[trace3], layout=layout)
py.iplot(fig3, filename='splom-showupperhalf')

