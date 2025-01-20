import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime as dt, timedelta
import pandas as pd

def convert_to_datetime(x, strftime_format = "%Y-%m-%d"):
    return dt.fromtimestamp(31536000+x*24*3600).strftime(strftime_format)

def create_gantt(subdf, Task_column, num_Start_column, num_Finish_column, color_list, tick_num = 3  ):
    subdf["Task"] = subdf[Task_column]
    subdf["Start"] = subdf[num_Start_column].apply(convert_to_datetime) 
    subdf["Finish"] = subdf[num_Finish_column].apply(convert_to_datetime)
     
    num_tick_labels = np.linspace(start = min(np.min(subdf[num_Start_column]),np.min(subdf[num_Finish_column])),
                                  stop = max(np.max(subdf[num_Start_column]),np.max(subdf[num_Finish_column])), num = tick_num, dtype = int)
    date_ticks = [convert_to_datetime(x) for x in num_tick_labels]
    px_fig = px.timeline(subdf,x_start="Start", x_end="Finish", y="Task", color_discrete_sequence=color_list , hover_data=[num_Start_column, num_Finish_column])
    for trace in range(len(px_fig["data"])):
        px_fig["data"][trace]["hovertemplate"] = 'Start=%{customdata[0]}<br>Finish=%{customdata[1]}'
    px_fig.layout.xaxis.update({
        'tickvals' : date_ticks,
        'ticktext' : num_tick_labels
        })
    
    return px_fig 

def embed_in_subplot(parent_fig, child_fig,  row, col ):
    for trace in range(len(child_fig["data"])):
      parent_fig.add_trace(child_fig["data"][trace],
                    row=row, col=col)
    parent_fig.update_xaxes(tickvals=child_fig.layout.xaxis.tickvals, ticktext=child_fig.layout.xaxis.ticktext, type=child_fig.layout.xaxis.type, row=row, col=col)
    parent_fig.update_yaxes(tickvals=child_fig.layout.yaxis.tickvals, ticktext=child_fig.layout.yaxis.ticktext, type=child_fig.layout.yaxis.type, row=row, col=col)

    return parent_fig

def test_gantt():
    df1 = [dict(Task="Job A", numStart=530 , numFinish=4456),
      dict(Task="Job B", numStart=343, numFinish=623),
      dict(Task="Job C", numStart=746, numFinish=1023),
      dict(Task="Job A", numStart=1000, numFinish=1023),
      dict(Task="Job B", numStart=800, numFinish=900),
      ]

   

    
    df1 = pd.DataFrame(df1)

    fig  = create_gantt(df1, 'Task', 'numStart', 'numFinish',   [px.colors.qualitative.Plotly[0]]* len(df1)  , tick_num = 5 )
    fig.show()
    
#### run test
test_gantt()  