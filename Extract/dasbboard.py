from typing import List, Optional
import re
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State,ClientsideFunction
import dash_bootstrap_components as dbc

from sentiment import SentimentCalculator
from labMTsimple.storyLab import emotionFileReader


file_path = "data/paragraph_alice.csv"
df = pd.read_csv(file_path, index_col=0)
x_range = [0, len(df)-1]
sentiment_calculator = SentimentCalculator()
df = pd.read_csv(file_path, index_col=0)
df["Sentiment"] = df["Content"].apply(sentiment_calculator.calculate_labMT)
#### Figureの作成用　
## draw timeline



def convert_to_datetime(x, delta=0):
  result = datetime(1971, 1, 1) + timedelta(days=int(x)) + timedelta(days=delta)

  return result.strftime("%Y-%m-%d")

def draw_timeline(df, attribute_name = "Location"): 

    _df = df.copy()
    _df["Attribute"] = _df[attribute_name]


    ## Common process
    _df = _df.dropna(subset=["Attribute"])
    grouped = (_df["Attribute"] != _df["Attribute"].shift()).cumsum()

    ## この部分はアトリビュートによって変える必要がある
    _df = _df.groupby(grouped).agg({
    "Attribute": "first",            
    "Index": ["first", "last"]
    }).reset_index(drop=True)

    _df.columns = ["Attribute", "Start", "Finish"]
    
    _df["Start"] = _df["Start"].apply(convert_to_datetime)
    _df["Finish"] = _df["Finish"].apply(convert_to_datetime, delta=1) # 1列だけの要素を表示するため
    print(_df)
    fig = px.timeline(_df, x_start="Start", x_end="Finish", y="Attribute", color="Attribute")
    
    return fig

def arrange_layout(fig, x_range):
    shared_layout = {
    "margin": {"l": 30, "r": 30, "t": 5, "b": 5},  # 左右の余白を固定
    "legend": {"x": 1.1, "y": 1.02, "xanchor":"left", "yanchor": "bottom","orientation" : "h"},  # 凡例を右揃え
    # "xaxis": {"title": "Time", "tickangle": -45, "tickfont": {"size": 12}},  # x軸設定
    # "yaxis": {"title": "Location / Arousal", "tickfont": {"size": 12}},  # y軸設定
    "xaxis": {"domain": [0.1, 0.6], "showline": True, "linewidth": 2, "linecolor":"black", "range":x_range}, # datetime or index
    "yaxis": {"showline": True, "linewidth": 2, "linecolor":"black"},
    "showlegend": False,
    # "xaxis": {"domain": [0.1, 0.9]},

    "plot_bgcolor": "white",  # 背景色を統一
    }
   
    fig.update_layout(**shared_layout)


## Character

def draw_character(df, selected=None):
    _df = df.copy()
    if selected:
        pass
    
    unique_values = set(val for sublist in _df["Character"] for val in sublist)

    for value in unique_values:
        _df[value] = _df["Character"].apply(lambda x: value in x)
    fig_list = []
    for value in unique_values:
        fig_list(draw_timeline(_df, attribute_name=value))

    return fig_list

## Setting(Location, Time, )

def draw_setting(df, selected=None):
    _df = df.copy()

    fig_location, fig_time, fig_locationtype = draw_timeline(_df, "Location"), draw_timeline(_df, "Time"), draw_timeline(_df, "LocationType")
    
    return fig_location, fig_time, fig_locationtype
## Event



## PoV
def draw_Pov(df):
    pass

def draw_Tone(df, selected=None):
    _df = df.copy()
    _df["Sentiment_mean"] =  df["Sentiment"].dropna().rolling(window=50, min_periods=1).mean()
    df["customdata"] = df["Event"]
    
    fig = px.line(_df, y="Sentiment_mean", hover_data=["Event"])
    return fig

    

app = dash.Dash(__name__)

app.layout = html.Div([

    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button(id="tone-btn"), width=2), 
            dbc.Col(dbc.Button(id="character-btn")), 
            dbc.Col(dbc.Button(id="location-btn")), 
            dbc.Col(dbc.Button(id="time-btn")), 
            dbc.Col(dbc.Button(id="location-type-btn")), 
            
        ]),
        dbc.Row([dbc.Col(dcc.RangeSlider(0,100, 1, value=[0, 100], id='range-slider')),]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="tone")),           
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="character")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="location")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="time")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="location-type")),
        ]),
    ])
])

@app.callback(
    Output("character", "figure"),
    [Input("character-btn","n_clicks")],
    State("range-slider", "value")
)
def udpate_character(click, value):
    # fig_list = draw_character(df)
    
    return go.Figure()

@app.callback(
    Output("location", "figure"),
    Output("time", "figure"),
    Output("location-type", "figure"),
    [Input("location-btn","n_clicks")],
    State("range-slider", "value")
)
def udpate_setting(vlick, value):
    fig_list = draw_setting(df)
  
    # for fig in fig_list:
    #     arrange_layout(fig, value)
    return fig_list

@app.callback(
    Output("tone", "figure"),
    [Input("tone-btn","n_clicks")],
    State("range-slider", "value")
)
def udpate_tone(click, value):
    # fig_list = draw_character(df)
    fig = draw_Tone(df)
    return fig


# @app.callback(
#     Output("facet-plot", "figure"),
#     [Input("category-dropdown", "value")]
# )





if __name__ == "__main__":
    app.run_server(debug=True)