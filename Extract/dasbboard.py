from typing import List, Optional
import re
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
import dash
from dash import html, dcc, Input, Output, State

import dash_bootstrap_components as dbc

from sentiment import SentimentCalculator
from labMTsimple.storyLab import emotionFileReader


file_path = "data/paragraph_alice.csv"
df = pd.read_csv(file_path, index_col=0)
x_range = [0, len(df)-1]
sentiment_calculator = SentimentCalculator()
df = pd.read_csv(file_path, index_col=0)
df["Sentiment"] = df["Content"].apply(sentiment_calculator.calculate_valence)
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

def arrange_layout(fig, x_range, x_type=None):

    if x_type == "datetime":
        x_range = list(map(convert_to_datetime, x_range))
    shared_layout = {
    #　マージン
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0}, 
    # 凡例
    "legend": {"x": 1.1, "y": 1.02, "xanchor":"left", "yanchor": "bottom","orientation" : "h"}, 
    # "xaxis": {"title": "Time", "tickangle": -45, "tickfont": {"size": 12}},  # x軸設定
    # "yaxis": {"title": "Location / Arousal", "tickfont": {"size": 12}},  
    # x-axis
    "xaxis": {"domain": [0.1, 0.95], "showline": True, "linewidth": 2, "linecolor":"black", "range":x_range}, # datetime or index

    # y-axis
    "yaxis": {"showline": True, "linewidth": 2, "linecolor":"black"},
    "showlegend": False,

    "height": 200,
    

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
    if selected:
        _df = _df[_df["Location"].isin(selected)]
    fig_location, fig_time, fig_locationtype = draw_timeline(_df, "Location"), draw_timeline(_df, "Time"), draw_timeline(_df, "LocationType")
    
    return fig_location, fig_time, fig_locationtype
# Location
def draw_location(df, selected=None):
    _df = df.copy()
    if selected:
        _df = _df[_df["Location"].isin(selected)]
    fig = draw_timeline(_df, "Location")
    return fig

# Time
def draw_time(df, selected=None):
    _df = df.copy()
    if selected:
        _df = _df[_df["Time"].isin(selected)]
    fig = draw_timeline(_df, "Time")
    return fig

# Location Type
def draw_locationtype(df, selected=None):
    _df = df.copy()
    if selected:
        _df = _df[_df["LocationType"].isin(selected)]
    fig = draw_timeline(_df, attribute_name="LocationType")
    return fig

## Event
def draw_event(df, selected=None):
    _df = df.copy()
    _df = _df.dropna(subset=["Event", "EImportance"])

    if selected:
        pass
    _df["Group"] = (_df["Event"] != _df["Event"].shift()).cumsum()  # 連続する値をグループ化
    grouped_df = (
        _df.groupby("Group")
        .agg(
            Start=("Index", "min"),
            Finish=("Index", "max"),
            Event=("Event", "first"),
            EImportance=("EImportance", "max")  # 最大の重要度を使用
        )
        .reset_index(drop=True)
    )
    print(grouped_df)
    fig = go.Figure()
    for _, row in grouped_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Start"]],  # イベント名をX軸に
                y=[row["EImportance"]],  # 重要度をY軸に
                width=  [(row["Finish"] - row["Start"])/3],  # 棒の幅をインデックス範囲に基づいて設定
                name=f"{row['Event']} ({row['Start']}-{row['Finish']})",  # 凡例に範囲を表示
                marker=dict(
                    color="blue" if row["Event"] == "A" else "green" if row["Event"] == "B" else "red",
                    opacity=0.7
                )
            )
        )
    fig.update_layout(yaxis=dict(visible=False))
    return fig


## PoV
def draw_Pov(df):
    pass

def draw_Tone(df, selected=None):
    _df = df.copy()
    _df["Sentiment_mean"] =  df["Sentiment"].dropna().rolling(window=20, min_periods=1).mean()
    _df["customdata"] = _df["Event"]
    
    fig = px.line(_df, y="Sentiment_mean", hover_data=["Event"])
    return fig

    

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    fluid=True,
    children=[
    dbc.Row(children=[
        dbc.Col(
            width=2,
            children=[
                html.H5("facet bar"),
                html.Label("Character"),
                dcc.Checklist(
                    id="character-filter",
                    options=[{"label": char, "value": char} for char in set(sublist for sublist in df["Character"].dropna()) ],
                    value=[],
                    inline=True,),
                html.Label("Time Range"),
                dcc.RangeSlider(
                    id="index-filter",
                    min=df["Index"].min(),
                    max=df["Index"].max(),
                    step=1,
                    marks={i: str(i) for i in range(df["Index"].min(), df["Index"].max() + 1, 50)},
                    value=[df["Index"].min(), df["Index"].max()],
                ),
                html.Label("Location"),
                dcc.Checklist(
                    id="location-filter",
                    options=[{"label": loc, "value": loc} for loc in df["Location"].dropna().unique()],
                    value=[],
                    inline=True,
                ),
                html.Label("LocationType"),
                dcc.Checklist(
                    id="locationtype-filter",
                    options=[{"label": loc, "value": loc} for loc in df["LocationType"].dropna().unique()],
                    value=[],
                    inline=True,
                ),
                html.Label("Time"),
                dcc.Checklist(
                    id="time-filter",
                    options=[{"label": loc, "value": loc} for loc in df["Time"].dropna().unique()],
                    value=[],
                    inline=True,
                ),
                # html.Label("Day/Night"),
                # dcc.RadioItems(
                #     id="day-night-filter",
                #     options=[
                #         {"label": "Day", "value": "day"},
                #         {"label": "Night", "value": "night"},
                #         {"label": "Both", "value": "both"},
                #     ],
                #     value="both",
                # ),
                html.Label("Sentiment Threshold"),
                dcc.RangeSlider(
                    id="sentiment-filter",
                    min=df["Sentiment"].min(),
                    max=df["Sentiment"].max(),
                    step=1,
                    marks={i: str(i) for i in range(int(df["Sentiment"].min()), int(df["Sentiment"].max()) + 1, 5)},
                    value=[df["Sentiment"].min(), df["Sentiment"].max()],
                ),
           ],
       ),
        ## main view
        dbc.Col(
            width = 10,
            children=[
                html.H3("visalization", className="text-center"),
                dcc.Graph(id="event"),
                dcc.Graph(id="tone"),
                dcc.Graph(id="location"),
                dcc.Graph(id="time"),
                dcc.Graph(id="location-type"),
            ]
        )
       


   ])
])

## Time

## Character
@app.callback(
    Output("character", "figure"),
    [Input("character-filter","value")],
    State("range-slider", "value")
)
def udpate_character(click, value):
    # fig_list = draw_character(df)
    
    return go.Figure()

## Location callback
@app.callback(
    Output("location", "figure"),
    [Input("location-filter","value"),
    Input("index-filter", "value")],
   
)
def udpate_location(filter, index_filter):
    fig = draw_location(df, filter)
    arrange_layout(fig, index_filter, x_type="datetime")
    fig.update_layout({"height": 350})
    return fig

## Time callback
@app.callback(
    Output("time", "figure"),
    [Input("time-filter", "value"),
    Input("index-filter", "value")],
    
)
def udpate_time(filter, index_filter):
    
    fig = draw_time(df, filter)
    arrange_layout(fig, index_filter, x_type="datetime")
    return fig

## LocationType callback
@app.callback(
    Output("location-type", "figure"),
    [Input("locationtype-filter", "value"),
    Input("index-filter", "value")],
   
)
def update_locationtype(filter, index_filter):
    fig = draw_locationtype(df, filter)
    arrange_layout(fig, index_filter, x_type="datetime")

    return fig

## Sentiment/Tone
@app.callback(
    Output("tone", "figure"),
    [Input("sentiment-filter","value"),
      Input("index-filter", "value")],
   
)
def udpate_tone(filter, index_filter):
    # fig_list = draw_character(df)
    fig = draw_Tone(df)
    arrange_layout(fig, index_filter)
    return fig

@app.callback(
    Output("event", "figure"),
    [
      Input("index-filter", "value")],
   
)
def udpate_event( index_filter):
    
    fig = draw_event(df)
    arrange_layout(fig, index_filter)
    return fig

# @app.callback(
#     Output("facet-plot", "figure"),
#     [Input("category-dropdown", "value")]
# )





if __name__ == "__main__":
    app.run_server(debug=True)