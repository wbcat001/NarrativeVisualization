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

# from sentiment import SentimentCalculator
from labMTsimple.storyLab import emotionFileReader


file_path = "data/paragraph_alice.csv"
df = pd.read_csv(file_path, index_col=0)

def convert_to_datetime(x, delta=0):
  result = datetime(1971, 1, 1) + timedelta(days=int(x)) + timedelta(days=delta)

  return result.strftime("%Y-%m-%d")
def draw_timeline(df, attribute_name = "Location", custom_color=None): 

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
    _df["Finish"] = _df["Finish"].apply(convert_to_datetime, delta=2) # 1列だけの要素を表示するため
    print(_df)

    color = [custom_color[value] for value in _df["Attribute"]] if custom_color else {"Attribute":"#000000"}
    print(color)
    
    fig = px.timeline(_df, x_start="Start", x_end="Finish", y="Attribute", color="Attribute", color_discrete_map=custom_color)
    
    return fig

def draw_character(df, selected=None, color=None):
    _df = df.copy()
    if selected:
        pass
    
    fig = draw_timeline(df, "Character", color)

    return fig

uniques = df["Character"].unique()
random_colors = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in uniques}

fig = draw_character(df, color=random_colors)
# fig.show()
df["Sentiment"] = [random.randint(0, 10) for _ in range(len(df))]
df["Sentiment"] = df["Sentiment"].rolling(window=50, min_periods=1).mean()
fig2 = go.Figure(go.Scatter(
    x=df["Index"],
    y=df["Sentiment"],
    mode="lines+markers",
    name="Values",
    

))

# print(len(df.dropna(subset=["Character"])))


for idx, row in df.dropna(subset=["Character"]).iterrows():
    fig2.add_shape(
        type="line",
        x0=row["Index"], x1=row["Index"],
        y0=row["Sentiment"] - 0.1, y1=row["Sentiment"] + 0.1,
        line=dict(color=random_colors[row["Character"]], width=5),
        name=row["Character"]  # タグ名を利用
    )

fig2.show()