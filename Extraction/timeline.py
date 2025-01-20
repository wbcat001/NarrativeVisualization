
from typing import List, Optional
import re
import pandas as pd
import random
import plotly.express as px
from datetime import datetime, timedelta


def convert_to_datetime(x, delta=0):
  result = datetime(1971, 1, 1) + timedelta(days=int(x)) + timedelta(days=delta)

  return result.strftime("%Y-%m-%d")


shared_layout = {
    "margin": {"l": 30, "r": 30, "t": 5, "b": 5},  # 左右の余白を固定
    "legend": {"x": 1.1, "y": 1.02, "xanchor":"left", "yanchor": "bottom","orientation" : "h"},  # 凡例を右揃え
    # "xaxis": {"title": "Time", "tickangle": -45, "tickfont": {"size": 12}},  # x軸設定
    # "yaxis": {"title": "Location / Arousal", "tickfont": {"size": 12}},  # y軸設定
    "xaxis": {"domain": [0.1, 0.6], "showline": True, "linewidth": 2, "linecolor":"black", "range":[min, max]},
    "yaxis": {"showline": True, "linewidth": 2, "linecolor":"black"},
    "showlegend": False,
    # "xaxis": {"domain": [0.1, 0.9]},

    "plot_bgcolor": "white",  # 背景色を統一
    

}
####
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

    fig_location = px.timeline(_df, x_start="Start", x_end="Finish", y="Attribute", color="Attribute")

    # calc min and max for x-range
    min = _df["Start"].min()
    max = _df["Finish"].max()


    shared_layout = {
    "margin": {"l": 30, "r": 30, "t": 5, "b": 5},  # 左右の余白を固定
    "legend": {"x": 1.1, "y": 1.02, "xanchor":"left", "yanchor": "bottom","orientation" : "h"},  # 凡例を右揃え
    # "xaxis": {"title": "Time", "tickangle": -45, "tickfont": {"size": 12}},  # x軸設定
    # "yaxis": {"title": "Location / Arousal", "tickfont": {"size": 12}},  # y軸設定
    "xaxis": {"domain": [0.1, 0.6], "showline": True, "linewidth": 2, "linecolor":"black", "range":[min, max]},
    "yaxis": {"showline": True, "linewidth": 2, "linecolor":"black"},
    "showlegend": False,
    # "xaxis": {"domain": [0.1, 0.9]},

    "plot_bgcolor": "white",  # 背景色を統一
    
}
    fig_location.update_layout(**shared_layout)
    fig_location.show()

if __name__ == "__main__":
    file_path = "data/paragraph_alice.csv"
    df = pd.read_csv(file_path, index_col=0)
    draw_timeline(df,"Character")