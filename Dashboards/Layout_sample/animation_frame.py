import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
# サンプルデータ
frames_data = [
    {
        "x": [j for j in range(10)],
        "y": [i+1 for _ in range(10)],  # フレームごとに位相をずらす
    
    }
    for i in range(3)
]
# 基本のフレーム（最初の状態）
base_frame = [go.Scatter(
    x=frames_data[0]["x"][:5],
    y=frames_data[0]["y"][:5],
    mode="markers+text+lines",
   
    marker=dict(size=5, color="blue")
),
go.Scatter(
    x=frames_data[0]["x"][5:],
    y=frames_data[0]["y"][5:],
    mode="markers+text+lines",
   
    marker=dict(size=5, color="blue")
)
]

# フレーム定義
frames = [
    go.Frame(
        data=[
            go.Scatter(
                x=frame["x"][:5],
                y=frame["y"][:5],
                mode="markers+lines",
                
                marker=dict(size=5, color="blue")
            ),
             go.Scatter(
                x=frame["x"][5:],
                y=frame["y"][5:],
                mode="markers+lines",
                
                marker=dict(size=5, color="blue")
            ),

        ],

    )
    for frame in frames_data
]

# Layoutの更新用設定
layout = go.Layout(
    xaxis=dict(range=[0, 15]),
    yaxis=dict(range=[0,5]),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
)

# アニメーションのFigure
fig = go.Figure(
    data=base_frame,
    layout=layout,
    frames=frames
)

# Dashアプリの構築
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
