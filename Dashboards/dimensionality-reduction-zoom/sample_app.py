import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 仮のデータを作成
np.random.seed(42)

# from_data (初期位置)
from_data = pd.DataFrame({
    'x': np.random.rand(10),
    'y': np.random.rand(10),
    'text': [f"Point {i}" for i in range(10)],
})

from_data1 = from_data.iloc[:5, :]
from_data2 = from_data.iloc[5:, :]

# to_data (最終位置)
to_data = pd.DataFrame({
    'x': np.random.rand(10),
    'y': np.random.rand(10),
    'text': [f"Point {i} - End" for i in range(10)],
})

# to_data1, to_data2 (最終位置のさらに別の変化)
to_data1 = to_data.iloc[:5, :]

to_data2 = to_data.iloc[5:, :]

# アニメーションのためのフレーム設定
frames = [
    go.Frame(
        data=[go.Scatter(
            x=from_data['x'] + (to_data['x'] - from_data['x']) * i / 10,  # x座標の補完
            y=from_data['y'] + (to_data['y'] - from_data['y']) * i / 10,  # y座標の補完
            mode='markers+lines',
            line=dict(color='blue'),
            marker=dict(size=10, color='blue'),
            text=from_data['text'],
            hoverinfo="text",
            
        )],
        name=f'Frame {i}'
    ) for i in range(11)
]

# 初期状態 (from_data)
trace_from1 = go.Scatter(
    x=from_data1['x'],
    y=from_data1['y'],
    mode='markers+lines',
    line=dict(color='green'),
    marker=dict(size=10, color='green'),
    name="From Data",
    text=from_data1['text'],
    hoverinfo="text",
)

trace_from2 = go.Scatter(
    x=from_data2['x'],
    y=from_data2['y'],
    mode='markers+lines',
    line=dict(color='green'),
    marker=dict(size=10, color='green'),
    name="From Data",
    text=from_data2['text'],
    hoverinfo="text",
)



# to_data1 と to_data2 を追加（最初は非表示）
trace_to1 = go.Scatter(
    x=to_data1['x'],
    y=to_data1['y'],
    mode='markers+lines',
    line=dict(color='red'),
    marker=dict(size=10, color='red'),
    name="To Data 1",
    text=to_data1['text'],
    hoverinfo="text",
    visible=False
)

trace_to2 = go.Scatter(
    x=to_data2['x'],
    y=to_data2['y'],
    mode='markers+lines',
    line=dict(color='red'),
    marker=dict(size=10, color='red'),
    name="To Data 2",
    text=to_data2['text'],
    hoverinfo="text",
    visible=False
)

# レイアウト設定
layout = go.Layout(
    title="Scatter Animation: From -> To",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    updatemenus=[
        {
            'buttons': [
                # ボタン1: from_data1 と from_data2 を非表示
                {
                    'args': [{'visible': [True, False, False, False]}],
                    'label': 'Hide from_data1 & from_data2',
                    'method': 'restyle'
                },
                # ボタン2: from_data -> to_data のアニメーション
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Animate from_data -> to_data',
                    'method': 'animate'
                },
                # ボタン3: to_data1 と to_data2 を表示
                {
                    'args': [{'visible': [False, False, True, True]}],
                    'label': 'Show to_data1 & to_data2',
                    'method': 'restyle'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]
)

# グラフの作成
fig = go.Figure(
    data=[trace_from1, trace_from2, trace_to1 ,trace_to2],
    layout=layout,
    frames=frames
)

# 表示
fig.show()
