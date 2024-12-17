import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d

# データの作成（例として、簡単な放物線）
x = np.linspace(0, 10, 10)
y = np.sin(x)

# 補完処理（間隔を補完する）
# 線形補完を使用して間隔を埋める
f = interp1d(x, y, kind='cubic')  # cubic補完
x_new = np.linspace(x.min(), x.max(), 200)  # 新しいx値を200個作成
y_new = f(x_new)  # 新しいy値を計算

# 色のグラデーション（青からオレンジ）
colors = np.linspace(0, 1, len(x_new))  # 0 から 1 の範囲で色を変化させる

# 補完した線とマーカーの描画
trace = go.Scatter(
    x=x_new,
    y=y_new,
    mode='lines+markers',  # 線とマーカーを両方表示
    marker=dict(
        color=colors,  # 各点に色を割り当て
        colorscale='YlOrRd',  # 青からオレンジへのカラースケール
        size=8,  # マーカーのサイズ
        colorbar=dict(title='Color scale')  # カラーバーを追加
    ),
    line=dict(
        color='gray',  # 線の色は固定（灰色）
        width=2
    )
)

# レイアウト設定
layout = go.Layout(
    title="Progressive Color Markers with Interpolation",
    xaxis=dict(title="X軸"),
    yaxis=dict(title="Y軸"),
)

# 図を作成して表示
fig = go.Figure(data=[trace], layout=layout)
fig.show()
