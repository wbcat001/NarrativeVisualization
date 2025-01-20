import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# 768次元の埋め込みデータを生成
np.random.seed(42)
data = np.random.rand(100, 768)

# PCAを適用して2次元に射影
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# 初期射影: データの最初の2次元を使用
original_projection = data[:, :2]
pca_projection = pca_result

# アニメーションデータの作成
frames = []
steps = 30
for i in range(steps + 1):
    t = i / steps
    interpolated = (1 - t) * original_projection + t * pca_projection
    frame = go.Frame(
        data=[go.Scatter(
            x=interpolated[:, 0],
            y=interpolated[:, 1],
            mode='markers',
            marker=dict(size=8, color=np.arange(interpolated.shape[0]), colorscale="Viridis")
        )],
        name=f'frame{i}'
    )
    frames.append(frame)

# 初期フレーム
initial_frame = go.Scatter(
    x=original_projection[:, 0],
    y=original_projection[:, 1],
    mode='markers',
    marker=dict(size=8, color=np.arange(original_projection.shape[0]), colorscale="Viridis")
)

# アニメーションの作成
fig = go.Figure(
    data=[initial_frame],
    layout=go.Layout(
        title="PCAによる2次元射影の更新",
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )]
    ),
    frames=frames
)

fig.show()
