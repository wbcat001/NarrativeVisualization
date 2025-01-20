import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def normalize(v):
    """正規化"""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def update_projection(xk, e1, e2, e3, wk1, wk2, wk3, wk1_prime, wk2_prime, wk3_prime):
    """射影の更新

    Args:
        xk: ノード k の高次元位置ベクトル (numpy array)
        e1, e2, e3: 可視空間の基底ベクトル (numpy arrays)
        wk1, wk2, wk3: ノード k の移動前の座標
        wk1_prime, wk2_prime, wk3_prime: ノード k の移動後の座標

    Returns:
        新しい基底ベクトル (e0, e1_prime, e2_prime, e3_prime)
    """
    # 新しいベクトル f0 を計算
    f0 = xk - wk1 * e1 - wk2 * e2 - wk3 * e3
    
    # e0 を正規化
    e0 = normalize(f0)

    # 新しい基底ベクトルの線形結合係数 (仮に初期化)
    alpha = np.zeros((3, 4))

    # 制約解消: 回転移動による新しい基底ベクトル
    # ここではNumPyの線形方程式ソルバーを利用
    A = np.array([
        [np.linalg.norm(f0), wk1, wk2, wk3],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    b1 = np.array([wk1_prime, 0, 0, 0])
    b2 = np.array([wk2_prime, 0, 0, 0])
    b3 = np.array([wk3_prime, 0, 0, 0])

    alpha[0] = np.linalg.solve(A, b1)
    alpha[1] = np.linalg.solve(A, b2)
    alpha[2] = np.linalg.solve(A, b3)

    # 新しい基底ベクトルを計算
    e1_prime = alpha[0, 0] * e0 + alpha[0, 1] * e1 + alpha[0, 2] * e2 + alpha[0, 3] * e3
    e2_prime = alpha[1, 0] * e0 + alpha[1, 1] * e1 + alpha[1, 2] * e2 + alpha[1, 3] * e3
    e3_prime = alpha[2, 0] * e0 + alpha[2, 1] * e1 + alpha[2, 2] * e2 + alpha[2, 3] * e3

    # 正規化
    e1_prime = normalize(e1_prime)
    e2_prime = normalize(e2_prime)
    e3_prime = normalize(e3_prime)

    return e0, e1_prime, e2_prime, e3_prime

# 768次元の埋め込みデータを生成
np.random.seed(42)
data = np.random.rand(100, 768)

# PCAを適用して3次元に射影
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)

# 初期射影とPCA後の射影
original_projection = data[:, :3]  # データの最初の3次元を使用
e1, e2, e3 = np.eye(3)  # 初期基底
pca_projection = pca_result

# アニメーションデータの作成
frames = []
steps = 30
for i in range(steps + 1):
    t = i / steps
    interpolated = (1 - t) * original_projection + t * pca_projection
    frame = go.Frame(
        data=[go.Scatter3d(
            x=interpolated[:, 0],
            y=interpolated[:, 1],
            z=interpolated[:, 2],
            mode='markers',
            marker=dict(size=3, color=np.arange(interpolated.shape[0]))
        )],
        name=f'frame{i}'
    )
    frames.append(frame)

# 初期フレーム
initial_frame = go.Scatter3d(
    x=original_projection[:, 0],
    y=original_projection[:, 1],
    z=original_projection[:, 2],
    mode='markers',
    marker=dict(size=3, color=np.arange(original_projection.shape[0]))
)

# アニメーションの作成
fig = go.Figure(
    data=[initial_frame],
    layout=go.Layout(
        title="PCAによる射影の更新",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
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