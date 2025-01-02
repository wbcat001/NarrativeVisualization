import numpy as np
import pandas as pd
import plotly.express as px

# ランダムデータの生成
np.random.seed(42)
data = np.random.rand(50, 3)  # 50点、3次元データ

# PCA の初期投影 (主成分計算)
mean = np.mean(data, axis=0)
data_centered = data - mean
cov_matrix = np.cov(data_centered.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 初期 PCA 投影
pca_initial = data_centered @ eigenvectors[:, :2]

# 新しい回転軸の定義 (例: x軸に基づいて回転)
theta = np.radians(30)  # 30度回転
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 回転を適用した新しい PCA 投影
pca_rotated = pca_initial @ rotation_matrix.T

# アニメーション用データフレーム作成
df_initial = pd.DataFrame(pca_initial, columns=["x", "y"])
df_initial["frame"] = 0
df_initial["label"] = [f"Point {i}" for i in range(len(pca_initial))]

df_rotated = pd.DataFrame(pca_rotated, columns=["x", "y"])
df_rotated["frame"] = 1
df_rotated["label"] = df_initial["label"]

df = pd.concat([df_initial, df_rotated])

# アニメーション作成
fig = px.scatter(
    df,
    x="x",
    y="y",
    animation_frame="frame",
    text="label",
    title="PCA Projection with Rotation",
)

# トレースの見た目調整
fig.update_traces(marker=dict(size=10, opacity=0.7), textposition="top center")
fig.update_layout(
    xaxis=dict(range=[-3, 3]),
    yaxis=dict(range=[-3, 3]),
    showlegend=False,
)

# 表示
fig.show()
