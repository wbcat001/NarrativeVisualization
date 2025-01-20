"""
ランダムデータを使ったPCA結果へのProcrustes解析の適用
data1: full data
data2: filtered data
data2のPCA結果をdata1にアライメントする
アニメーションもついてる
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import plotly.graph_objects as go

# Step 1: Generate synthetic PCA results (or replace with your data)
np.random.seed(42)
n=80
data1 = np.random.rand(100, 5)  # Dataset 1
data2 = data1[:n, :] # Dataset 2 with slight random variations

# Perform PCA
pca = PCA(n_components=2)
pca_result1 = pca.fit_transform(data1)
pca2 = PCA(n_components=2)

pca_result2 = pca2.fit_transform(data2)

# Step 2: Procrustes analysis
_, pca_result2_aligned, disparity = procrustes(pca_result1[:n], pca_result2)
print(disparity)

# scaling
scale_factor = np.std(pca_result1) / np.std(pca_result2_aligned)
pca_result2_aligned *= scale_factor
# Step 3: Plot the results with Plotly (including animation)
fig = go.Figure()

# Add the initial PCA results
fig.add_trace(go.Scatter(
    x=pca_result1[:, 0],
    y=pca_result1[:, 1],
    mode='markers',
    marker=dict(color='blue', size=8),
    name='PCA Result 1'
))

fig.add_trace(go.Scatter(
    x=pca_result2[:, 0],
    y=pca_result2[:, 1],
    mode='markers',
    marker=dict(color='red', size=8),
    name='PCA Result 2 (Original)'
))

# Add the aligned PCA results (animated transition)
frames = [
    go.Frame(
        data=[
            go.Scatter(
                x=pca_result2_aligned[:, 0],
                y=pca_result2_aligned[:, 1],
                mode='markers',
                marker=dict(color='green', size=8),
                name='PCA Result 2 (Aligned)'
            )
        ]
    )
]

fig.frames = frames

# Define animation settings
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play Animation",
                     method="animate",
                     args=[None, dict(frame=dict(duration=1000, redraw=True),
                                      fromcurrent=True)])
            ]
        )
    ]
)

# Add axis labels and title
fig.update_layout(
    title="Procrustes Analysis of PCA Results",
    xaxis_title="PC1",
    yaxis_title="PC2",
    legend=dict(x=0.8, y=0.9),
    height=600,
    width=800
)

# Show the plot
fig.show()
