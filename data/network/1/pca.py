# pca
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objs as go

file_path = "data/network/1/paragraph_embedding.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(data.shape)

pca = PCA(n_components=2)

pca_data = pca.fit_transform(data)

print(pca_data.shape)

df = pd.DataFrame(pca_data, columns=["x", "y"])
# n分割して色付け、線で結ぶ
n = 20
num_points = len(df)
split_size = num_points // n

plot_list = []
df["index"] = df.reset_index(inplace=True)
for i in range(n):
    start = i * split_size
    end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
    segment = df.loc[start:end]
    parts = []
    segment_color = f"rgb({int(255 * i / n)}, {int(255 * (n - i) / n)}, 0)"
    
    current_part = []
    
    previous_index = None
    for index, row in segment.iterrows():
        if previous_index is not None and row["index"] != previous_index + 1:
            parts.append(current_part)
            current_part = []
        current_part.append(row)
        previous_index = row["index"]

    if current_part:
        parts.append(current_part)

    for part in parts:
        part_df = pd.DataFrame(part)
        # セグメントをプロット
        
        plot_list.append(go.Scatter(
            x=part_df["x"],
            y=part_df["y"],
            mode='lines',
            line=dict(
                color=segment_color,
                width=2  # ライン幅
            ),
            # merker=dict(
            #     color=segment_color, size=3
            # ),
            showlegend=False
        ))

    # add text for each segment center
    center_x = np.mean(part_df["x"])
    center_y = np.mean(part_df["y"])
    plot_list.append(go.Scatter(
        x=[center_x + 0.01 * i/5],
        y=[center_y + 0.02],
        mode="text",
        text=[f" {i}"],
        textposition="top center",
        textfont=dict(
            size=18,
            
        ),
        showlegend=False
    ))

    # text size





fig = go.Figure(data=plot_list)
fig.show()