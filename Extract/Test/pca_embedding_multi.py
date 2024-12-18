"""
複数の物語を同時にPCAによって次元削減する。
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import random
from scipy.interpolate import interp1d

# ファイルパス
narrative_csv_files = ["data/harrypotter/harry1_df.csv", "data/alice/alice_df.csv"]
narrative_pkl_files = ["data/harrypotter/paragraph_embedding.pkl", "data/alice/paragraph_embedding.pkl"]

# データの格納リスト
all_embeddings = []
all_dataframes = []
narrative_labels = []

def sliding_average(vector_list, window):
    vector_length = len(vector_list[0])  # 各ベクトルの長さ
    num_vectors = len(vector_list)

    # 平均結果を格納するリスト
    result = []

    for i in range(num_vectors):
        # ウィンドウ内のベクトルを収集
        window_vectors = []
        for j in range(window):
            index = (i + j) % num_vectors  # 循環インデックス
            window_vectors.append(vector_list[index])

        # ウィンドウ内の平均を計算
        window_mean = np.mean(window_vectors, axis=0)
        result.append(window_mean)
    return np.array(result)


# 各ファイルからデータを読み込み
for i, (csv_file, pkl_file) in enumerate(zip(narrative_csv_files, narrative_pkl_files)):
    # DataFrameの読み込み
    df = pd.read_csv(csv_file)
    all_dataframes.append(df)

    # 埋め込みデータの読み込み
    with open(pkl_file, "rb") as f:
        embeddings = np.array(pickle.load(f))
    embeddings = sliding_average(embeddings, 50)
    all_embeddings.append(embeddings)

    # ラベルを付与（物語名など）
    narrative_labels.extend([f"Narrative {i+1}"] * len(df))

# 埋め込みデータの結合
combined_embeddings = np.vstack(all_embeddings)

# PCAを実行
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(combined_embeddings)

# データフレームを結合
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df["PCA1"] = reduced_embeddings[:, 0]
combined_df["PCA2"] = reduced_embeddings[:, 1]
combined_df["Narrative"] = narrative_labels

colormap_event =  {"Setup": "skyblue",
                   "Inciting Incident": "green",
                   "Turning Point": "orange",
                   "Climax": "red",
                   "Resolution": "purple",
                #    "Development": "yellow",
}

def generate_colormap(df, attribute_name, default_colormap=None):
    if default_colormap:

        colormap = {value: default_colormap[value] if value in default_colormap else f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique() }
    else:
        colormap = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique()}
    return colormap

colors = generate_colormap(combined_df, "ERole", default_colormap=colormap_event)
# 補完用関数
def interpolate_points(x, y, num_points=1000):
    f_x = interp1d(np.arange(len(x)), x, kind='linear', fill_value="extrapolate")
    f_y = interp1d(np.arange(len(y)), y, kind='linear', fill_value="extrapolate")
    new_indices = np.linspace(0, len(x) - 1, num_points)
    return f_x(new_indices), f_y(new_indices)

# プロット
fig = go.Figure()

# 各物語ごとに異なる色でプロット
unique_narratives = combined_df["Narrative"].unique()
colormap = {narrative: f"#{random.randint(0, 0xFFFFFF):06x}" for narrative in unique_narratives}

for narrative in unique_narratives:
    narrative_data = combined_df[combined_df["Narrative"] == narrative]

    # 補完
    interpolated_x, interpolated_y = interpolate_points(narrative_data["PCA1"], narrative_data["PCA2"])

    # 時間による色のグラデーション
    time_colors = np.linspace(0, 1, len(interpolated_x))

    # プロット
    fig.add_trace(go.Scatter(
        x=interpolated_x,
        y=interpolated_y,
        mode='lines+markers',
        marker=dict(
            color=time_colors,
            colorscale='RdBu',  # カラースケール
            size=4,
            colorbar=dict(title='Color scale')
        ),
        line=dict(color=colormap[narrative]),
        name=narrative,
        text=narrative_data["Content"]
    ))

    for category in colors.keys():
        filtered = narrative_data[narrative_data["ERole"] == category]

        fig.add_trace(go.Scatter(
            x=filtered['PCA1'],
            y=filtered['PCA2'],
            mode="markers",
            marker=dict(color=colors[category], size=8),
            text=filtered["Event"],
        
        ))


# プロットを表示
fig.update_layout(
    title="PCA Projection of Multiple Narratives",
    xaxis_title="PCA1",
    yaxis_title="PCA2",
    template="plotly_white",
    showlegend=True
)
fig.show()
