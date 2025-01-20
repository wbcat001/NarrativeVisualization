import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
def concatenate_vectors(vector_list, window):
    vector_length = len(vector_list[0])  # 元のベクトルの長さ
    num_vectors = len(vector_list)

    # 出力用配列
    result = []

    for i in range(num_vectors):
        # ウィンドウ幅に合わせてスライス
        concatenated = []
        for j in range(window):
            index = (i + j) % num_vectors #(i + j) % num_vectors  # 循環インデックス # min(i+j, len(df-1))
            concatenated.extend(vector_list[index])
        result.append(concatenated)
    return np.array(result)


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
df = pd.read_csv("data/alice/alice_df.csv")

with open("data/alice/paragraph_embedding_gpt.pkl", "rb") as f:
    embeddings = np.array(pickle.load(f))


##
exclude_row_index = []
for i in exclude_row_index:
    embeddings[:, i] = 0
pca = PCA(n_components=2)
embeddings = sliding_average(embeddings, 20) #  sliding_average
reduced_embeddings = pca.fit_transform(embeddings)

## TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(embeddings)
# tsne_df = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2"])


# 主成分のloadingsを取得
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])

loadings['Dimension'] = range(1, loadings.shape[0] + 1)



# 重要な次元を上位10個表示
important_dims_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
important_dims_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)

print("重要な次元 (PC1):")
print(important_dims_pc1)

print("重要な次元 (PC2):")
print(important_dims_pc2)

# Add reduced dimensions back to the DataFrame
df['tSNE1'] = tsne_result[:, 0]
df['tSNE2'] = tsne_result[:, 1]

event_labels = df['ERole'].dropna().unique()
import random
def generate_colormap(df, attribute_name, default_colormap=None):
    if default_colormap:

        colormap = {value: default_colormap[value] if value in default_colormap else f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique() }
    else:
        colormap = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique()}
    return colormap

colormap_event =  {"Setup": "skyblue",
                   "Inciting Incident": "green",
                   "Turning Point": "orange",
                   "Climax": "red",
                   "Resolution": "purple",
                #    "Development": "yellow",
}
colors = generate_colormap(df, "ERole", default_colormap=colormap_event)

fig = go.Figure()

#############################
# 分割数 n を指定
num_points = len(df)
n = 40
split_size = num_points // n

# 青からオレンジへのカラースケールを生成

def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale
def generate_custom_colorscale2(n, mid_color=(0, 255, 0)):
    """
    カスタムカラースケールを生成。
    青 → 緑 → オレンジ のグラデーション。

    Parameters:
        n (int): グラデーションのステップ数。
        mid_color (tuple): 中間色 (デフォルトは緑 RGB (0, 255, 0))。

    Returns:
        list: Plotlyで使えるカラースケール [(比率, 色)] のリスト。
    """
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    mid = np.array(mid_color)  # 中間色 (RGB)

    # 青→緑部分
    first_half = [
        tuple((1 - i / (n // 2)) * blue + (i / (n // 2)) * mid)
        for i in range(n // 2)
    ]
    # 緑→オレンジ部分
    second_half = [
        tuple((1 - i / (n // 2)) * mid + (i / (n // 2)) * orange)
        for i in range(n // 2, n)
    ]

    colors = first_half + second_half
    colorscale = [
        (i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})")
        for i, c in enumerate(colors)
    ]
    return colorscale

custom_colorscale = generate_custom_colorscale(n)
for i in range(n):
    start = i * split_size
    end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
    x_segment = df.loc[start:end,"tSNE1"]
    y_segment = df.loc[start:end, "tSNE2"]

    # カラースケールからこのセグメントの色を取得
    segment_color = custom_colorscale[i][1]

    # セグメントをプロット
    fig.add_trace(go.Scatter(
        x=x_segment,
        y=y_segment,
        mode='lines',
        line=dict(
            color=segment_color,
            width=3  # ライン幅
        ),
        showlegend=False
    ))

for category in colors.keys():
    filtered = df[df["ERole"] == category]

    fig.add_trace(go.Scatter(
        x=filtered['tSNE1'],
        y=filtered['tSNE2'],
        mode="markers",
        marker=dict(color=colors[category], size=8),
        text=filtered["Event"],
      
    ))


fig.show()