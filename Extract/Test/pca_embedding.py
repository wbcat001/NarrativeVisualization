import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

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
df = pd.read_csv("data/harrypotter/harry1_df.csv")

with open("data/harrypotter/paragraph_embedding.pkl", "rb") as f:
    embeddings = pickle.load(f)


pca = PCA(n_components=2)
embeddings = sliding_average(embeddings, 50) #  sliding_average
reduced_embeddings = pca.fit_transform(embeddings)

# 主成分のloadingsを取得
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])

loadings['Dimension'] = range(1, loadings.shape[0] + 1)

# 主成分ごとの寄与度をプロット
fig = px.scatter(loadings, 
                 x='PC1', 
                 y='PC2', 
                 hover_name='Dimension',
                 title="PCA主成分に対する次元の寄与度",
                 labels={'PC1': '主成分1 (PC1)', 'PC2': '主成分2 (PC2)'},
                 template='plotly_white')

fig.show()

# 重要な次元を上位10個表示
important_dims_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
important_dims_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)

print("重要な次元 (PC1):")
print(important_dims_pc1)

print("重要な次元 (PC2):")
print(important_dims_pc2)

# Add reduced dimensions back to the DataFrame
df['PCA1'] = reduced_embeddings[:, 0]
df['PCA2'] = reduced_embeddings[:, 1]

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
# fig.add_trace(go.Scatter(
#     x=df['PCA1'],
#     y=df['PCA2'],
#     mode="lines",
#     line=dict(color="black"),
#     text=df["Content"],
# ))
from scipy.interpolate import interp1d
f_x = interp1d(np.arange(len(df)), df["PCA1"], kind='linear', fill_value="extrapolate")
f_y = interp1d(np.arange(len(df)), df["PCA2"], kind='linear', fill_value="extrapolate")

# 新しい点を補完（細かく補完する）
new_x = np.linspace(0, len(df) - 1, 1000)
new_y = np.linspace(0, len(df) - 1, 1000)

# 補完された新しいデータを計算
x_pca_new = f_x(new_x)
y_pca_new = f_y(new_y)
time_colors = np.linspace(0, 1, len(x_pca_new))

fig.add_trace(go.Scatter(
    x=x_pca_new,
    y=y_pca_new,
    mode='markers',  # 線とマーカーを両方表示
    marker=dict(
        color=time_colors,  # 各点に色を割り当て
        colorscale='RdBu',  # 青からオレンジへのカラースケール
        size=4,  # マーカーのサイズ
        colorbar=dict(title='Color scale')  # カラーバーを追加
    ),
    
    
))
for category in colors.keys():
    filtered = df[df["ERole"] == category]

    fig.add_trace(go.Scatter(
        x=filtered['PCA1'],
        y=filtered['PCA2'],
        mode="markers",
        marker=dict(color=colors[category], size=8),
        text=filtered["Content"],
      
    ))


fig.show()