
"""
"""
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from dash import Dash, html, Input, Output, State
import dash_daq as daq
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import random
from dash import dcc
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud
# Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# window process 
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



## param
window = 50
pca_num = 10
#### Data
_df = pd.read_csv("data/harrypotter/harry1_df.csv")
with open("data/harrypotter/paragraph_embedding.pkl", "rb") as f:
    _embeddings = np.array(pickle.load(f))

_df = _df[_df["Chapter"]  < 8  ]
print(_embeddings.shape)
_embeddings = _embeddings[:len(_df), :]
embeddings = sliding_average(_embeddings, window) #  sliding_average
_df["SlideEmbedding"] = list(embeddings)


# PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
_df['PCA1'] = reduced_embeddings[:, 0]
_df['PCA2'] = reduced_embeddings[:, 1]
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])
loadings['Dimension'] = range(1, loadings.shape[0] + 1)
def plot_attribution(loadings):
    fig = px.scatter(loadings, 
                 x='PC1', 
                 y='PC2', 
                 hover_name='Dimension',
                 title="PCA主成分に対する次元の寄与度",
                 labels={'PC1': '主成分1 (PC1)', 'PC2': '主成分2 (PC2)'},
                 template='plotly_white')
    return fig
## important dimension
important_dims_pca1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
important_dims_pca2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)
print(list(important_dims_pca1.index))
print(list(important_dims_pca2))



# Color
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
colors = generate_colormap(_df, "ERole", default_colormap=colormap_event)

fig = go.Figure()
num_points = len(_df)
n = 20
split_size = num_points // n
def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale
custom_colorscale = generate_custom_colorscale(n)
for i in range(n):
    start = i * split_size
    end = (i + 1) * split_size if i < n - 1 else num_points 
    x_segment = _df.loc[start:end,"PCA1"]
    y_segment = _df.loc[start:end, "PCA2"]

    # カラースケールからこのセグメントの色を取得
    segment_color = custom_colorscale[i][1]

    # セグメントをプロット
    fig.add_trace(go.Scatter(
        x=x_segment,
        y=y_segment,
        mode='lines',
        line=dict(
            color=segment_color,
            width=2  # ライン幅
        ),
        showlegend=False
    ))

for category in colors.keys():
    filtered = _df[_df["ERole"] == category]

    fig.add_trace(go.Scatter(
        x=filtered['PCA1'],
        y=filtered['PCA2'],
        mode="markers",
        marker=dict(color=colors[category], size=8),
        text=filtered["Event"],
      
    ))

fig.update_layout(width=1000, height=1000)


def draw_pca(df, embeddings, n=20, colors=colors):
    """
    すでにslidingで平滑化されたembeddingを受け取る

    """
    df = df.reset_index()
    print(f"data num{len(embeddings)}")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    df['PCA1'] = reduced_embeddings[:, 0]
    df['PCA2'] = reduced_embeddings[:, 1]
    print(df.index)
    # colors = generate_colormap(df, "ERole", default_colormap=colormap_event)

    fig = go.Figure()
    num_points = len(df)
    split_size = num_points // n
    custom_colorscale = generate_custom_colorscale(n)

    ## Draw
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
        segment = df.loc[start:end]
        parts = []
        # y_segment = df.loc[start:end, "PCA2"]
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

        # カラースケールからこのセグメントの色を取得
        segment_color = custom_colorscale[i][1]

        for part in parts:
            part_df = pd.DataFrame(part)
            # セグメントをプロット
            fig.add_trace(go.Scatter(
                x=part_df["PCA1"],
                y=part_df["PCA2"],
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

    for category in colors.keys():
        filtered = df[df["ERole"] == category]

        fig.add_trace(go.Scatter(
            x=filtered['PCA1'],
            y=filtered['PCA2'],
            mode="markers",
            marker=dict(color=colors[category], size=8),
            text=filtered["Event"],
        
        ))

    fig.update_layout(width=1000, height=1000)
    return fig

def calc_word_position(data, x_limit=0.01, y_limit=0.01):
    words = list(data.keys())
    importances = list(data.values())
    max_importance = max(importances)
    text_sizes = [importance / max_importance * 2 for importance in importances]  # サイズの計算
    positions = []

    max_attempts = 100  # ランダム配置の最大試行回数

    for i, word in enumerate(words):
        size = text_sizes[i] / 5  # テキストのサイズ
        attempts = 0

        # ランダムな位置を決定
        while attempts < max_attempts:
            x = random.uniform(size, x_limit - size)
            y = random.uniform(size, y_limit - size)

            # 他のテキストと重なっていないかチェック
            overlap = False
            for pos in positions:
                prev_x, prev_y, prev_size = pos
                distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                if distance < (size + prev_size):  # 重なっている場合
                    overlap = True
                    break

            if not overlap:  # 重なりがなければ配置する
                positions.append((x, y, size))
                break
            attempts += 1

        # 最大試行回数に達した場合、無理やり配置
        if attempts >= max_attempts:
            print(f"Word '{word}' placed forcibly after {max_attempts} attempts.")
            x = random.uniform(size, x_limit - size)
            y = random.uniform(size, y_limit - size)
            positions.append((x, y, size))

    return words, positions, text_sizes

## dbscan

dbscan = DBSCAN(eps=0.08, min_samples=5)
_df["cluster"] = dbscan.fit_predict(_df[["PCA1", "PCA2"]])
print(f"cluster num: {_df.cluster.max()}")

## tfidf
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')

wordcloud_dict = {}
for cluster_id in np.unique(_df['cluster']):
    if cluster_id == -1:
        continue  # -1はノイズ点を示しているのでスキップ
    
    # クラスタ内のテキストデータを抽出
    cluster_data = _df[_df['cluster'] == cluster_id]
    texts = cluster_data['Content']
    
    # TF-IDFのフィッティング
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 上位のトピック（特徴語）を抽出
    feature_names = np.array(vectorizer.get_feature_names())
    summed_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = summed_tfidf.argsort()[-20:][::-1]
    
    print(f"Cluster {cluster_id} - Top topics:")
    for idx in top_indices:
        print(f'{feature_names[idx]}')
    wordcloud_data = dict(zip(feature_names[top_indices], summed_tfidf[top_indices]))

    wordcloud_dict[cluster_id] = wordcloud_data

fig_db = go.Figure()
for i in range(n):
    start = i * split_size
    end = (i + 1) * split_size if i < n - 1 else num_points 
    x_segment = _df.loc[start:end,"PCA1"]
    y_segment = _df.loc[start:end, "PCA2"]

    # カラースケールからこのセグメントの色を取得
    segment_color = custom_colorscale[i][1]

    # セグメントをプロット
    fig_db.add_trace(go.Scatter(
        x=x_segment,
        y=y_segment,
        mode='lines',
        line=dict(
            color=segment_color,
            width=2  # ライン幅
        ),
        showlegend=False
    ))



fig_db.add_trace(go.Scatter(
x=_df["PCA1"],
y=_df["PCA2"],
mode='markers',
marker=dict(size=8,
            color=_df["cluster"])
))

# テキストをプロットする座標の計算
for cluster_id in np.unique(_df['cluster']):
    if cluster_id == -1:
        continue
    cluster_data = _df[_df['cluster'] == cluster_id]
    center_x = cluster_data['PCA1'].mean()
    center_y = cluster_data['PCA2'].mean()

    wordcloud_data = wordcloud_dict[cluster_id]
    
    words, positions, sizes = calc_word_position(wordcloud_data)
    for i, (word, pos, size) in enumerate(zip(words, positions, sizes)):
        fig_db.add_trace(go.Scatter(
            x=[center_x + pos[0]],  # X方向にオフセット
            y=[center_y + pos[1]],  # Y方向にオフセット
            mode='text',
            text=[word],
            textfont=dict(
            size=max(10, size * 10),  # 各単語のフォントサイズ
            color="black"
        ),
            textposition='top center',
            showlegend=False
        )) 
fig_db.update_layout(width=1000, height=1000)

    # offset_x = 0.02  # X軸のオフセット量
    # offset_y = 0.02  # Y軸のオフセット量

    # for i, topic in enumerate(wordcloud_dict[cluster_id]):
    #     fig.add_trace(go.Scatter(
    #         x=[center_x + (i * offset_x)],  # X方向にオフセット
    #         y=[center_y + (i * offset_y)],  # Y方向にオフセット
    #         mode='text',
    #         text=[topic],
    #         textposition='top center',
    #         showlegend=False
    #     )) 

#### Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col(
            dcc.Graph(figure=fig, id="pca", config={'scrollZoom': False}),
         width=6),

        dbc.Col(
            dcc.Graph(id="zoomed-pca"),   width=6
        )
 

    ], align="center"),
    
    dbc.Row([
        dbc.Col(
            dcc.Graph(id="dbscan", figure=fig_db), width=6
        ),
        dbc.Col(
            dcc.Graph(id="wordcloud"), width=6
        ),
    ])
], fluid=True)

@app.callback(
    Output('zoomed-pca', 'figure'),
    Input('pca', 'relayoutData'),
    # prevent_initial_call=True
)
def update_zoomed_pca(relayoutData):
    if relayoutData:
        

        # if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']
        
        # 範囲でフィルタリング
        filtered_df = _df[(_df['PCA1'] >= x_min) & (_df['PCA1'] <= x_max) & 
                            (_df['PCA2'] >= y_min) & (_df['PCA2'] <= y_max)]
        filtered_indices = filtered_df.index
        filtered_embeddings = embeddings[filtered_indices]
        print(f"filterd: {len(filtered_df)}")
    else:
        print("none")
        return go.Figure()
    print("filter")
    fig = draw_pca(filtered_df, filtered_embeddings)

    return fig
if __name__ == "__main__":
    app.run_server(debug=True)