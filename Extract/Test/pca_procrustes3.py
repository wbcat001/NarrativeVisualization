"""
物語のデータセットでやってみる
"""


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
from scipy.spatial import procrustes
import dash


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

fig_default = go.Figure()
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
    fig_default.add_trace(go.Scatter(
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

    fig_default.add_trace(go.Scatter(
        x=filtered['PCA1'],
        y=filtered['PCA2'],
        mode="markers",
        marker=dict(color=colors[category], size=8),
        text=filtered["Event"],
      
    ))

fig_default.update_layout(width=1000, height=1000)


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
def graph_range(data_list):
    x_min, x_max, y_min, y_max = 100, -100, 100, -100
    for data in data_list:
        x_min = min(x_min, min(data[:, 0]))
        x_max = max(x_max, max(data[:, 0]))

        y_min = min(y_min, min(data[:, 1]))
        y_max = max(y_max, max(data[:, 1]))

    return x_min, x_max, y_min, y_max

#### Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col(
            dcc.Graph(figure=fig_default, id="pca", config={'scrollZoom': True}),
         width=12),

        
 

    ], align="center"),
     dcc.Interval(id='interval', interval=1000, max_intervals=10),
       html.Div(id="dummy-output")
    
   
], fluid=True)

@app.callback(
    Output('pca', 'figure'),
    Input('pca', 'relayoutData'),
 
)
def update_zoomed_pca(relayoutData):
    if relayoutData:
        
        # Get filter
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']
        
        # filter data
        filtered_df = _df[(_df['PCA1'] >= x_min) & (_df['PCA1'] <= x_max) & (_df['PCA2'] >= y_min) & (_df['PCA2'] <= y_max)]
        filtered_indices = filtered_df.index
        filtered_embeddings = embeddings[filtered_indices]
        print(f"filterd: {len(filtered_df)}")

        if len(filtered_df) < 2:
            return dash.no_update  # Avoid errors when filtered data is too small

        # PCA
        pca_result_new = pca.fit_transform(filtered_embeddings)

        # Align with Procrustes analysis
        _, pca_result_aligned, d = procrustes(reduced_embeddings[filtered_indices], pca_result_new)
        scale_factor = np.std(reduced_embeddings[filtered_indices]) / np.std(pca_result_aligned)

        range1 = graph_range([reduced_embeddings[filtered_indices]])
        range2 = graph_range([pca_result_aligned])
        # scale_factor = max((range1[1] - range1[0]) / (range2[1] - range1[0]), (range1[3] - range1[2]) / (range2[3] - range1[2]))
        pca_result_aligned *= scale_factor

        # debug for scale
        print(f"scale: {scale_factor:4f} = {np.std(reduced_embeddings[filtered_indices]):4f} / {np.std(pca_result_aligned):4f}")
        print(f"{reduced_embeddings[filtered_indices].shape}, {len(pca_result_aligned), len(pca_result_aligned[0])}")
        print(f"d: {d}")

        pca_result_aligned = pca_result_new
        x_min, x_max, y_min, y_max = graph_range([reduced_embeddings[filtered_indices],pca_result_aligned])
        
        fig = go.Figure(data= [go.Scatter(
            x=filtered_df["PCA1"],
            y=filtered_df["PCA2"],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Original PCA Result'
        )],
            layout=go.Layout(
            xaxis=(dict(range=[x_min, x_max])),
            yaxis=(dict(range=[y_min, y_max])),
            title=dict(text="Start Title"),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Replay",
                            method="animate",
                             args=[None],
                             execute=True,
                            )])]
        ),
        frames = [go.Frame(data= [go.Scatter(
            x=filtered_df["PCA1"],
            y=filtered_df["PCA2"],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Original PCA Result'
        )]),
        go.Frame(
                data=[
                    go.Scatter(
                        x=pca_result_aligned[:, 0],
                        y=pca_result_aligned[:, 1],
                        mode='markers',
                        marker=dict(color='green', size=8),
                        name='Transition'
                    )
                ]
        )])

        # fig = draw_pca(filtered_df, filtered_embeddings)

        return fig
    else:
        print("none")
        return fig_default
    
# クライアントサイドのコールバック：アニメーションを自動で開始
app.clientside_callback(
        """
        function (n_intervals) {
            const btn = document.querySelector("#pca > div.js-plotly-plot > div > div > svg:nth-child(3) > g.menulayer > g.updatemenu-container > g.updatemenu-header-group > g.updatemenu-button");
            console.log("btn", btn);
            console.log(n_intervals)
            if (btn != null){
            btn.dispatchEvent(new Event('click'));
            }
            return []
        }
        """,
    Output('dummy-output', 'children'),
    Input('interval', 'n_intervals'),
    prevent_initial_call=True  # 初回のコールバックを防ぐ
)

    
if __name__ == "__main__":
    app.run_server(debug=True)