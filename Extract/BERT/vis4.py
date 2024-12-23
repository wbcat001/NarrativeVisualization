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

# Calculate contribution for all dimension.
def calculate_contributions(embedding):
    contributions = embedding[0].numpy()  # shape: (seq_len, hidden_size)
    return contributions
def get_hoge(text):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    # 埋め込み取得
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    contributions = calculate_contributions(embedding)

    return tokens, contributions
    
def draw_snapshot(vector_list, selected_index, window, filter=None):
    
    window_vectors = []
    num_vectors = len(vector_list)
    for j in range(window):
        index = (selected_index + j) % num_vectors  # 循環インデックス
        window_vectors.append(vector_list[index])
    window_vectors = np.array(window_vectors)
    
    if filter:
        window_vectors = window_vectors[:, filter] # reshaped_data = window_vectors[:np.prod(filter)].reshape(*filter)
    fig = go.Figure(data=go.Heatmap(
        z=window_vectors,
        colorscale='Viridis',  # カラースケールを指定
        colorbar=dict(title='Intensity')  # カラーバーのタイトル
        ))
    return fig

def draw_snapshot_line(vector_list, selected_index, window, filter=None):

    window_vectors = []
    num_vectors = len(vector_list)
    for j in range(window):
        index = (selected_index + j) % num_vectors  # 循環インデックス
        window_vectors.append(vector_list[index])
    window_vectors = np.array(window_vectors)
    
    if filter:
        window_vectors = window_vectors[:, filter] # reshaped_data = window_vectors[:np.prod(filter)].reshape(*filter)
    fig = go.Figure()
    for i in range(len(window_vectors[1])):
        fig.add_trace(go.Scatter(
            y=window_vectors[:, i],
            mode='lines',
            # name=f'Dimension {adjusted_dimensions[i]}'
        ))

    return fig
def draw_line(vector_list, filter=None):
    window_vectors = vector_list
    if filter:
        window_vectors = window_vectors[:, filter]
    fig = go.Figure()
    for i in range(len(window_vectors[1])):
        fig.add_trace(go.Scatter(
            y=window_vectors[:, i],
            mode='lines',
            # name=f'Dimension {adjusted_dimensions[i]}'
        ))   
    return fig

# Text vis
def prepare_colored_tokens(tokens, contributions, dimension):
    values = contributions[:, dimension]
    print(len(values))
    print(len(tokens))
    max_val, min_val = values.max(), values.min()
    normalized_values = (values - min_val) / (max_val - min_val + 1e-9)  # 正規化
    colors = [f"rgba(255, 0, 0, {v})" for v in normalized_values]  # 赤色スケール

    styled_tokens = [
        html.Span(token, style={"backgroundColor": color, "padding": "5px", "margin": "2px"})
        for token, color in zip(tokens, colors)
    ]
    return styled_tokens

## param
window = 50
pca_num = 10
#### Data
df = pd.read_csv("data/harrypotter/harry1_df.csv")
with open("data/harrypotter/paragraph_embedding.pkl", "rb") as f:
    _embeddings = np.array(pickle.load(f))
embeddings = sliding_average(_embeddings, window) #  sliding_average


# PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
df['PCA1'] = reduced_embeddings[:, 0]
df['PCA2'] = reduced_embeddings[:, 1]
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
colors = generate_colormap(df, "ERole", default_colormap=colormap_event)

fig = go.Figure()
num_points = len(df)
n = 40
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
    end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
    x_segment = df.loc[start:end,"PCA1"]
    y_segment = df.loc[start:end, "PCA2"]

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
        x=filtered['PCA1'],
        y=filtered['PCA2'],
        mode="markers",
        marker=dict(color=colors[category], size=8),
        text=filtered["Event"],
      
    ))

fig.update_layout(width=1000, height=1000)
#### Dash
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(figure=draw_line(embeddings, list(important_dims_pca1.index)), id="pca-result")
    ]),
    html.Div([
        dcc.Graph(figure=fig, id="pca-result")
    ]),
    html.Div([
        dcc.Graph(id="snapshot")
    ]),
    html.H1("BERT Token Contributions Visualization"),
    html.Div([
        html.Label("Select Dimension:"),
        dcc.Dropdown(
        id='dimension-selector',
        options=[{'label': str(value), 'value': value} for value in list(important_dims_pca1.index)],
        placeholder="値を選択してください",
    ),
    ], style={"marginBottom": "20px"}),
    html.Div(id="colored-tokens", style={"fontSize": "20px", "lineHeight": "2em"}),
    
])

#### Callback
# pcaクリック -> テキストの表示
@app.callback(
    [Output("colored-tokens", "children"),
    Output("snapshot", "figure")],
    [Input("pca-result", "clickData"),
     State("dimension-selector", "value")]
)
def update_text(click_data, dimension):
    selected_row_text = "No point selected."
    # dimension = 10
    if click_data:
        selected_point = click_data["points"][0]
        selected_row = df[(df[f"PCA1"] == selected_point["x"]) & (df[f"PCA2"] == selected_point["y"])].index[0]
        selected_row_text = df.loc[selected_row, "Content"]
        print(f"Selected Row Text: {selected_row_text}")

        tokens, contributions = get_hoge(selected_row_text)

        
    
    return prepare_colored_tokens(tokens, contributions, dimension), draw_snapshot_line(_embeddings, selected_row, window, list(important_dims_pca1.index))




# @app.callback(
#     Output("colored-tokens", "children"),
#     Input("dimension-selector", "value")
# )
# def update_colored_tokens(dimension):
#     return prepare_colored_tokens(tokens, contributions, dimension)

if __name__ == "__main__":
    app.run_server(debug=True)