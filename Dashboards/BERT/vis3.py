import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from dash import Dash, html, dcc, Input, Output
import dash_daq as daq
import plotly.express as px
import numpy as np

# モデルとトークナイザの読み込み
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# サンプルデータフレームの作成
data = {"text": [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The only thing we have to fear is fear itself."
]}
df = pd.DataFrame(data)

# 各行に対して埋め込みを計算
def compute_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # 平均プーリング

df["embedding"] = df["text"].apply(compute_embeddings)

# PCAによる次元削減
def perform_pca(embeddings, n_components=2):
    flattened_embeddings = np.vstack(embeddings)  # shape: (num_texts, hidden_size)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flattened_embeddings)
    return pca_result, pca

# 初期PCA計算
pca_result, pca_model = perform_pca(df["embedding"], n_components=2)
df["pca1"] = pca_result[:, 0]
df["pca2"] = pca_result[:, 1]

# Dashアプリケーションの構築
app = Dash(__name__)

app.layout = html.Div([
    html.H1("BERT Embedding PCA Visualization"),
    html.Div([
        html.Label("Select PCA Dimensions:"),
        dcc.Dropdown(
            id="dimension-selector",
            options=[
                {"label": f"{i} Dimensions", "value": i} for i in range(2, 6)
            ],
            value=2
        )
    ], style={"marginBottom": "20px"}),
    html.Div([
        html.H4("PCA Projection with Similarity:"),
        dcc.Graph(id="pca-plot")
    ]),
    html.Div([
        html.H4("Selected Text:"),
        html.Div(id="selected-text", style={"fontSize": "18px", "marginBottom": "20px"})
    ])
])

@app.callback(
    [Output("pca-plot", "figure"), Output("selected-text", "children")],
    [Input("pca-plot", "clickData"), Input("dimension-selector", "value")]
)
def update_visualization(click_data, n_dimensions):
    # PCA次元の更新
    pca_result, _ = perform_pca(df["embedding"], n_components=n_dimensions)
    for i in range(n_dimensions):
        df[f"pca{i+1}"] = pca_result[:, i]

    selected_row = 0  # デフォルトの選択
    if click_data:
        selected_point = click_data["points"][0]
        selected_row = df[(df[f"pca1"] == selected_point["x"]) & (df[f"pca2"] == selected_point["y"])].index[0]

    selected_embedding = df.loc[selected_row, "embedding"].reshape(1, -1)
    similarities = cosine_similarity(selected_embedding, np.vstack(df["embedding"])).flatten()

    # プロットの作成
    fig = px.scatter(
        df, x="pca1", y="pca2", text="text",
        color=similarities, color_continuous_scale="Viridis",
        labels={"color": "Cosine Similarity"},
        title="PCA Projection with Similarity"
    )
    fig.update_traces(marker=dict(size=12), textposition="top center")

    return fig, df.loc[selected_row, "text"]

if __name__ == "__main__":
    app.run_server(debug=True)
