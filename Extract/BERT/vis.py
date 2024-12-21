import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# モデルとトークナイザの読み込み
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# サンプルテキスト
text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

# 埋め込み取得
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

# トークンリストを取得
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 全次元の寄与を計算
def calculate_contributions(embeddings):
    contributions = embeddings[0].numpy()  # shape: (seq_len, hidden_size)
    return contributions

contributions = calculate_contributions(embeddings)

# 可視化データ準備
def prepare_visualization_data(tokens, contributions):
    data = []
    seq_len, hidden_size = contributions.shape
    for token_idx in range(seq_len):
        for dim_idx in range(hidden_size):
            data.append({
                "Token": tokens[token_idx],
                "Dimension": dim_idx,
                "Contribution": contributions[token_idx, dim_idx]
            })
    return pd.DataFrame(data)

visualization_data = prepare_visualization_data(tokens, contributions)

# Dashアプリケーションの構築
app = Dash(__name__)

app.layout = html.Div([
    html.H1("BERT Token Contributions by Dimension"),
    dcc.Dropdown(
        id="dimension-selector",
        options=[{"label": f"Dimension {i}", "value": i} for i in range(embeddings.shape[2])],
        value=0,
        multi=False
    ),
    dcc.Graph(id="contribution-graph")
])

@app.callback(
    Output("contribution-graph", "figure"),
    Input("dimension-selector", "value")
)
def update_graph(selected_dimension):
    filtered_data = visualization_data[visualization_data["Dimension"] == selected_dimension]
    fig = px.bar(
        filtered_data,
        x="Token",
        y="Contribution",
        color="Contribution",
        title=f"Token Contributions for Dimension {selected_dimension}",
        labels={"Contribution": "Contribution Value", "Token": "Token"}
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Tokens",
        yaxis_title="Contribution",
        showlegend=False
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
