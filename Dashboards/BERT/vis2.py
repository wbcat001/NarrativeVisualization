import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from dash import Dash, html, Input, Output
import dash_daq as daq

# モデルとトークナイザの読み込み
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# サンプルテキスト
text = "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."
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

# 背景色付け用データ準備
def prepare_colored_tokens(tokens, contributions, dimension):
    values = contributions[:, dimension]
    max_val, min_val = values.max(), values.min()
    normalized_values = (values - min_val) / (max_val - min_val + 1e-9)  # 正規化
    colors = [f"rgba(255, 0, 0, {v})" for v in normalized_values]  # 赤色スケール

    styled_tokens = [
        html.Span(token, style={"backgroundColor": color, "padding": "5px", "margin": "2px"})
        for token, color in zip(tokens, colors)
    ]
    return styled_tokens

# Dashアプリケーションの構築
app = Dash(__name__)

app.layout = html.Div([
    html.H1("BERT Token Contributions Visualization"),
    html.Div([
        html.Label("Select Dimension:"),
        daq.NumericInput(
            id="dimension-selector",
            min=0,
            max=embeddings.shape[2] - 1,
            value=0,
            size=120
        )
    ], style={"marginBottom": "20px"}),
    html.Div(id="colored-tokens", style={"fontSize": "20px", "lineHeight": "2em"})
])

@app.callback(
    Output("colored-tokens", "children"),
    Input("dimension-selector", "value")
)
def update_colored_tokens(dimension):
    return prepare_colored_tokens(tokens, contributions, dimension)

if __name__ == "__main__":
    app.run_server(debug=True)
