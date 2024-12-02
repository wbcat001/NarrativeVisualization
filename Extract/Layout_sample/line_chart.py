import plotly.graph_objects as go
import pandas as pd

# サンプルデータの作成
data = {
    "timestamp": list(range(10)),
    "value": [1.2, 3.4, 2.3, 4.5, 3.2, 5.0, 4.2, 5.8, 6.1, 4.9],
    "tag": ["A", "B", "A", "C", "B", "A", "C", "A", "B", "C"]
}
df = pd.DataFrame(data)

# カラー定義
colors = {"A": "red", "B": "blue", "C": "green"}

# Plotlyラインチャートの作成
fig = go.Figure()

# ラインチャートを追加
fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["value"],
    mode="lines+markers",
    name="Values"
))

# タグに基づくアノテーションの追加
for idx, row in df.iterrows():
    fig.add_shape(
        type="line",
        x0=row["timestamp"], x1=row["timestamp"],
        y0=row["value"] - 1, y1=row["value"] + 1,
        line=dict(color=colors[row["tag"]], width=5),
        name=row["tag"]  # タグ名を利用
    )

# グラフのレイアウト設定
fig.update_layout(
    title="Line Chart with Annotations",
    xaxis_title="Timestamp",
    yaxis_title="Value",
    showlegend=True
)

# グラフの表示
fig.show()
