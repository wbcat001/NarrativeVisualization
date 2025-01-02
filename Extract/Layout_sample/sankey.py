import pandas as pd
import plotly.graph_objects as go
import itertools

## data analysis
df = pd.read_csv("data/harrypotter/harry1_df.csv")

df = df[df["Event"] != df["Event"].shift()].dropna(subset=["ERole"]).reset_index()
print(len(df))
role = list(df["ERole"].unique())

#itertools.conbinations(role, 2)
data = { v: 0 for v in itertools.permutations(role, 2)}

for i in range(1, len(df)):
    row1 = df.loc[i-1, "ERole"]
    row2 = df.loc[i, "ERole"]
    print((row1, row2))
    if (row1, row2) in data:
        data[(row1, row2)] += 1
# print(data)





# データ
nodes = list(set([key[0] for key in data.keys()] + [key[1] for key in data.keys()]))
node_indices = {node: i for i, node in enumerate(nodes)}

links = {
    "source": [node_indices[from_node] for from_node, to_node in data.keys()],
    "target": [node_indices[to_node] for from_node, to_node in data.keys()],
    "value": list(data.values())
}

# サンキーダイアグラムの作成
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes
    ),
    link=dict(
        source=links["source"],
        target=links["target"],
        value=links["value"]
    )
))

# レイアウトの設定
fig.update_layout(title_text="event->event", font_size=10)

# 描画
fig.show()
