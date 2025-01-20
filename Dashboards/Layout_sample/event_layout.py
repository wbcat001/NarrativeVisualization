import pandas as pd
import plotly.graph_objects as go

# サンプルデータフレーム（仮想データ）
data = {
    "Index": [1, 2, 3, 4, 5, 6, 7, 8],
    "Event": ["Alice want to", "Alice want to", "Bob marry with Cate", "Bob marry with Cate", "Bob marry with Cate", "C", "C", "A"],
    "EImportance": [10, 10, 20, 20, 20, 15, 15, 10]
}
df = pd.DataFrame(data)

# 連続する同じイベントをまとめる
df["Group"] = (df["Event"] != df["Event"].shift()).cumsum()  # 連続する値をグループ化
grouped_df = (
    df.groupby("Group")
    .agg(
        Start=("Index", "min"),
        Finish=("Index", "max"),
        Event=("Event", "first"),
        EImportance=("EImportance", "max")  # 最大の重要度を使用
    )
    .reset_index(drop=True)
)

# 可視化用のプロット
fig = go.Figure()

# 散布図をプロット
fig.add_trace(
    go.Scatter(
        x=(grouped_df["Start"] + grouped_df["Finish"]) / 2,  # 範囲の中心をX座標に
        y=[0] * len(grouped_df),  # 全て同じY座標（水平ライン）
        mode="markers+text",
        marker=dict(
            size=grouped_df["EImportance"]*10,  # 重要度を円のサイズに反映
            color=["blue", "green", "red", "orange"],  # 仮の色分け
            opacity=0.7
        ),
        text=grouped_df["Event"],  # イベント名を表示
        textposition="top center",
        name="Events"
    )
)

# レイアウト調整
fig.update_layout(
    title="Event Flow Visualization",
    xaxis_title="Index",
    yaxis=dict(visible=False),  # Y軸を非表示
    showlegend=False,  # 凡例を非表示
    template="plotly_white"
)

# グラフを表示
fig.show()
