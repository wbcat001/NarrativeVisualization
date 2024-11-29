import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

# サンプルデータフレームを作成
df = pd.DataFrame({
    "character": ["Alice", "Rabbit", "Queen", "Alice", "Rabbit"],
    "time": [5, 10, 15, 20, 25],
    "location": ["Garden", "Castle", "Castle", "Forest", "Garden"],
    "day_night": ["day", "night", "day", "night", "day"],
    "sentiment": [10, -5, 0, 15, -10],
})

# Dashアプリケーションの初期化
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# レイアウト
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            children=[
                # サイドバー
                dbc.Col(
                    width=3,
                    children=[
                        html.H3("ファセット検索", className="text-center"),
                        html.Label("Character"),
                        dcc.Checklist(
                            id="character-filter",
                            options=[{"label": char, "value": char} for char in df["character"].unique()],
                            value=[],
                            inline=True,
                        ),
                        html.Label("Time Range"),
                        dcc.RangeSlider(
                            id="time-filter",
                            min=df["time"].min(),
                            max=df["time"].max(),
                            step=1,
                            marks={i: str(i) for i in range(df["time"].min(), df["time"].max() + 1, 5)},
                            value=[df["time"].min(), df["time"].max()],
                        ),
                        html.Label("Location"),
                        dcc.Checklist(
                            id="location-filter",
                            options=[{"label": loc, "value": loc} for loc in df["location"].unique()],
                            value=[],
                            inline=True,
                        ),
                        html.Label("Day/Night"),
                        dcc.RadioItems(
                            id="day-night-filter",
                            options=[
                                {"label": "Day", "value": "day"},
                                {"label": "Night", "value": "night"},
                                {"label": "Both", "value": "both"},
                            ],
                            value="both",
                        ),
                        html.Label("Sentiment Threshold"),
                        dcc.RangeSlider(
                            id="sentiment-filter",
                            min=df["sentiment"].min(),
                            max=df["sentiment"].max(),
                            step=1,
                            marks={i: str(i) for i in range(df["sentiment"].min(), df["sentiment"].max() + 1, 5)},
                            value=[df["sentiment"].min(), df["sentiment"].max()],
                        ),
                    ],
                ),
                # メイン画面
                dbc.Col(
                    width=9,
                    children=[
                        html.H3("検索結果グラフ", className="text-center"),
                        dcc.Graph(id="result-graph"),
                    ],
                ),
            ],
        )
    ],
)

# コールバックでフィルタリングとグラフ更新を実装
@app.callback(
    Output("result-graph", "figure"),
    [
        Input("character-filter", "value"),
        Input("time-filter", "value"),
        Input("location-filter", "value"),
        Input("day-night-filter", "value"),
        Input("sentiment-filter", "value"),
    ],
)
def update_graph(selected_characters, time_range, selected_locations, day_night, sentiment_range):
    # データフレームをフィルタリング
    filtered_df = df[
        (df["character"].isin(selected_characters) if selected_characters else True) &
        (df["time"] >= time_range[0]) & (df["time"] <= time_range[1]) &
        (df["location"].isin(selected_locations) if selected_locations else True) &
        (df["day_night"].isin([day_night]) if day_night != "both" else True) &
        (df["sentiment"] >= sentiment_range[0]) & (df["sentiment"] <= sentiment_range[1])
    ]

    # グラフを作成
    fig = go.Figure()
    for char in filtered_df["character"].unique():
        char_data = filtered_df[filtered_df["character"] == char]
        fig.add_trace(go.Scatter(
            x=char_data["time"],
            y=char_data["sentiment"],
            mode="lines+markers",
            name=char,
        ))

    # グラフのレイアウトを設定
    fig.update_layout(
        title="キャラクターごとの感情値と時間の関係",
        xaxis_title="Time",
        yaxis_title="Sentiment",
        template="plotly_white",
        legend_title="Character",
    )

    return fig

# アプリケーションの実行
if __name__ == "__main__":
    app.run_server(debug=True)
