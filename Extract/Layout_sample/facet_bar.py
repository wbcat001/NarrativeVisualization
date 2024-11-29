#### ファセット検索用のコンポーネントのサンプルレイアウト

import dash
from dash import dcc, html, Input, Output, State
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
app = dash.Dash(__name__)

# サイドバーと結果表示のレイアウト
app.layout = html.Div([
    html.Div(
        id="sidebar",
        style={"width": "20%", "float": "left", "padding": "10px", "border": "1px solid #ddd"},
        children=[
            html.H3("ファセット検索"),
            html.Label("Character"),
            dcc.Checklist(
                id="character-filter",
                options=[{"label": char, "value": char} for char in df["character"].unique()],
                value=[],
                inline=True
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
                inline=True
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
        ]
    ),
    html.Div(
        id="result",
        style={"width": "75%", "float": "right", "padding": "10px"},
        children=[
            html.H3("結果表示"),
            html.Div(id="filtered-table"),
        ]
    )
])

# コールバックでフィルタリングを実装
@app.callback(
    Output("filtered-table", "children"),
    [
        Input("character-filter", "value"),
        Input("time-filter", "value"),
        Input("location-filter", "value"),
        Input("day-night-filter", "value"),
        Input("sentiment-filter", "value"),
    ],
)
def update_results(selected_characters, time_range, selected_locations, day_night, sentiment_range):
    # データフレームをフィルタリング
    filtered_df = df[
        (df["character"].isin(selected_characters) if selected_characters else True) &
        (df["time"] >= time_range[0]) & (df["time"] <= time_range[1]) &
        (df["location"].isin(selected_locations) if selected_locations else True) &
        (df["day_night"].isin([day_night]) if day_night != "both" else True) &
        (df["sentiment"] >= sentiment_range[0]) & (df["sentiment"] <= sentiment_range[1])
    ]

    # フィルタリング結果をHTMLテーブルとして表示
    return html.Table(
        # テーブルヘッダー
        [html.Tr([html.Th(col) for col in filtered_df.columns])] +
        # テーブル行
        [html.Tr([html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns]) for i in range(len(filtered_df))]
    )

# アプリケーションの実行
if __name__ == "__main__":
    app.run_server(debug=True)
