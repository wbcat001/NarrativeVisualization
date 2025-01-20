from dash import Dash, html, Input, Output
from dash_sortable_listview import SortableListView

# アプリケーションの作成
app = Dash(__name__)

# 初期リストデータ
initial_items = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]

# レイアウト設定
app.layout = html.Div([
    html.H1("Sortable List Example", style={"textAlign": "center"}),
    SortableListView(
        id="sortable-list",
        items=[{"id": f"item-{i}", "content": item} for i, item in enumerate(initial_items)],
        style={"border": "1px solid black", "padding": "10px", "width": "300px"}
    ),
    html.Div(id="output", style={"marginTop": "20px"})
])

# 並べ替え後のリストを表示
@app.callback(
    Output("output", "children"),
    [Input("sortable-list", "items")]
)
def display_sorted_list(items):
    sorted_list = [item["content"] for item in items]
    return f"Sorted List: {', '.join(sorted_list)}"

# サーバーの実行
if __name__ == "__main__":
    app.run_server(debug=True)
