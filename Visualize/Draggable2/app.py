import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State,ClientsideFunction

# Dashアプリケーションの初期設定
app = dash.Dash(
    __name__,
    external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

initial_cards = [
    {"id": "card-1", "header": "Card 1", "body": "Some content"},
    {"id": "card-2", "header": "Card 2", "body": "Some other content"},
    {"id": "card-3", "header": "Card 3", "body": "Some more content"},
]

children = [dbc.Card([
            dbc.CardHeader(card["header"]),
            dbc.CardBody(card["body"]),
            html.Button("Delete", id={"type": "delete-btn", "index": f"delete-{card['id']}"}, className="btn btn-danger btn-sm"),
        ], className="draggable", id=card["id"]) for card in initial_cards]

# レイアウト
app.layout = html.Div([
    html.Div(id="drag_container", className="container", children=children),
    html.Div(id="output", style={"marginTop": "20px"}),
])

@app.callback(
    Output("drag_container", "children"),
    Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
    [State("drag_container", "children")]
)
def remove_card(*args):
    ctx = dash.callback_context

    # ignore when load
    if not ctx.triggered:
        return dash.no_update
    
    else:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]  #??????
        print(f"this is delete button of card:{triggered_id}")
        card_id_to_delete = eval(triggered_id)["index"].replace("delete-", "")

       

        # get current card list
        children = args[-1]

        updated_children = [card for card in children if card["props"]["id"] != card_id_to_delete]

        return updated_children
   

# クライアントサイドコールバックの設定
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output("drag_container", "data-drag"),
    Input("drag_container", "id"),
)

# サーバーの実行
if __name__ == "__main__":
    app.run_server(debug=True)
