import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(__name__)

row = dbc.Row(
    [
        dbc.Col([html.Div("One of three columns")], md=5),
        dbc.Col(html.Div("One of three columns")),
        dbc.Col(html.Div("One of three columns")),
    ],
    className="g-0",
)

app.layout = dbc.Container([
    row,
    html.P("text")
])

if __name__ == "__main__":
    app.run_server(debug=True)