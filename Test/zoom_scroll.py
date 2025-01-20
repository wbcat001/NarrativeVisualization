import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Dashアプリケーションの作成
app = dash.Dash(__name__)

# 初期のグラフデータの設定
initial_fig = go.Figure(
    data=[go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[10, 11, 12, 13, 14],
        mode='lines+markers'
    )],
    layout=go.Layout(
        title='Anti-Zoom Example',
        xaxis=dict(range=[0, 6]),
        yaxis=dict(range=[9, 15]),
    )
)
x_range = [0, 6]
y_range=[9, 15]

# レイアウトの設定
app.layout = html.Div([
    html.H1("Anti-Zoom Example"),
    dcc.Graph(
        id='anti-zoom-graph',
        figure=initial_fig,
        config={'scrollZoom': True}  # スクロールズームを有効にする
    )
])

# コールバック: スクロールに対してアンチズーム動作を実装
@app.callback(
    Output('anti-zoom-graph', 'figure'),
    [Input('anti-zoom-graph', 'relayoutData')]  # relayoutDataを使ってズーム後の状態を受け取る
)
def update_graph(relayoutData):
    if relayoutData:
        # x軸とy軸の範囲を取得
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']
        
        # アンチズーム効果を適用 (逆転させる)
        new_x_min = 2 * x_range[0] - x_min
        new_x_max = 2 * x_range[1] - x_max
        
        new_y_min = 2 * y_range[0] - y_min
        new_y_max = 2 * y_range[1] - y_max

        x_range[0] = new_x_min
        x_range[1] = new_x_max
        y_range[0] = new_y_min
        y_range[1] = new_y_max
        
        # 新しいズーム範囲を返す
        return {
            'data': initial_fig['data'],
            'layout': go.Layout(
                title='Anti-Zoom Example',
                xaxis=dict(range=[new_x_min, new_x_max]),
                yaxis=dict(range=[new_y_min, new_y_max]),
            )
        }
    
    # ズームなしで返す場合の初期figure
    return initial_fig

if __name__ == '__main__':
    app.run_server(debug=True)
