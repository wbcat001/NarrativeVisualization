from dash import Dash, dcc, html
import plotly.graph_objects as go

# 全体の値と現在の値
total_value = 1000
current_value = 500

# 進捗率の計算
progress_percentage = (current_value / total_value) * 100

# Dash アプリケーションの作成
app = Dash(__name__)

# 進捗バーの作成
fig = go.Figure(go.Bar(
    x=[current_value, total_value - current_value],
    y=['Progress', "Progress"],
    orientation='h',
    text=[f'{progress_percentage:.2f}%', ''],
    textposition='inside',
    marker=dict(color=['#4CAF50', '#000000'])  # 緑と灰色
))
fig.update_layout(
    title="Progress Bar",
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    barmode='stack',
    plot_bgcolor='white',
    height=100,
    dragmode=False,
    xaxis_fixedrange=True,
    yaxis_fixedrange=True,
)

# レイアウト
app.layout = html.Div([
    html.H1("進捗バーのデモ", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig,  config={'displayModeBar': False}),
])

# アプリケーションの実行
if __name__ == '__main__':
    app.run_server(debug=True)
