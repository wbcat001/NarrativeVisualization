from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objs as go

from core import DataManager, DimensionalityReducer, AlignmentHandler, AnimationManager, TransitionData, AnnotationManager
# サンプルデータの生成
raw_data = np.random.rand(100, 5)

# クラスの初期化
data_manager = DataManager(raw_data)
reducer = DimensionalityReducer()
aligner = AlignmentHandler()
animator = AnimationManager()
transition_data = TransitionData()
annotation_manager = AnnotationManager()


# Dashアプリケーションの作成
app = Dash(__name__)

app.layout = html.Div([
    # 次元削減手法の選択
    html.Label("次元削減手法:"),
    dcc.Dropdown(
        id="reduction-method",
        options=[{"label": method, "value": method} for method in ["PCA", "t-SNE"]],
        value="PCA"
    ),
    # アライメント手法の選択
    html.Label("アライメント手法:"),
    dcc.Dropdown(
        id="alignment-method",
        options=[{"label": "Linear", "value": "linear"}],
        value="linear"
    ),
    # アニメーション速度
    html.Label("アニメーション速度 (ms):"),
    dcc.Slider(
        id="animation-speed",
        min=100, max=2000, step=100, value=500,
        marks={i: f"{i}ms" for i in range(100, 2001, 400)}
    ),
    # アニメーションのステップ数
    html.Label("アニメーションステップ数:"),
    dcc.Input(id="animation-steps", type="number", value=20),
    # 更新ボタン
    html.Button("更新", id="update-button", n_clicks=0),
    # グラフ
    dcc.Graph(id="scatter-plot"),
    # インターバルコンポーネント
    dcc.Interval(id="animation-interval", interval=500, n_intervals=0)
])

# サーバーサイドでデータをキャッシュする
processed_data = data_manager.preprocess()
reduced_before = None
reduced_after = None
frames = None

@app.callback(
    Output('pca', 'figure'),
    Input('pca', 'relayoutData'),
 
)
def zoom_figure(relayoutData):

    # Get Selected area
    if relayoutData:
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']

    fig = go.Figure()

    # annotate
    annotation_manager.annotate(fig)
    

    # 



    # 


@app.callback(
    Output("scatter-plot", "figure"),
    Output("animation-interval", "interval"),
    Input("animation-interval", "n_intervals"),
    State("reduction-method", "value"),
    State("alignment-method", "value"),
    State("animation-speed", "value"),
    State("animation-steps", "value")
)
def update_plot(n_intervals, reduction_method, alignment_method, speed, steps):
    global reduced_before, reduced_after, frames
    if n_intervals == 0:
        reduced_before = reducer.reduce(processed_data, method=reduction_method)
        reduced_after = reducer.reduce(processed_data + np.random.normal(0, 0.1, processed_data.shape), 
                                       method=reduction_method)
        aligned_after = aligner.align(reduced_before, reduced_after, method=alignment_method)
        frames = animator.create_frames(reduced_before, aligned_after, steps=steps)

    current_frame = frames[n_intervals % len(frames)]
    fig = go.Figure(data=go.Scatter(
        x=current_frame[:, 0],
        y=current_frame[:, 1],
        mode="markers",
        marker=dict(size=10, color=np.arange(len(current_frame)), colorscale="Viridis"),
        text=[f"Point {i}" for i in range(len(current_frame))]
    ))
    fig.update_layout(title="次元削減とアニメーション")

    return fig, speed


@app.callback(
    Output("animation-interval", "n_intervals"),
    Input("update-button", "n_clicks")
)
def reset_animation(n_clicks):
    return 0


if __name__ == "__main__":
    app.run_server(debug=True)
