from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd

from core import *
# サンプルデータの生成
raw_data = np.random.rand(100, 5)

# クラスの初期化
data_manager = DataManager("data/books")
reducer = DimensionalityReducer()
aligner = AlignmentHandler()
animator = AnimationManager(data_manager, aligner, reducer)
transition_data = TransitionData(data_manager.data, reducer)
# annotation_manager = AnnotationManager(data_manager, aligner, reducer)
colors = data_manager.get_colors()
def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale

## 
def get_plots(data:Data, n=20, colors=colors):
    df = data.df.copy()
    df = df.reset_index()

    num_points = len(df)
    split_size = num_points // n
    custom_colorscale = generate_custom_colorscale(n)
    plot_list = []

    ## Draw
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
        segment = df.loc[start:end]
        parts = []
        
        current_part = []
        previous_index = None
        for index, row in segment.iterrows():
            if previous_index is not None and row["index"] != previous_index + 1:
                parts.append(current_part)
                current_part = []
            current_part.append(row)
            previous_index = row["index"]

        if current_part:
            parts.append(current_part)

        # カラースケールからこのセグメントの色を取得
        segment_color = custom_colorscale[i][1]

        for part in parts:
            part_df = pd.DataFrame(part)
            # セグメントをプロット
            
            plot_list.append(go.Scatter(
                x=part_df["x"],
                y=part_df["y"],
                mode='lines',
                line=dict(
                    color=segment_color,
                    width=2  # ライン幅
                ),
                # merker=dict(
                #     color=segment_color, size=3
                # ),
                showlegend=False
            ))

    for category in colors.keys():
        filtered = df[df["ERole"] == category]

        plot_list.append(go.Scatter(
            x=filtered['x'],
            y=filtered['y'],
            mode="markers",
            marker=dict(color=colors[category], size=4),
            text=filtered["Event"],
        
        ))

    return plot_list
def generate_fig(transition_data: TransitionData, x_min=-100, x_max=100,        y_min=-100, y_max=100):
    fig = go.Figure()
    # Todo
    # calc position, make go.Scatter
    frames, transition_data= animator.create_frames(x_min, x_max, y_min, y_max, transition_data)

    
    # Todo 
    # get list of go.Scatter, annotation
    plot_from = get_plots(transition_data.from_data)
    plot_to = get_plots(transition_data.to_data)

    len_from, len_to = len(plot_from), len(plot_to)

    # annotate
    # annotation_manager.annotate(fig)
    
    for plot in plot_to:
        fig.add_trace(plot)
    fig.frames = [
        go.Frame(data= [go.Scatter(
            x=frame[:, 0],
            y=frame[:, 1],
            mode='markers+lines',
            marker=dict(color='blue', size=8),
            name='frames'
        )])
        for frame in frames
    ]

    # Layout
    x_range0, x_range1, y_range0, y_range1 = transition_data.get_position_range()
    fig.layout = go.Layout(
            xaxis=(dict(range=[x_range0, x_range1])),
            yaxis=(dict(range=[y_range0, y_range1])),
            title=dict(text="Start Title"),
            # Todo: 
            # アニメーションの始動、遷移後のプロットの表示
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Replay",
                            method="animate",
                             args=[None],
                             execute=True,
                            )])],
            
        )
    fig.update_layout(width=1000, height=1000)
    
    return fig
fig_default = generate_fig(transition_data)
# Dashアプリケーションの作成
app = Dash(__name__)

app.layout = html.Div([
    # # 次元削減手法の選択
    # html.Label("次元削減手法:"),
    # dcc.Dropdown(
    #     id="reduction-method",
    #     options=[{"label": method, "value": method} for method in ["PCA", "t-SNE"]],
    #     value="PCA"
    # ),
    # # アライメント手法の選択
    # html.Label("アライメント手法:"),
    # dcc.Dropdown(
    #     id="alignment-method",
    #     options=[{"label": "Linear", "value": "linear"}],
    #     value="linear"
    # ),
    # # アニメーション速度
    # html.Label("アニメーション速度 (ms):"),
    # dcc.Slider(
    #     id="animation-speed",
    #     min=100, max=2000, step=100, value=500,
    #     marks={i: f"{i}ms" for i in range(100, 2001, 400)}
    # ),
    # # アニメーションのステップ数
    # html.Label("アニメーションステップ数:"),
    # dcc.Input(id="animation-steps", type="number", value=20),
    # # 更新ボタン
    html.Button("Reset", id="reset-button", n_clicks=0),
    # # グラフ
    dcc.Graph(id="main", figure=fig_default, config={'scrollZoom': True}),
    # インターバルコンポーネント
    dcc.Interval(id="animation-interval", interval=500, n_intervals=0)
])

# サーバーサイドでデータをキャッシュする
processed_data = data_manager.preprocess()
reduced_before = None
reduced_after = None
frames = None

@app.callback(
    Output('main', 'figure', allow_duplicate=True),
    Input('main', 'relayoutData'),
    prevent_initial_call=True
 
)
def zoom_figure(relayoutData):

    # Get Selected area
    if relayoutData:
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']

    

    fig = generate_fig(transition_data, x_min, x_max, y_min, y_max)

    return fig


@app.callback(
    Output("main", "figure"),
    Input("reset-button", "n_clicks"),
)
def reset_animation(n_clicks):
    transition_data.reset()
    fig = generate_fig(transition_data)
    return fig


# @app.callback(
#     Output("scatter-plot", "figure"),
#     Output("animation-interval", "interval"),
#     Input("animation-interval", "n_intervals"),
#     State("reduction-method", "value"),
#     State("alignment-method", "value"),
#     State("animation-speed", "value"),
#     State("animation-steps", "value")
# )
# def update_plot(n_intervals, reduction_method, alignment_method, speed, steps):
#     global reduced_before, reduced_after, frames
#     if n_intervals == 0:
#         reduced_before = reducer.reduce(processed_data, method=reduction_method)
#         reduced_after = reducer.reduce(processed_data + np.random.normal(0, 0.1, processed_data.shape), 
#                                        method=reduction_method)
#         aligned_after = aligner.align(reduced_before, reduced_after, method=alignment_method)
#         frames = animator.create_frames(reduced_before, aligned_after, steps=steps)

#     current_frame = frames[n_intervals % len(frames)]
#     fig = go.Figure(data=go.Scatter(
#         x=current_frame[:, 0],
#         y=current_frame[:, 1],
#         mode="markers",
#         marker=dict(size=10, color=np.arange(len(current_frame)), colorscale="Viridis"),
#         text=[f"Point {i}" for i in range(len(current_frame))]
#     ))
#     fig.update_layout(title="次元削減とアニメーション")

#     return fig, speed





if __name__ == "__main__":
    app.run_server(debug=True)
