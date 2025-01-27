"""
人工ネットワークデータの次元削減
app.pyを参考に
"""

from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from evaluate import *
from core3 import *





# クラスの初期化
data_manager = DataManager("data/network")
reducer = DimensionalityReducer()
aligner = AlignmentHandler(method="Procrustes")
animator = AnimationManager(data_manager, aligner, reducer)
transition_data = TransitionData(data_manager.data, reducer)
# annotation_manager = AnnotationManager(data_manager, aligner, reducer)
# colors = data_manager.get_colors()
annotation_category = None

frame_duration = 1000   
transition_duration = 1000

def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale

## from or to 
def get_plots(data:Data, n=20, from_to="from"):
    visible = True if from_to == "from" else False
    print(f"length: {from_to} {len(data.df)}")



    x_range0, x_range1, y_range0, y_range1 = transition_data.get_position_range()
    fig.layout = go.Layout(
            xaxis=(dict(range=[x_min, x_max])),
            yaxis=(dict(range=[y_min, y_max])),
            title=dict(text="Start Title"),
          
            # Todo: 
            # アニメーションの始動、遷移後のプロットの表示
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Replay",
                            method="animate",
                             args=[None, {"frame": {"duration": frame_duration, "redraw": False}, "transition": {"duration": transition_duration, "easing": "linear"}}, ],
                             execute=True,
                            ),
                        dict(
                            label="Show 'to'",
                            method="restyle",
                            args=[
                                {"visible": [False for _ in range(len_from)] + [True for _ in range(len_to)]},  # "from" を非表示、"to" を表示
                                {"title": "Showing 'to'"},  
                            ],
                        ),
                            ])],
            
        )
    fig.update_layout(width=1000, height=1000)
    
    return fig
fig_default = generate_fig(transition_data)
# Dashアプリケーションの作成
app = Dash(__name__)

app.layout = html.Div([

    # # 更新ボタン
    html.Button("Reset", id="reset-button", n_clicks=0),
    # # グラフ
    dcc.Graph(id="main", figure=fig_default, config={'scrollZoom': True}),
    # インターバルコンポーネント
    dcc.Interval(id="interval", interval=1000, n_intervals=0,max_intervals=10),
    html.Div(id="dummy-output"),
    html.Div(id="dummy-output2"),

])

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


app.clientside_callback(
    """
    function (n_intervals, fig_data) {
        // グローバルフラグ管理
        if (window.lastFigData === undefined) {
            window.lastFigData = null;
        }
        
        // 新しいfig_dataが渡された場合のみ処理
        if (JSON.stringify(window.lastFigData) !== JSON.stringify(fig_data)) {
            window.lastFigData = fig_data;

            const btn = document.querySelector("#main > div.js-plotly-plot > div > div > svg:nth-child(3) > g.menulayer > g.updatemenu-container > g.updatemenu-header-group > g.updatemenu-button");
            console.log("btn", btn);

            if (btn != null) {
                btn.dispatchEvent(new Event('click'));
            }
        }
        return [];
    }
    """,
    Output('dummy-output', 'children'),  # 必須ダミー
    [Input('interval', 'n_intervals'),  # 定期的に監視
     Input('main', 'figure')]           # figの生成を監視
)



if __name__ == "__main__":
    app.run_server(debug=True)



