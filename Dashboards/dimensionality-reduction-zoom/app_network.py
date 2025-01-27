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

    df = data.df.copy()
    df = df.reset_index()
    num_points = len(df)
    split_size = num_points // n
    custom_colorscale = generate_custom_colorscale(n)
    plot_list = []

    ## Draw
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size if i < n - 1 else num_points # last segment adjustment
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

                mode='markers+lines',

                line=dict(
                    color=segment_color,
                    width=4  # ライン幅
                ),

                marker=dict(
                    color=segment_color, size=5
                ),

                showlegend=False,
                name=from_to,
                visible=visible
            ))

          

    # if annotation_category:
        
    #     for category in colors.keys():
    #         filtered = df[df["ERole"] == category]

    #         plot_list.append(go.Scatter(
    #             x=filtered['x'],
    #             y=filtered['y'],
    #             mode="markers",
    #             marker=dict(color=colors[category], size=4),
    #             text=filtered["Event"],
    #             name=from_to,
    #             visible=visible
            
    #         ))

    return plot_list


def generate_fig(transition_data: TransitionData, x_min=-1, x_max=1, y_min=-1, y_max=1):

    fig = go.Figure()
    
    frames, transition_data = animator.create_frames(x_min, x_max, y_min, y_max, transition_data)
    
    plot_from = get_plots(transition_data.from_data, from_to="from")
    plot_to = get_plots(transition_data.to_data, from_to="to")
    len_from, len_to = len(plot_from), len(plot_to)

    
    for plot in plot_from + plot_to:
        fig.add_trace(plot)

    # Frames
    fig.frames = [
        go.Frame(data= [go.Scatter(
            x=frame[:, 0],
            y=frame[:, 1],
            mode='markers+lines',
            marker=dict(color='green', size=3),
            line=dict(color='green', width=2),
            name='frames',
        )])
        for index, frame in enumerate(frames)
    ]

    # Layout
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



