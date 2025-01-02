import numpy as np
import plotly.graph_objects as go

# サンプルデータを生成
np.random.seed(42)
num_points = 1000
x = np.linspace(0, 10, num_points)
y = np.sin(x) + np.random.normal(scale=0.1, size=num_points)  # サンプル波形データ

# 分割数 n を指定
n = 5
split_size = num_points // n

# 青からオレンジへのカラースケールを生成
def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale

custom_colorscale = generate_custom_colorscale(n)

# プロットの準備
fig = go.Figure()

# n 分割してカスタムカラースケールを適用
for i in range(n):
    start = i * split_size
    end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
    x_segment = x[start:end]
    y_segment = y[start:end]

    # カラースケールからこのセグメントの色を取得
    segment_color = custom_colorscale[i][1]

    # セグメントをプロット
    fig.add_trace(go.Scatter(
        x=x_segment,
        y=y_segment,
        mode='lines',
        line=dict(
            color=segment_color,
            width=3  # ライン幅
        ),
        showlegend=False
    ))

# グラフのレイアウト設定
fig.update_layout(
    title="Gradient Line Plot with Custom Blue-Orange Colorscale",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    template="plotly_white"
)

# プロットを表示
fig.show()
