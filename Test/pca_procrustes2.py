"""
zoom機能に pca_procrustes.pyの内容を適用する
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Step 1: Generate synthetic data
np.random.seed(42)
data1 = np.random.rand(100, 5)

def filter_mask(data, x_min, x_max, y_min, y_max):
    """Filter data based on x and y range."""
    mask = (data[:, 0] >= x_min) & (data[:, 0] <= x_max) & (data[:, 1] >= y_min) & (data[:, 1] <= y_max)
    return mask

# Initialize PCA
pca = PCA(n_components=2)
pca_result1 = pca.fit_transform(data1)

# Dash app setup
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    dcc.Graph(id='zoom-plot', style={'height': '80vh'}),
    
    html.Button('Zoom In', id='zoom-button', n_clicks=0),
])

# Zoom-in logic
@app.callback(
    Output('zoom-plot', 'figure'),
    Input('zoom-button', 'n_clicks')
)
def update_zoom(n_clicks):
    if n_clicks == 0:
        # Initial plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pca_result1[:, 0],
            y=pca_result1[:, 1],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='PCA Result 1'
        ))
        fig.update_layout(
            title="Zoom Interaction with PCA and Procrustes Alignment",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=600,
            width=800
        )
        return fig

    # Define zoom area (each zoom step reduces the range by a factor of sqrt(2))
    scale_factor = 1 / (2 ** (n_clicks / 2))
    x_center, y_center = 0, 0  # Center remains fixed at origin for simplicity
    x_range = 1 * scale_factor
    y_range = 1 * scale_factor

    # Filter data
    mask= filter_mask(pca_result1, x_center - x_range, x_center + x_range, y_center - y_range, y_center + y_range)
    filtered_pca_result = pca_result1[mask]
    filtered_data = data1[mask]
    
    if len(filtered_data) < 2:
        return dash.no_update  # Avoid errors when filtered data is too small

    # Perform PCA on filtered data
    pca2 = PCA(n_components=2)

    pca_result2 = pca2.fit_transform(filtered_data)

    # Align with Procrustes analysis
    _, pca_result2_aligned, d = procrustes(filtered_pca_result, pca_result2)
    print(f"d: {d}")
    # Scale aligned data
    scale_factor = np.std(pca_result1) / np.std(pca_result2_aligned)
    pca_result2_aligned *= scale_factor

    # Create animation frames
    frames = []
    for alpha in np.linspace(0, 1, 10):
        interpolated = (1 - alpha) * filtered_pca_result + alpha * pca_result2_aligned
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=interpolated[:, 0],
                    y=interpolated[:, 1],
                    mode='markers',
                    marker=dict(color='green', size=8),
                    name='Transition'
                )
            ]
        )
        frames.append(frame)
    print(len(frames))
    # np.logical_not(np_bool_list)
    # Create figure
    fig = go.Figure(data= [go.Scatter(
        x=filtered_pca_result[:, 0],
        y=filtered_pca_result[:, 1],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Original PCA Result'
    )],
        layout=go.Layout(
        
        title=dict(text="Start Title"),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames = [go.Frame(
            data=[
                go.Scatter(
                    x=pca_result2_aligned[:, 0],
                    y=pca_result2_aligned[:, 1],
                    mode='markers',
                    marker=dict(color='green', size=8),
                    name='Transition'
                )
            ]
        )])
    # fig.add_trace(go.Scatter(
    #     x=filtered_pca_result[:, 0],
    #     y=filtered_pca_result[:, 1],
    #     mode='markers',
    #     marker=dict(color='blue', size=8),
    #     name='Original PCA Result'
    # ))
    
    # fig.add_trace(go.Scatter(
    #     x=pca_result2[:, 0],
    #     y=pca_result2[:, 1],
    #     mode='markers',
    #     marker=dict(color='red', size=8),
    #     name='Filtered PCA Result'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pca_result2_aligned[:, 0],
    #     y=pca_result2_aligned[:, 1],
    #     mode='markers',
    #     marker=dict(color='green', size=8),
    #     name='Aligned PCA Result'
    # ))

    

    # # Add layout details with animation settings
    # fig.update_layout(
    #     title=f"Zoom Level: {n_clicks}",
    #     xaxis_title="PC1",
    #     yaxis_title="PC2",
    #     height=600,
    #     width=800,
        
    # )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
