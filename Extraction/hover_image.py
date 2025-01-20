import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
sample_data = {
    'Value': np.random.uniform(10, 50, 10),  # Random float values between 10 and 50
    'URL': ['./static/sample.png'] * 10  # Default image URL for all points
}

df_sample = pd.DataFrame(sample_data)

# Create Dash app
app = dash.Dash(__name__)

# Create the plot
fig = go.Figure()

# Add trace for the line chart
fig.add_trace(go.Scatter(
    x=df_sample.index,  # X axis as the index (or other appropriate time/sequence variable)
    y=df_sample['Value'],  # Y axis as the Value
    mode="lines+markers",  # Line chart with markers
    marker=dict(size=10, color='blue'),  # Customize marker style
    hovertemplate=(
        "<b>Value:</b> %{y}<br>"  # Show Value on hover
        "<b>Index:</b> %{x}<br>"  # Show Index on hover
        "<extra></extra>"
    ),
    customdata=df_sample['URL'].values  # Pass the URL as custom data to be used in hover
))

# Layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='line-chart',
        figure=fig,
        hoverData=None  # Initialize hoverData to None
    ),
    html.Div(id='hover-image', children='Hover over a data point to see the image.')  # Placeholder for image
])

# Callback to update the image on hover
@app.callback(
    Output('hover-image', 'children'),
    Input('line-chart', 'hoverData')
)
def update_image(hoverData):
    if hoverData is None:
        return 'Hover over a data point to see the image.'
    
    # Get the index of the hovered data point
    point_index = hoverData['points'][0]['pointIndex']
    image_url = df_sample.iloc[point_index]['URL']
    
    # Return HTML to display the image
    return html.Div([
        html.H4(f"Value: {df_sample.iloc[point_index]['Value']:.2f}"),
        html.Td(html.Img(src=image_url, width=200, height=200))
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
