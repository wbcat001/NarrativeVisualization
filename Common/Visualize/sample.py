import dash
from dash import dcc, html, Input, Output
import dash_table
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from movie_parser import Tag
from movie_parser import MovieScriptParser
from transformers import pipeline


script_data = [
    {"tag": "Action", "content": "The scene opens with a dramatic sunset."},
    {"tag": "Action", "content": "The hero prepares for his final battle."},
    {"tag": "Dialogue", "content": "Hero: 'I will not give up.'"},
    {"tag": "Action", "content": "The hero charges into the enemy lines."},
    {"tag": "Action", "content": "The battle rages on as both sides clash."},
]


#### Preprocess
## Parse
with open("data/Star-Wars-A-New-Hope.txt", "r") as f:
    text = f.read()
movie_parser = MovieScriptParser(text)
movie_parser.parse()

tagged_contents = movie_parser.lines

## Sentiment
def sentiment_pipeline(data):
    # result = pipeline("sentiment-analysis")(data)
    # result = [r["score"] if r["label"] == "POSITIVE" else - (r["score"])  for r in result]
    result = [0.5] * len(data)
    return result

movie_parser.set_sentiment(sentiment_pipeline)
movie_parser.add_sentiment()
data_dicts = [{"tag": tagged_content.tag, "content": tagged_content.content, "name": tagged_content.name ,"sentiment": tagged_content.other.get("sentiment")} for tagged_content in tagged_contents]

df = pd.DataFrame(data_dicts)

tag_list = [ value for key, value in vars(Tag).items() if not key.startswith("__") ]
print(tag_list)
# [ {'label': attr, 'value': attr} for attr in dir(Tag) if not callable(getattr(Tag, attr)) and not attr.startswith("__")]

#### Dashboard App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Movie Script Scenes with Sentiment Analysis"),

    ## Sentiment Graph
    dcc.Graph(id='sentiment-graph'),

    ## Dropdown for filter df
    dcc.Dropdown(
        id="filter-tag",
        options= [{"label": attr, "value": attr} for attr in tag_list] + [{"label": "All", "value": "All"}],
        value='All',
        style={'width': '50%', 'margin-bottom': '20px'}
    ),

    ## Datatable
    dash_table.DataTable(
        id='table',
        columns=[
            {"name": "Tag", "id": "tag", "presentation": "dropdown"},
            {"name": "Name", "id": "name"},
            {"name": "Content", "id": "content"},
            {"name": "Sentiment Score", "id": "sentiment"}
        ],
        data= df.to_dict("records") ,
        style_table={'width': '80%', 'margin': 'auto'},
        style_cell={
            'textAlign': 'left',  # セル内テキストの左揃え
            'whiteSpace': 'normal',  # 折り返しを有効にする
            'height': 'auto',  # 高さを自動に設定
            'minWidth': '120px',  # 最小列幅
            'width': '180px',  # 列幅
            'maxWidth': '300px'  # 最大列幅
        },
        editable=True,
        dropdown={
            'tag': {
                'options': [ {'label': attr, 'value': attr} for attr in tag_list]
            }
        }
    ),
    html.Br(),
    # センチメントスコアのグラフ
    
])


@app.callback(
    Output('sentiment-graph', 'figure'),
    Input('table', 'data')
)
def update_graph(rows):
    df_updated = pd.DataFrame(rows)
    fig = {
        'data': [
            {'x': df_updated.index, 'y': df_updated['sentiment'], 'type': 'bar', 'name': 'Sentiment Score', 'marker': {'color': 'orange'}}
        ],
        'layout': {
            'title': 'Sentiment Score per Scene',
            'xaxis': {'title': 'Scene Index'},
            'yaxis': {'title': 'Sentiment Score'}
        }
    }
    return fig

@app.callback(
    Output("table", "data"),
    Input("filter-tag", "value")
)
def update_table(selected_tag):
    if selected_tag == "All":
        filterd_df = df
    else:
        filterd_df = df[df["tag"] == selected_tag]

    table_data = filterd_df.to_dict("records")  
    return  table_data  #table_data

if __name__ == '__main__':
    app.run_server(debug=True)