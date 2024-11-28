from typing import List, Optional
import re
import pandas as pd
import random
import plotly.express as px
from Sentiment.test import SentimentCalculator
from timeline import draw_gantt
from datetime import datetime, timedelta
## type
from movie_parser import MovieScriptParser, TaggedContent, Tag

## module



file_path = ""
df = pd.read_csv("data/script_macbeth3.csv")

def parse_location(text):
    match = re.match(r"^(INT|EXT)\.\s*(.+?)\s*-\s*(.+?)(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?$", text)

    if match:
        group = match.groups()
        exterior = group[0]
        location = group[1]
        time = None
        for element in group[0:]:
            if (element == "Day"):
                time = element
    
        return exterior, location, time
    else:
        print(f"unmatch text:{text}")
        return "", "", ""
    


time = "Init"
exterior = "Init"
location = "Init"
tag = "Init"
entities = set()
timeline_data = []
length = 0
content = []
data = []
for index, line in df.iterrows():
    ## 
    tmp_tag = line["tag"]
    tmp_content = line["content"]
    tmp_name = line["name"]
    tmp_length = len(tmp_content.split(" ")) # line.length
    

    if tmp_tag == "LOCATION":
        
        ## reset
        if location != "":
            timeline_data.append({
                "location": location,
                "time": time,
                "exterior": exterior,
                "entities": list[entities],
                "length": length // 10,
                "content": content,
                "data": data,
            })
            time = ""
            exterior = ""
            location = ""
            entities = set()
            length = 0
            content = []
            data = []
        
        ## new scene
        # location, exterior, time

        exterior, location, time = parse_location(tmp_content)

        

    elif tmp_tag == "STATEMENT":
        # add content
        length += tmp_length

    elif tmp_tag == "DIALOGUE":
        # get name, add content

        length += tmp_length
        if tmp_name != None:
            entities.add(tmp_name)

    else:
        # add content
        length += tmp_length
    
    content.append(tmp_content)
    data.append(line)

def assign_color(strings, color_map=None):
    if color_map is None:
        color_map = ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(len(strings))]

    return {string: color_map[i % len(color_map)] for i, string in enumerate(strings)}



print(f"timeline_data's length: {len(timeline_data)}")


"""timeline data

"location": location, str
"time": time,  str
"exterior": exterior, str
"entities": list[entities], list[string]
"length": length // 10, int
"content": content List[string]
"""

## Add Arousal Attribute
sentiment_calculator = SentimentCalculator()
##
arousal_list = []
prev_sentence = ""
for i, d in enumerate(timeline_data):
    tmp_sentence = " ".join(d["content"])
    if prev_sentence == "":
        sentence = tmp_sentence
    else:
        sentence = prev_sentence + " " + " ".join(d["content"])
    arousal_value = sentiment_calculator.calculate_arousal(sentence)
    arousal_list.append(arousal_value)
    prev_sentence = tmp_sentence
print(f"Arousal list length is {len(arousal_list)}")

######### Visulaization Test: Location TimeLine Chart
processed_data = []
prev_time = 0
colors = {}



for i, d in enumerate(timeline_data):
    row = dict(
        # Task = d["location"],
        Location = d["location"],
        Time = d["time"],
        Exterior = d["exterior"],
        Start = prev_time ,
        Finish = prev_time + d["length"],
        Data = d["data"],
        # Characters = d["entities"] 
        

    )
    
    processed_data.append(row)

    prev_time = d["length"] + prev_time + 1


def convert_to_datetime(x):
  result = datetime(1971, 1, 1) + timedelta(days=int(x))

  return result.strftime("%Y-%m-%d")



df = pd.DataFrame(processed_data)
df["Start"] = df["Start"].apply(convert_to_datetime)
df["Finish"] = df["Finish"].apply(convert_to_datetime)
colors = assign_color(df["Location"].unique())
df["Arousal"] = arousal_list
# draw_gantt(df, "Task",colors=colors)
min = df["Start"].min()
max = df["Finish"].max()
shared_layout = {
    "margin": {"l": 30, "r": 30, "t": 5, "b": 5},  # 左右の余白を固定
    "legend": {"x": 1.1, "y": 1.02, "xanchor":"left", "yanchor": "bottom","orientation" : "h"},  # 凡例を右揃え
    # "xaxis": {"title": "Time", "tickangle": -45, "tickfont": {"size": 12}},  # x軸設定
    # "yaxis": {"title": "Location / Arousal", "tickfont": {"size": 12}},  # y軸設定
    "xaxis": {"domain": [0.1, 0.6], "showline": True, "linewidth": 2, "linecolor":"black", "range":[min, max]},
    "yaxis": {"showline": True, "linewidth": 2, "linecolor":"black"},
    "showlegend": False,
    # "xaxis": {"domain": [0.1, 0.9]},

    "plot_bgcolor": "white",  # 背景色を統一
    

}
fig_timeline1 = px.timeline(df, x_start="Start", x_end="Finish", y="Location", color="Location")
fig_timeline1.update_layout(**shared_layout)

# INT/EXT timeline
fig_timeline2 = px.timeline(df, x_start="Start", x_end="Finish", y="Exterior", color="Exterior")
fig_timeline2.update_layout(**shared_layout)

for i in range(10):
    print(df.iloc[i])
# character Timeline
sample_script = [
    {"tag": "LOCATION", "content": "Alice's house", "name": None},
    {"tag": "STATEMENT", "content": "It's a beautiful day.", "name": None},
    {"tag": "DIALOGUE", "content": "What should we do next?", "name": "Alice"},
    {"tag": "LOCATION", "content": "The forest", "name": None},
    {"tag": "STATEMENT", "content": "I feel like something is watching us.", "name": None},
    {"tag": "DIALOGUE", "content": "Maybe it's just your imagination.", "name": "Bob"},
    {"tag": "LOCATION", "content": "A clearing in the woods", "name": None},
    {"tag": "STATEMENT", "content": "Look, there's a rabbit!", "name": None},
    {"tag": "DIALOGUE", "content": "Where? I don't see it.", "name": "Alice"},
    {"tag": "LOCATION", "content": "A deep cave", "name": None},
    {"tag": "STATEMENT", "content": "It's getting darker.", "name": None},
    {"tag": "DIALOGUE", "content": "We should be careful.", "name": "Bob"},
]

#### Dashboard
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
## Sentiment Arc
fig_arousal = px.line(df, x="Start", y="Arousal", )
fig_arousal.update_layout(**shared_layout)

## Script Visuzalization Design
tag_colors = {
    "LOCATION": "#d3d3d3",  # 灰色
    "STATEMENT": "#b0e0e6",  # 淡い青
    "DIALOGUE": "#ffc0cb",  # 淡いピンク
}
app = dash.Dash(__name__)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Row([
           
                dcc.Graph(figure=fig_timeline1, style={"height": "200px"}), 
          
                dcc.Graph(figure=fig_timeline2, style={"height": "200px"}),  
        
         
                dcc.Graph(figure=fig_arousal, style={"height": "200px"}, id="line-plot"), 
           
            ]), width = 6
            ),
        dbc.Col(
            html.Div([
                html.Div([
                    html.P(f"{entry['tag']} - {entry['content']}",
                        style={
                            'backgroundColor': tag_colors.get(entry['tag'], "#ffffff"),
                            'padding': '10px',
                            'marginBottom': '5px',
                            'borderRadius': '5px',
                            'fontWeight': 'bold' if entry['tag'] == "DIALOGUE" else 'normal',
                            'color': 'black' if entry['tag'] != "DIALOGUE" else 'red'
                        })
                    if entry['tag'] != "DIALOGUE" else
                    html.P(f"{entry['name']}: {entry['content']}",
                        style={
                            'backgroundColor': tag_colors.get(entry['tag'], "#ffffff"),
                            'padding': '10px',
                            'marginBottom': '5px',
                            'borderRadius': '5px',
                            'fontWeight': 'bold',
                            'color': 'red'
                        })
                    ])
                for entry in sample_script
            ], id="script") #, style={"overflowY": "scroll", "maxHeight": "600px"}
        , width = 4),
    ])

])


@app.callback(
    
    Output("script", "children"),
    Input("line-plot", "clickData"),
    prevent_initial_call=True
)
def display_click_data(clickData):
    if clickData is not None:
        print(clickData)# {'points': [{'curveNumber': 0, 'pointNumber': 40, 'pointIndex': 40, 'x': '1971-09-18', 'y': 0.5695714285714286, 'bbox': {'x0': 1475.3300000000002, 'x1': 1477.3300000000002, 'y0': 853.65, 'y1': 855.65}}]}
        # クリックされた点のデータを取得
        point = clickData['points'][0]
        data = df.iloc[point["pointIndex"]]["Data"]
        print(data)
        
        children = [
        dbc.Row([
            html.P(f"{entry['tag']} - {entry['content']}",
                   style={
                       'backgroundColor': tag_colors.get(entry['tag'], "#ffffff"),
                       'padding': '10px',
                       'marginBottom': '5px',
                       'borderRadius': '5px',
                       'fontWeight': 'bold' if entry['tag'] == "DIALOGUE" else 'normal',
                       'color': 'black' if entry['tag'] != "DIALOGUE" else 'red'
                   })
            if entry['tag'] != "DIALOGUE" else
            html.P(f"{entry['name']}: {entry['content']}",
                   style={
                       'backgroundColor': tag_colors.get(entry['tag'], "#ffffff"),
                       'padding': '10px',
                       'marginBottom': '5px',
                       'borderRadius': '5px',
                       'fontWeight': 'bold',
                       'color': 'red'
                   })
        ], style={"display": "flex", "flexDirection": "column"})
        for entry in data
    ]

        return children#"Clicked Data:{data}"
    return html.P("Click on a point on the graph to display its data.")


if __name__ == "__main__":
    app.run_server(debug=True)