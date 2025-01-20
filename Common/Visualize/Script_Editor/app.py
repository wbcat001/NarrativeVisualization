import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State,ClientsideFunction
import plotly.graph_objects as go
from dash import dcc
import pandas as pd

from generate_movie_script import generate_movie_script 


def draw_progress_bar(current_value, total_value):
        
        progress_percentage = (current_value / total_value) * 100
        ## Progress bar
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

        return fig
class ScriptEditorApp:

    
    def __init__(self):
        ## init app
        self.app = dash.Dash(
            __name__,
            external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"],
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )

        

        self.initial_card = self.create_card_components(self.get_initial_data())
        
        self.set_layout()
        self.set_callback()

        self.source_df = self.load_text()

        self.config = {
            "token_length": 1000,
            "max_row_num": 80,
        }

        self.current_df_index = 0
        self.current_result = []
        self.saved_data = []

        self.prev_data = None

    def get_progress(self):
        whole_data = len(self.source_df)
        current_progress = self.current_df_index
        return whole_data, current_progress
    # 
    def load_text(self):
        source_file_path = "Visualize/Script_Editor/data/gutenberg_text.csv"
        source_df = pd.read_csv(source_file_path)
        

        return source_df



    def get_initial_data(self):
        ## Data Format
        return [
        {"id": "card-1", "header": "Card 1", "body": "Some content"},
        {"id": "card-2", "header": "Card 2", "body": "Some other content"},
        {"id": "card-3", "header": "Card 3", "body": "Some more content"},
    ]   
    

    def create_card_components(self, data):
        cards = [dbc.Card([
            dbc.CardHeader(card["header"]),
            dbc.CardBody(card["body"]),
            html.Button("Delete", id={"type": "delete-btn", "index": f"delete-{card['id']}"}, className="btn btn-danger btn-sm"),
        ], className="draggable", id=card["id"]) for card in data]

        return cards
    
    ## Return next batch 
    def next(self):
        # get text from self.source_df
        chunk = ""
        token_count = 0
        max_row_num = self.config["max_row_num"]#
        current_process_index = -1

        for i in range(self.current_df_index, min(self.current_df_index + max_row_num, len(self.source_df))):
            row = self.source_df.iloc[i]
            chunk += row["Content"] + "\n"
            token_count += row["Word_Count"]
            current_process_index = i

            if token_count > self.config["token_length"]:
                break
        
        self.current_df_index = current_process_index
        print(f"token_count: {token_count} \n current_index:{self.current_df_index}")

        if self.prev_data != None:
            chunk += "\n\n this is This is the end of one previous process. You can follow this up by writing.\n"
            for d in self.prev_data:
                p = d["script_data"]
                p_tag, p_content = p["tag"], p["content"]
                if "name" in p:
                    p_name = p["name"]
                    chunk += f"Tag: {p_tag}, Content: {p_content}, Name: {p_name}"

                else:
                    chunk += f"Tag: {p_tag}, Content: {p_content}"
        
        return chunk, token_count

    def generate_data(self):
        
        chunk, token_count = self.next()
        
        script_result = generate_movie_script(chunk) # data format is differenct from card data.
        print(f"script_result {script_result}")
        result = [{"id": f"card-{index}", "header": f"card-{index}", "body": d["content"], "script_data": d }
                                                    for index, d in enumerate(script_result)]
        
        print(f"length of script data for current chunk is {len(result)}")

        return chunk, token_count, result
    

    ## When Generate button was clicked
    def do_next(self):
        ## Save previous cards
        # order

        # save
        if self.current_result != []:
            self.saved_data.extend( [ card["script_data"] for card in self.current_result])


        chunk, token_count, result = self.generate_data()
        print(f"result data {result}")
        cards = self.create_card_components(result)
        self.current_result = result
        # print(f"cards: {cards}")

        self.prev_data = self.current_result[-5:]

        return chunk, token_count, cards

    def save_data(self):
        save_path = "Visualize/Script_Editor/data/script_macbeth2.csv"
        df = pd.DataFrame(self.saved_data)
        df.to_csv(save_path, index=False)
        print(f"Save data")


    def set_layout(self):
        self.app.layout = html.Div([

            dbc.Container([

                dbc.Row([
                    dbc.Col(
                        [html.Button("Generate", id={"type": "global","index":"generate-button"}, className="btn btn-danger btn-sm", style={"margin": 0, "width": 100}) ],
                        width = 1
                    ),
                    dbc.Col(
                        [html.Button("ReGenerate", id={"type": "global","index":"regenerate-button"}, className="btn btn-danger btn-sm", style={"margin": 0,"width": 100}),],
                        width = 1
                    ),
                    dbc.Col(
                        [html.Button("save", id={"type": "global","index":"save-button"}, className="btn btn-danger btn-sm", style={"margin": 0,"width": 100}),],
                        width = 1
                        ),
                    dbc.Col(
                        [
                        dcc.Graph(figure=draw_progress_bar(0,100),  config={'displayModeBar': False}, id="progress-bar"),]
                        ,width=8
                    )
                        
                    ],
                    style={"height": "10vh", "margin": 10}
                    
                    ),
                dbc.Row(
                    
                    [
                        dbc.Col([
                            dash.html.Plaintext("display source text", id="source-text",
                                                style={
                                "margin-top": "20px",
                                "font-size": "18px",
                                "white-space": "pre-wrap",  # 保持した改行をそのまま適用
                                "word-wrap": "break-word",  # 長い単語を折り返し
                                "overflow": "hidden",       # ボックス外にはみ出さない
                                # "max-width": "400px",       # 最大幅を指定
                                "border": "1px solid #ddd", # 境界線で区切り
                                "padding": "10px",          # 内側に余白
                              
                            }),
                    ], width=6),
                        dbc.Col(
                            [html.Div(id="drag_container", className="container", children=self.initial_card),
                            html.Div(id="output", style={"marginTop": "20px"}),],
                            width=6
                        )
                    ],
                    style={"height": "50vh"}
                )
                
        ]), html.Div([],id="dummy")

        ])

        
    def set_callback(self):


        ## Generate Data
        @self.app.callback(
            Output("drag_container", "children", allow_duplicate=True),
            Output("source-text", "children"),
            Output("progress-bar", "figure"),
            Input({"type": "global", "index": "generate-button"}, "n_clicks"),
            prevent_initial_call=True
        )
        def generate_card(n):
            print("generate new data")
            chunk, token_count, cards = self.do_next()

            total_value, current_value = self.get_progress()

            

            fig = draw_progress_bar(total_value, current_value)


            return cards, chunk, fig # dummy result self.create_card_components(self.get_initial_data())
        ## Edit Item

        @self.app.callback(
                Output("dummy", "children"),
            Input({"type": "global","index":"save-button"}, "n_clicks") ,
        )
        def save_data(n_click):
            self.save_data()
            return html.P(id='placeholder')

        ## Remove Item
        @self.app.callback(
            Output("drag_container", "children"),
            Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
            [State("drag_container", "children")],
            

        )
        def remove_card(*args):
            ctx = dash.callback_context
            print(ctx.triggered)
            # ignore when load
            if not ctx.triggered or len(ctx.triggered) >=2:  # generate_card Callbackでdrag_contaianerを更新したあと、このcallbackが発生し、ctx.triggeredにすべてのボタンが含まれてしまっている。lenを調べて削除実行を防いでいるけど根本的な対策がわからない。
                return dash.no_update
            
            else:
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]  #??????
                print(f"this is delete button of card:{triggered_id}")
                card_id_to_delete = eval(triggered_id)["index"].replace("delete-", "")

            

                # get current card list
                children = args[-1]

                updated_children = [card for card in children if card["props"]["id"] != card_id_to_delete]

                return updated_children
            
        
        self.app.clientside_callback(
            ClientsideFunction(namespace="clientside", function_name="make_draggable"),
            Output("drag_container", "data-drag"),
            Input("drag_container", "id"),
         

            )
        
    

    

    def run(self):
        self.app.run_server(debug=True)


if __name__ == "__main__":

    app = ScriptEditorApp()
    app.run()

    ## Test
    
