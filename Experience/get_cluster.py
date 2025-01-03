import pandas as pd
import numpy as np
import openai
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
# OpenAI API Key (set your key here)
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
# サンプルデータ作成
from core import *
def dbscan_with_sliding_window(df, x_col, y_col, time_col, window_size=10, eps=4, min_samples=5):
    """
    時間順に前後の点を制限してDBSCANを適用。
    
    Args:
        df (pd.DataFrame): 入力データフレーム（time_colでソートされていることを前提）。
        x_col (str): x 座標のカラム名。
        y_col (str): y 座標のカラム名。
        time_col (str): 時間カラム名。
        window_size (int): クラスタリング対象の時間的ウィンドウサイズ。
        eps (float): DBSCANのepsパラメータ。
        min_samples (int): DBSCANのmin_samplesパラメータ。
    
    Returns:
        pd.Series: クラスタラベル。
    """
    # df = df.sort_values(time_col).reset_index(drop=True)
    clusters = []
    global_cluster_id = 0  # クラスタIDを一意に管理するための変数

    for i in range(len(df)):
        # 前後のデータポイントを選択
        start = max(0, i - window_size // 2)
        end = min(len(df), i + window_size // 2)
        subset = df.iloc[start:end]
        
        # サブセットでのクラスタリング
        features = subset[[x_col, y_col]].to_numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        
        # 現在の点のクラスタIDを決定
        cluster_label = db.labels_[i - start] if i - start < len(db.labels_) else -1
        
        # -1 (ノイズ) の場合はスキップ
        if cluster_label == -1:
            clusters.append(-1)
        else:
            # 新しいクラスタIDを付与
            if cluster_label not in clusters:
                clusters.append(global_cluster_id)
                global_cluster_id += 1
            else:
                clusters.append(cluster_label)
    
    return pd.Series(clusters, index=df.index)

data_manager = DataManager("data/books")
df = data_manager.data.df
embeddings = data_manager.data.slided_embeddings
# --- 類似度と距離の計算 ---
# コサイン類似度の計算
cos_similarities = [1,] + [
    cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0, 0]
    for i in range(len(embeddings) - 1)
]

# 次元削減後の距離計算
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
distances = [0 ] + [
    np.linalg.norm(reduced_embeddings[i] - reduced_embeddings[i + 1])
    for i in range(len(reduced_embeddings) - 1)
]
df["reduce_distance"] = distances
df["reduce_distance_stack"] = np.cumsum(distances)
df["x_reduce"] = reduced_embeddings[:, 0]
df["y_reduce"] = reduced_embeddings[:, 1]

## 
def print_low_similarity_sections(cos_similarities, df_annotations, threshold=0.5):
    """
    類似度が閾値を下回った部分を検出し、その前後のアノテーションを表示。

    Args:
        cos_similarities (list): コサイン類似度のリスト。
        df_annotations (pd.DataFrame): アノテーションデータフレーム。
        threshold (float): 類似度の閾値。
    """
    for i, similarity in enumerate(cos_similarities):
        if similarity < threshold:
            prev_annotation = df_annotations.iloc[i]['Content'] if i > 0 else "N/A"
            current_annotation = df_annotations.iloc[i + 1]['Content']
            next_annotation = df_annotations.iloc[i + 2]['Content'] if i + 2 < len(df_annotations) else "N/A"
            
            print(f"Low similarity detected at index {i}:")
            print(f"  Cosine Similarity: {similarity:.3f}")
            print(f"  Previous Annotation: {prev_annotation}")
            print(f"  Current Annotation: {current_annotation}")
            print(f"  Next Annotation: {next_annotation}")
            print("-" * 50)

# 使用例
print_low_similarity_sections(cos_similarities, df, threshold=0.5)


def create_clustered_scatter_plot(df, y_col):
    """
    DBSCANのクラスタ結果に基づく散布図を作成。
    
    Args:
        df (pd.DataFrame): クラスタ情報を含むデータフレーム。`cluster` カラムを含む。
        x_col (str): x 軸に使用するカラム名。
        y_col (str): y 軸に使用するカラム名。
    
    Returns:
        dict: Dash 互換の散布図データとレイアウト。
    """
    unique_clusters = df["cluster"].unique()
    scatter_data = []

    for cluster in unique_clusters:
        cluster_data = df[df["cluster"] == cluster]
        scatter_data.append({
            'x': cluster_data["_Index"],
            'y': cluster_data[y_col],
            'mode': 'markers+lines',
            'marker': {
                'size': 10,
                'opacity': 0.7,
                'color': 'gray' if cluster == -1 else None,  # ノイズは灰色
                'symbol': 'circle'
            },
            'name': f'Cluster {cluster}' if cluster != -1 else 'Noise',
            'hovertemplate': (
                "Index: %{x}<br>" +
                "distance: %{y:.3f}<br>" +
                "Annotation: %{customdata[0]}<br>" +
                "Extra Info: %{customdata[1]}<extra></extra>"
            ),
            'customdata': cluster_data[['Event', 'Content']].values.tolist(),
  
        })

    return {
        'data': scatter_data,
        'layout': {
            'title': 'DBSCAN Clustering Scatter Plot',
            'xaxis': {'title':"x"},
            'yaxis': {'title': "y"},
            'hovermode': 'closest'
        }
    }

def assign_global_cluster_ids(df, x_col, y_col, eps=0.5, min_samples=2):
    """
    DBSCANクラスタリングを行い、連続した同じクラスタIDにグローバルクラスタIDを割り当てる関数

    Parameters:
    - df: 入力データフレーム
    - x_col: X座標の列名
    - y_col: Y座標の列名
    - eps: DBSCANのepsパラメータ（デフォルト: 0.5）
    - min_samples: DBSCANのmin_samplesパラメータ（デフォルト: 2）

    Returns:
    - df: グローバルクラスタIDを追加したデータフレーム
    """
    
    # DBSCANクラスタリングの適用
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df['tmp_cluster'] = db.fit_predict(df[[x_col, y_col]])
    
    # グローバルクラスタIDを割り当てる
    global_cluster_id = -1  # 初期値としてグローバルクラスタIDを設定
    previous_cluster_id = None
    global_cluster_ids = []

    for i, row in df.iterrows():
        if row['tmp_cluster'] != -1:  # ノイズを無視
            if row['tmp_cluster'] == previous_cluster_id:  # 前の行と同じクラスタID
                global_cluster_ids.append(global_cluster_id)
            else:  # 新しいクラスタIDの場合
                global_cluster_id += 1  # 新しいグローバルクラスタIDを作成
                global_cluster_ids.append(global_cluster_id)
            previous_cluster_id = row['tmp_cluster']
        else:
            global_cluster_ids.append(-1)  # ノイズには-1を設定

    # グローバルクラスタIDを新たに列として追加
    return global_cluster_ids

def dbscan_with_custom_distance(df, x_col, y_col, time_col, eps=0.5, min_samples=5, time_weight=0.1):
    """
    カスタム距離関数を使用した時空間クラスタリング。
    
    Args:
        df (pd.DataFrame): 入力データフレーム。
        x_col (str): x 座標のカラム名。
        y_col (str): y 座標のカラム名。
        time_col (str): 時間カラム名。
        eps (float): DBSCANのepsパラメータ。
        min_samples (int): DBSCANのmin_samplesパラメータ。
        time_weight (float): 時間次元の重み。
    
    Returns:
        pd.Series: クラスタラベル。
    """
    from scipy.spatial.distance import euclidean
    def time_space_distance(p1, p2):
        spatial_distance = euclidean(p1[:2], p2[:2])
        time_distance = abs(p1[2] - p2[2]) * time_weight
        return spatial_distance + time_distance
    
    features = df[[x_col, y_col, time_col]].to_numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=time_space_distance).fit(features)
    return pd.Series(db.labels_, index=df.index)
# DBSCAN
from sklearn.cluster import DBSCAN
df["cluster"] = dbscan_with_custom_distance(df, "x_reduce", "y_reduce","_Index",eps=0.14, min_samples=2)
# DBSCAN(eps=5, min_samples=5).fit_predict(df["reduce_distance_stack"].to_numpy().reshape(-1,1))
 # dbscan_with_sliding_window(df, "x_reduce", "y_reduce", "_Index", window_size=10, eps=0.3, min_samples=2)#DBSCAN(eps=0.2, min_samples=5).fit_predict(df[["x_reduce", "y_reduce"]])

df.to_csv("Experience/_df.csv")
# --- Dash アプリの作成 ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Embedding Analysis Dashboard", style={'textAlign': 'center'}),
    dcc.Graph(id='cosine-similarity-graph'),
    dcc.Graph(id='distance-graph'),
    html.Div([
        html.Label("Annotation"),
        dcc.Dropdown(
            id='annotation-dropdown',
            options=[{'label': text, 'value': idx} for idx, text in enumerate(df['Event'])],
            placeholder="Select an annotation"
        ),
        html.Div(id='annotation-output')
    ])
])

@app.callback(
    [Output('cosine-similarity-graph', 'figure'),
     Output('distance-graph', 'figure')],
    Input('cosine-similarity-graph', 'id')  # ダミー入力（更新トリガー用）
)
def update_graphs(_):
    # コサイン類似度グラフ
    cosine_fig = {
        'data': [{
            'x': list(range(len(cos_similarities))),
            'y': cos_similarities,
            'type': 'line',
            'name': 'Cosine Similarity',
            'hovertemplate': (
                "Index: %{x}<br>" +
                "Cosine Similarity: %{y:.3f}<br>" +
                "Annotation: %{customdata[0]}<br>" +
                "Extra Info: %{customdata[1]}<extra></extra>"
            ),
            'customdata': df[['Event', 'Content']].values.tolist(),
        }],
        'layout': {
            'title': 'Cosine Similarity Between Consecutive Embeddings',
            'xaxis': {'title': 'Index'},
            'yaxis': {'title': 'Cosine Similarity'}
        }
    }
    
    # 距離グラフ
    distance_fig = create_clustered_scatter_plot(df, "reduce_distance")
    
    return cosine_fig, distance_fig

@app.callback(
    Output('annotation-output', 'children'),
    Input('annotation-dropdown', 'value')
)
def update_annotation(selected_idx):
    if selected_idx is not None:
        return f"Selected Annotation: {df.iloc[selected_idx]['Event']}"
    return "No annotation selected"

if __name__ == '__main__':
    app.run_server(debug=True)