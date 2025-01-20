import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
import numpy as np

from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go


def normalize_features(data_features):
    """
    ベクトルを正規化する関数
    """
    normalizer = Normalizer()
    normalized_features = normalizer.fit_transform(data_features)
    normalized_df = pd.DataFrame(normalized_features, columns=data_features.columns)
    return normalized_df


def process_features(df, numeric_attrs, text_attrs, category_attrs):
    # 数値データの処理
    numeric_features = []
    if numeric_attrs:
        for attr in numeric_attrs:
            if attr in df:
                df[attr] = df[attr].fillna(0)  # 欠損値を0で補完
                scaler = StandardScaler()
                scaled_col = f"{attr}_scaled"
                df[scaled_col] = scaler.fit_transform(df[[attr]])
                numeric_features.append(scaled_col)

    # テキストデータのベクトル化
    text_features = []
    if text_attrs:
        for attr in text_attrs:
            if attr in df:
                tfidf_vectorizer = TfidfVectorizer(max_features=100)  # 最大特徴量数を50に制限
                tfidf_matrix = tfidf_vectorizer.fit_transform(df[attr].fillna("")).toarray()
                tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"{attr}_TFIDF_{i}" for i in range(tfidf_matrix.shape[1])])
                text_features.append(tfidf_df)

    # カテゴリーデータのエンコーディング
    category_features = []
    if category_attrs:
        for attr in category_attrs:
            if attr in df:
                dummies = pd.get_dummies(df[attr], prefix=attr)
                category_features.append(dummies)

    # 特徴量の統合
    feature_parts = []
    if numeric_features:
        feature_parts.append(df[numeric_features])
    if text_features:
        feature_parts.extend(text_features)
    if category_features:
        feature_parts.extend(category_features)

    data_features = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame()

    data_features = normalize_features(data_features)
    return df, data_features

# データフレームの読み込み
# data = {
#     'Index': [0, 1],
#     'Chapter': [0, 1],
#     'Content': [
#         "Ron tries to calm Harry, suggesting there's no need to worry yet.",
#         "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."
#     ],
#     'EImportance': [None, 6.0],
#     'ERole': [None, 'Setup'],
#     'Character': ['Ron Weasley', None],
#     'Location': [None, 'house'],
#     'LocationType': [None, 'INT'],
#     'Time': [None, 'day']
# }


def generate_features(df):
    # サンプルデータフレーム作成
    

    # 処理対象の属性リスト
    numeric_attrs = ['EImportance']
    text_attrs = ["ESummary"]
    category_attrs = ['LocationType']

    # 特徴量化
    df, data_features = process_features(df, numeric_attrs, text_attrs, category_attrs)

    # 結果を確認
    print("統合された特徴量の形状:", data_features.shape)
    print("統合された特徴量:", data_features.head())

    # 拡張されたデータフレームを表示
    print("拡張されたデータフレーム:")
    print(df.head())

    return data_features

import random
def generate_colormap(df, attribute_name, default_colormap=None):
    if default_colormap:

        colormap = {value: default_colormap[value] if value in default_colormap else f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique() }
    else:
        colormap = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique()}
    return colormap

def apply_pca(data_features, n_components=2):
    # PCAの実行
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_features)
    
    # 結果をデータフレームに変換
    pca_columns = [f"PCA{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns)

    # 分散比を表示
    explained_variance = pca.explained_variance_ratio_
    print("PCAの分散比:", explained_variance)

    # 結果をプロット
    

    return pca_df

def visualize_pca(pca_df, df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=pca_df["PCA1"],
    y=pca_df["PCA2"],
    mode='lines',  # 折れ線と点を両方表示
    # marker=dict(color=colors, size=10),  # 色とサイズの設定
    # text=event_labels, 
    line=dict(width=1, 
            #   color=colors_time,
            #   colorscale="Blues"
             )  # 線の幅を設定
))
    colormap_event =  {"Setup": "skyblue",
                   "Inciting Incident": "green",
                   "Turning Point": "orange",
                   "Climax": "red",
                   "Resolution": "purple",
                #    "Development": "yellow",
}
    colors = generate_colormap(df, "ERole", default_colormap=colormap_event)
    
    for category in colors.keys():
        filtered = pca_df[df["ERole"] == category]
        fig.add_trace(go.Scatter(
            x=filtered["PCA1"],
            y=filtered["PCA2"],
            mode="markers",
            marker=dict(color=colors[category], size=10),
        ))
    fig.show()


    
if __name__ == "__main__":
    df = pd.read_csv("data/harrypotter/harry1_df.csv") 
    data_features = generate_features(df)
    pca_df = apply_pca(data_features)
    print(f"pca length: {len(pca_df)}")
    visualize_pca(pca_df, df)


