import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("data/harrypotter/harry1_df.csv", index_col=0)# 
def smooth_text(df, window=50):
    smoothed_texts = []
    
    for i in range(len(df)):
        start = max(0, i - window)  # 前5パラグラフまで
        end = min(len(df), i + window + 1)  # 後5パラグラフまで
        
        # 現在の段落とその前後5パラグラフを結合
        smoothed_text = ' '.join(df['Content'][start:end].tolist())
        smoothed_texts.append(smoothed_text)
    
    df['SmoothedContent'] = smoothed_texts
    return df

df = smooth_text(df)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# BERTの事前学習済みモデルとトークナイザーをロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# BERTで埋め込みを取得する関数
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[0][0]
    return cls_embedding.cpu().numpy()

# content列から埋め込みを取得
embeddings = [get_bert_embeddings(text) for text in tqdm(df['SmoothedContent'])]

# PCAで次元削減
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 次元削減後のデータをDataFrameに変換
df_pca = pd.DataFrame(reduced_embeddings, columns=['PCA1', 'PCA2'])
df_pca['text'] = df['SmoothedContent']  # 元のテキストも保持
df_pca["ERole"] = df["ERole"]
# Plotlyで可視化
fig = px.line(df_pca, x='PCA1', y='PCA2', title='PCA of BERT Embeddings')
# fig.update_traces(textposition='top center')
fig.show()

x_values = df_pca['PCA1']
y_values = df_pca['PCA2']
# 'Event'列を色分けのためにユニークなラベルを取得
event_labels = df_pca['ERole'].dropna().unique()
import random
def generate_colormap(df, attribute_name, default_colormap=None):
    if default_colormap:

        colormap = {value: default_colormap[value] if value in default_colormap else f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique() }
    else:
        colormap = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique()}
    return colormap

colormap_event =  {"Setup": "skyblue",
                   "Inciting Incident": "green",
                   "Turning Point": "orange",
                   "Climax": "red",
                   "Resolution": "purple",
                #    "Development": "yellow",
}
colors = generate_colormap(df, "ERole", default_colormap=colormap_event)
#colors_time = [f'hsl({int(hue)}, 100%, 50%)' for hue in np.linspace(240, 30, len(df_pca)-1)] # 'hsl(0,100%,50%)'
colors_time = np.linspace(0, 1, len(df_pca) - 1)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines',  # 折れ線と点を両方表示
    # marker=dict(color=colors, size=10),  # 色とサイズの設定
    # text=event_labels, 
    line=dict(width=1, 
            #   color=colors_time,
            #   colorscale="Blues"
             )  # 線の幅を設定
))

for category in colors.keys():
    filtered = df_pca[df_pca["ERole"] == category]
    fig.add_trace(go.Scatter(
        x=filtered["PCA1"],
        y=filtered["PCA2"],
        mode="markers",
        marker=dict(color=colors[category], size=10),
))

# レイアウト設定
layout = go.Layout(
    title='PCA of BERT Embeddings (with Event Coloring)',
    xaxis=dict(title='PCA1'),
    yaxis=dict(title='PCA2'),
    hovermode='closest'  # ホバー時に詳細を表示
)

# 図を表示
fig.update_layout(layout)
fig.show()