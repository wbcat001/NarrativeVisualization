import gensim.downloader as api
from sklearn.decomposition import PCA
import plotly.express as px

# Gensimから日本語の事前学習済みWord2Vecモデルをダウンロードしてロード
model = api.load("word2vec-google-news-300")  # 例えばGoogleのニュースデータセットを利用する場合
# 日本語用のモデル（例）もありますが、適宜調整してください。e.g., 'ja'のモデルを探す。

# 上位10,000単語を取得
words = model.index_to_key[:10000]

# 単語ベクトルを取得
vectors = [model[word] for word in words]

# PCAで次元削減（2次元）
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Plotlyで可視化
fig = px.scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    text=words,
    labels={'x': 'PCA Dimension 1', 'y': 'PCA Dimension 2'},
    title="Word2Vec Embeddings Reduced to 2D with PCA (Top 10,000 Words)"
)

# テキストをホバー時にのみ表示
fig.update_traces(textposition='top center', textfont=dict(size=9), hoverinfo="text")

fig.show()
