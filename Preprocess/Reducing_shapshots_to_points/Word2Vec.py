from gensim.models import KeyedVectors
from sklearn.decomposition import PCA


import gensim
model_path = 'Preprocess/Reducing_shapshots_to_points/content/ja.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True )
# 上位10000単語を取得
words = list(model.index_to_key[:10000])
vectors = [model[word] for word in words]

# PCAで2次元に次元削減
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

import plotly.express as px

fig = px.scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    text=words
)

fig.show()