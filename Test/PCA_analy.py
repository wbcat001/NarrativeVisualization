import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# サンプルデータ作成 (例: 100サンプル, 768次元)
np.random.seed(42)
embeddings = np.random.rand(100, 768)

# PCAの実行 (2次元まで次元削減)
pca = PCA(n_components=2)
pca.fit(embeddings)

# 主成分に対する次元ごとの寄与度 (loadings)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])
loadings['Dimension'] = range(1, loadings.shape[0] + 1)

# 主成分ごとの寄与度をプロット
fig = px.scatter(loadings, 
                 x='PC1', 
                 y='PC2', 
                 hover_name='Dimension',
                 title="PCA主成分に対する次元の寄与度",
                 labels={'PC1': '主成分1 (PC1)', 'PC2': '主成分2 (PC2)'},
                 template='plotly_white')

fig.show()
