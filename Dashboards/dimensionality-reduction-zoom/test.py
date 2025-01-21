from sklearn.decomposition import PCA

import pickle
import numpy as np
import plotly.express as px
file_path = "data/network/0/paragraph_embedding.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(len(data[0]))


# pca
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

print(pca_data.shape)

fig = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1])
fig.show()