"""
PCAの結果に対してその寄与度が大きい列を指定
列の値が高いデータを確認する


"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

df = pd.read_csv("data/harrypotter/harry1_df.csv")

with open("data/harrypotter/paragraph_embedding.pkl", "rb") as f:
    embeddings = np.array(pickle.load(f))

target_dimension = 10
embedding_values = embeddings[:, target_dimension]

# DataFrameに埋め込み値を追加
df["Embedding"] = embedding_values

# 対象次元の値が高い順にソート
df_sorted = df.sort_values(by="Embedding", ascending=False).reset_index(drop=True)


# 上位5件を表示
top_k = 10
print(f"Top {top_k} results for Embedding Dimension {target_dimension}:\n")
for i in range(min(top_k, len(df_sorted))):
    print(f"Rank {i+1}:")
    print(f"Content: {df_sorted.loc[i, 'Content']}")
    print(f"Embedding Dimension {target_dimension}: {df_sorted.loc[i, 'Embedding']}")
    print("-" * 40)