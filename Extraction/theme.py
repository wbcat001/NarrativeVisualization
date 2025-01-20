## tfidf

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 必要なデータのダウンロード (初回のみ)
nltk.download("stopwords")
nltk.download("punkt")

df = pd.read_csv("data/paragraph_alice.csv", index_col=0)

stop_words = set(stopwords.words("english"))

# 前処理関数
def preprocess_text(text):
    # 小文字化
    text = text.lower()
    # 記号や数字の削除
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # トークン化
    tokens = word_tokenize(text)
    # ストップワード除去
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Content列の前処理
df["Cleaned_Content"] = df["Content"].apply(preprocess_text)
# ChapterごとにContentを結合
chapter_contents = df.groupby("Chapter")["Cleaned_Content"].apply(" ".join)

# TF-IDFベクトライザの初期化
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
tfidf_matrix = vectorizer.fit_transform(chapter_contents)

# 特徴語を取得
feature_names = vectorizer.get_feature_names()
scores = tfidf_matrix.toarray()

# 結果の整形
result = {}
for i, chapter in enumerate(chapter_contents.index):
    chapter_scores = {feature_names[j]: scores[i, j] for j in range(len(feature_names))}
    sorted_scores = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
    result[f"Chapter {chapter}"] = sorted_scores

# 出力
for chapter, keywords in result.items():
    print(f"{chapter}:")
    for word, score in keywords[:20]:
        print(f"  {word}: {score:.4f}")
