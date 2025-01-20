import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# 必要なデータをダウンロード
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# サンプルデータフレーム
data = {
    "index": [0, 1, 2],
    "paragraph": [
        "Alice was beginning to get very tired of sitting by her sister on the bank.",
        "So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid.",
        "But it was all very well to say 'Drink me,' but the wise little Alice was not going to do that in a hurry."
    ]
}
df = pd.DataFrame(data)

# テキスト前処理関数
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text.lower())  # 小文字化してトークン化
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]  # ストップワードと記号を除去
    return " ".join(tokens)

# 前処理を適用
df["processed"] = df["paragraph"].apply(preprocess_text)

# TF-IDFベクトル化
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # 上位10語を抽出
tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed"])

# 全体の重要な単語を取得
tfidf_features = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray().sum(axis=0)  # 単語ごとにスコアを合計

# 単語とスコアのペアを作成
word_scores = list(zip(tfidf_features, tfidf_array))

# スコアが高い順にソート
sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

# 上位の単語を表示
print("Top Words Across All Paragraphs:")
for word, score in sorted_words:
    print(f"{word}: {score:.4f}")
