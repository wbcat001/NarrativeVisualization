from google.cloud import language_v1
from google.oauth2 import service_account

def analyze_sentiment(text: str, credentials_path: str):
    # 認証情報をロード
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = language_v1.LanguageServiceClient(credentials=credentials)

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment
    print(f"Text: {text}")
    print(f"Sentiment score: {sentiment.score}")
    print(f"Sentiment magnitude: {sentiment.magnitude}")
    return {
        "score": sentiment.score,
        "magnitude": sentiment.magnitude
    }

# JSONキーのパスを指定
credentials_path = "C:/Users/acero/Downloads/story-vis-0d2ad1b5f763.json"
text_to_analyze = "I love using Google Cloud. It's so powerful and easy to use!"
result = analyze_sentiment(text_to_analyze, credentials_path)
print(result)

import pandas as pd
from tqdm import tqdm
df = pd.read_csv("data/harrypotter/harry1_df.csv",index_col=0)

scores = []
magnitudes = []
for index, row in tqdm(df.iterrows()):
    content = row["Content"]
    result = analyze_sentiment(content, credentials_path)

    scores.append(result["score"])
    magnitudes.append(["magnitude"])

import pickle
with open("score.pkl","wb") as f:
    pickle.dump(scores, f)
with open("magnitude.pkl","wb") as f:
    pickle.dump(magnitudes, f)

df["SentimentScore"] = scores
df["SentimentMagnitude"] = magnitudes

df.to_csv("data/harrypotter/harry1_df.csv", )
# AIzaSyC7iu8ncFGMxFTUeyOy64UyMUM4ros-I3I