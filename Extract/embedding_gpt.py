from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import pickle
client = OpenAI()

def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def sliding_window(content_list, window):
    num = len(content_list)
    result = []
    for i in range(num):
        t = ""
        for j in range(window):
            index = (i+j) % num
            t += content_list[index] + " "

        result.append(t)

    return result

#### Main  
dir_path = "data/dummy/"
save_name = "paragraph_embedding_gpt.pkl"
df = pd.read_csv("data/dummy/event.csv", index_col=0)

df["slidingtext"] = sliding_window(df["Content"], 50)

# extract embedding
embeddings = [get_embeddings(text) for text in tqdm(df['Content'])]

# save embedding
with open(dir_path + save_name, "wb") as f:
    pickle.dump(embeddings, f)