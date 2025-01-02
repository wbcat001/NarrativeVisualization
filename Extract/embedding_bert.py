from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pandas as pd
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# BERTの事前学習済みモデルとトークナイザーをロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[0][0]
    return cls_embedding.cpu().numpy()


#### Main  
dir_path = "data/dummy/"
save_name = "paragraph_embedding.pkl"
df = pd.read_csv("data/dummy/event.csv", index_col=0)

# extract embedding
embeddings = [get_bert_embeddings(text) for text in tqdm(df['Content'])]

# save embedding
with open(dir_path + save_name, "wb") as f:
    pickle.dump(embeddings, f)