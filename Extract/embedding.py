from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
# BERTの事前学習済みモデルとトークナイザーをロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 入力文
text = "This is an example sentence for BERT embedding."
for i in tqdm(range(100)):
    # トークン化（入力文をBERTに対応するトークンIDに変換）
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # BERTモデルを通して埋め込みを取得
    with torch.no_grad():
        outputs = model(**inputs)

    # BERTの最後の層の埋め込み（[CLS]トークンを使用）
    last_hidden_states = outputs.last_hidden_state

    # [CLS]トークンの埋め込みを取得（文全体を表すベクトル）
    cls_embedding = last_hidden_states[0][0]

    # print(cls_embedding)
