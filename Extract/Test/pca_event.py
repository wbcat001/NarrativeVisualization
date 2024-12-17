"""

"""
import pandas as pd
import numpy as np
import openai
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
# OpenAI API Key (set your key here)
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
# BERTの事前学習済みモデルとトークナイザーをロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
# BERTで埋め込みを取得する関数
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[0][0]
    return cls_embedding.cpu().numpy()


    
# Read the CSV into a DataFrame
file_path = "data/dummy/event.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Generate embeddings for the Content column
def get_embedding(text):
    
    return 

# Apply embedding generation to the Content column
df['Embedding'] = df['Content'].apply(lambda x: get_bert_embeddings(x))

# Convert embeddings to a numpy array
embeddings = np.array(df['Embedding'].tolist())

# Reduce dimensions using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Add reduced dimensions back to the DataFrame
df['PCA1'] = reduced_embeddings[:, 0]
df['PCA2'] = reduced_embeddings[:, 1]

df["Genre_ID"] = (df["Genre"] != df["Genre"].shift()).cumsum()

# Create a Plotly scatter plot
fig = go.Figure()

# Group data by Genre and Genre_ID
grouped = df.groupby(['Genre', 'Genre_ID'])

# Add traces for each Genre and Genre_ID
for (genre, genre_id), group in grouped:
   
    # Add scatter points and separate lines for different Genre_IDs
    group["text"] = group.apply(lambda x: f"{x['Genre']}: {x['From']}", axis=1)
    fig.add_trace(go.Scatter(
        x=group['PCA1'],
        y=group['PCA2'],
        mode='markers+lines+text',  # Includes both points and lines
        name=f"{genre} (ID: {genre_id})",
        text=group["text"],
        hoverinfo='text',
        line=dict(shape='linear')  # Straight line connections
    ))

# Update layout
fig.update_layout(
    title="Story Flow Embeddings Visualization",
    xaxis_title="PCA1",
    yaxis_title="PCA2",
    legend_title="Genre",
    showlegend=True
)

# Show the plot
fig.show()
