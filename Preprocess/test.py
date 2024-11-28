import pandas as pd


df = pd.read_csv("gutenberg_text.csv", )

for i in range(10):
    print(df.iloc[i]["content"])