
import pandas as pd

df = pd.read_csv("data/divided_text.csv")

text = ""
for index, row in df[df["Chapter"] == 1].iterrows():
    text += str(row["Index"]) + " " + row["Content"] + "\n"


print(text)

