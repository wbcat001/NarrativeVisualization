import pandas as pd
import re
df = pd.read_csv("data/harrypotter/harry_potter_books.csv")
df = df[df["Book"] == "Book 1: Philosopher's Stone"]

df["Chapter"] = df["Chapter"].apply(lambda x: x.replace("chap-", ""))


paragraphs = []
current_paragraph = []

def clean_text(text):
    text = re.sub("`", "'", text)# replace("`", "'").replace("`", "'")
    return text
chapters = []
chap = 1
for index, row in df.iterrows():
    # 2スペース以上で分割
    chap = row["Chapter"]
    content = clean_text(row["Content"])
    parts = [part.strip() for part in content.split("  ") if part.strip()]
    for i, part in enumerate(parts):
        if i > 0:  # 新しいパラグラフが始まる
            if current_paragraph:
                chapters.append(chap)
                paragraphs.append(" ".join(current_paragraph))
            current_paragraph = [part]
        else:  # 同じ段落の続き
            current_paragraph.append(part)

# 最後のパラグラフを追加
if current_paragraph:
    paragraphs.append(" ".join(current_paragraph))
    chapters.append(chap)

new_df = pd.DataFrame({
    "Index": range(len(paragraphs)),
    "Chapter": chapters,  # ここではChapterがすべて1と仮定
    "Content": paragraphs
})
new_df.to_csv("data/harrypotter/harry1_df.csv")
    