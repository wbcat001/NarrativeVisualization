
prompt = """
あなたは映画の脚本制作者です。文章から脚本を制作する補助をしてもらいます。

# 概要
この文章から、起こった主要な出来事を列挙してください。
出来事のリストを見れば、この章の概要を理解できるようにしてください。
出来事には、説明を加える必要があります。

# 出力形式
1. イベント名
2. イベントの概要: 内容を説明を含んで記述してください
3. 対応するインデックス範囲: 一つの場合は[10,10]のように記述してください
3. 重要度: 1-7で評価してみて

# Example
text:
1 ...
2 ...
3 ...

output:
[
  {
    "name": "Alice encounters the White Rabbit",
    "summary": "Alice notices a peculiar White Rabbit wearing a red jacket and holding a pocket watch. The rabbit appears in a hurry and mumbling about being late.",
    "range": [1, 2],
    "importance": 6
  },
  {
    "name": "Alice follows the White Rabbit",
    "summary": "Fascinated by the White Rabbit, Alice decides to follow him, which leads her to a rabbit hole.",
    "range": [4, 7],
    "importance": 7
  },
  {
    "name": "Alice falls into the rabbit hole",
    "summary": "Alice tumbles down a deep rabbit hole, experiencing a strange and surreal descent, marking the beginning of her journey in Wonderland.",
    "range": [10, 15],
    "importance": 7
  },
  {
    "name": "Alice discovers the tiny door",
    "summary": "Alice finds herself in a room with a tiny door. She peeks through it and sees a beautiful garden that she cannot enter due to her size.",
    "range": [16, 20],
    "importance": 5
  }
]
"""

from openai import OpenAI
client = OpenAI()

def extract_event(text, prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"{text}"}
    ]
    )

    print(completion.choices[0].message)
    return completion.choices[0].message.content


if __name__ == "__main__":
    import json
    import pandas as pd

    file_path = "data/paragraph_alice.csv"

    df = pd.read_csv(file_path, index_col=0)
    df.assign(Event = "")
    df.assign(ESummary = "")
    df.assign(EImportance=0)
    for i in range(1, 13):
      text = ""
      for index, row in df[df["Chapter"] == i].iterrows():
          text += str(row["Index"]) + " " + row["Content"] + "\n"
      result = extract_event(text, prompt)
      result = json.loads(result.replace("json", "").replace("```", ""))

      for scene in result:
          name = scene["name"]
          summary = scene["summary"]
          importance = int(scene["importance"])
          range = scene["range"]
          start, end = range[0], range[1]

          df.loc[start:end, "Event"] = name
          df.loc[start:end, "ESummary"] = summary
          df.loc[start:end, "EImportance"] = importance        

      df.to_csv(file_path)


## by chatGPT
output_sample = [
  {
    "name": "Alice grows bored and curious by the riverbank",
    "summary": "Alice, tired of her sister's book that lacks pictures and conversation, begins daydreaming about making a daisy chain but is interrupted by the sudden appearance of a peculiar White Rabbit.",
    "range": [1, 2],
    "importance": 4
  },
  {
    "name": "Alice encounters the White Rabbit and follows it",
    "summary": "Alice notices the White Rabbit, remarkable for its pink eyes, waistcoat-pocket, and pocket watch. Intrigued by its hurried muttering, she chases it across a field to a rabbit hole.",
    "range": [2, 3],
    "importance": 6
  },
  {
    "name": "Alice enters the rabbit hole and falls into Wonderland",
    "summary": "Without hesitation, Alice jumps into the rabbit hole and experiences a surreal, seemingly endless fall. Along the way, she observes shelves, cupboards, and a jar labeled 'Orange Marmalade.'",
    "range": [4, 7],
    "importance": 7
  },
  {
    "name": "Alice lands in a hall and spots the White Rabbit",
    "summary": "Alice lands safely and sees the White Rabbit hurrying down a passage. Following it, she enters a long, low hall illuminated by lamps, only to lose sight of the Rabbit.",
    "range": [10, 11],
    "importance": 5
  },
  {
    "name": "Alice discovers the tiny door and the golden key",
    "summary": "Exploring the hall, Alice finds locked doors and a small glass table with a golden key. She discovers a tiny door behind a curtain and sees a beautiful garden through it but cannot enter due to her size.",
    "range": [12, 14],
    "importance": 6
  },
  {
    "name": "Alice drinks from the 'Drink Me' bottle and shrinks",
    "summary": "Alice drinks from a bottle labeled 'Drink Me,' carefully confirming it isn't poisonous. She enjoys its flavor and shrinks to ten inches tall, becoming the right size to enter the garden.",
    "range": [15, 18],
    "importance": 7
  },
  {
    "name": "Alice realizes she left the key on the table",
    "summary": "Now tiny, Alice heads to the door but realizes she left the golden key on the table. Unable to reach it, she grows frustrated and cries, lamenting her predicament.",
    "range": [20, 21],
    "importance": 5
  },
  {
    "name": "Alice finds a cake labeled 'Eat Me'",
    "summary": "Under the table, Alice discovers a cake labeled 'Eat Me.' Hoping it will either enlarge her to reach the key or shrink her to fit through the door, she eats a piece, awaiting a new transformation.",
    "range": [22, 23],
    "importance": 6
  }
]
