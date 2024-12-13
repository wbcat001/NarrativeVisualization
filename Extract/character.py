

prompt = """
あなたは脚本制作者です。物語の文章から脚本を制作する補助をしてもらいます。

# 概要
以下の文章から登場人物を抜き出してもらいます。
情報を持ったキャラクターのリストを返してください

# 出力形式
json形式で返してください
1. 名前: キャラクターの名前を答えてください
2. 属性: キャラクターの外見や内面的な性質を答えてください
3. 記述: そのキャラクターに関する行動や記述を列挙してください。
    - インデックス番号
    - tag: "Action"または"Statement"
    - summary: 記述に関して要約してください

# 注意事項

# Example
text:
1 ...
2 ...
3 ...

output:
[
  {
    "name": "Alice",
    "attributes": {
      "appearance": "Blonde hair and wearing a blue dress",
      "personality": "Curious and adventurous"
    },
    "descriptions": [
      {
        "index": 0,
        "tag": "Action",
        "summary": "Alice chased the White Rabbit."
      },
      {
        "index": 28,
        "tag": "Statement",
        "summary": "'This is such a strange place,' Alice said to herself."
      },
      {
        "index": 60,
        "tag": "Action",
        "summary": "Alice peeked through the tiny door into a beautiful garden."
      }
    ]
  },
  {
    "name": "White Rabbit",
    "attributes": {
      "appearance": "Small and wearing a red jacket with a pocket watch",
      "personality": "Nervous and hurried"
    },
    "descriptions": [
      {
        "index": 15,
        "tag": "Action",
        "summary": "The White Rabbit ran past Alice, mumbling to himself."
      },
      {
        "index": 43,
        "tag": "Statement",
        "summary": "'Oh dear! I shall be too late!' exclaimed the White Rabbit."
      },
      {
        "index": 75,
        "tag": "Action",
        "summary": "The White Rabbit disappeared down the rabbit hole."
      }
    ]
  }
]

"""

from openai import OpenAI
client = OpenAI()

def extract_character(text, prompt):
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
    dir_path = "data/harrypotter/"
    file_name = "harry1_df.csv"
    df = pd.read_csv(dir_path + file_name, index_col=0)
    df.assign(CTag = "")
    df.assign(CText = "")
    df.assign(Character="") # rowが作られない: [[] for _ in range(len(df))]

    result_list = []
    max_chapter = df["Chapter"].max()
    min_chapter = df["Chapter"].min() 
    print((min_chapter, max_chapter))
    for i in range(min_chapter, max_chapter):
      text = ""
      for index, row in df[df["Chapter"] == i].iterrows():
          text += str(row["Index"]) + " " + row["Content"] + "\n"
      result = extract_character(text, prompt)

      result = json.loads(result.replace("json", "").replace("```", ""))
      result_list.append(result)
      for scene in result:
          name = scene["name"]
          attributes = scene["attributes"]
          
          descriptions = scene["descriptions"]
          print(descriptions)
          for d in descriptions:
              index = int(d["index"])
              df.loc[index, "CTag"] = d["tag"]
              df.loc[index, "CText"] = d["summary"]
          
              df.loc[index, "Character"] = name
  

      df.to_csv(dir_path + file_name)


      with open(dir_path + "character.json", "w", encoding="utf-8") as file:
          json.dump(result_list, file, ensure_ascii=False, indent=4)