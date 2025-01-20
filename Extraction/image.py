
import pandas as pd
from openai import OpenAI
import re

import json


client = OpenAI()

df = pd.read_csv("data/paragraph_alice.csv", index_col=0)
prompt = """

# 概要
文章、イベント名、イベントの要約をもとに、各イベントに関して画像の描画をします。画像生成AIに適したプロンプトを作成してください。

プロンプトには背景、キャラクター、雰囲気、行動などを含むことでそのイベントをわかりやすく表してください。

# 出力形式
[{
"event_name": "hogehoge",
"prompt": "hogehoge",
},
{
"event_name": "hogehoge",
"prompt": "hogehoge",
}]

"""

def extract_prompt(text, prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"{text}"}
    ]
    )
    print(completion.choices[0].message)
    return completion.choices[0].message.content

result_list = []
for i in range(1, 13):
    _df = df[df["Chapter"] == i]

    text = ""
    for index, row in _df.iterrows():
        text += str(row["Index"]) + " " + row["Content"] + "\n"
    # result = extract_prompt(text, prompt)
    event = _df[["Event", "ESummary"]].dropna().drop_duplicates().reset_index(drop=True)
    for index, row in event.iterrows():
        text += f"Event_{index}:" + str(row["Event"]) + "\n" + "Event_Summary:" + str(row["ESummary"]) + "\n" 
    

    result = extract_prompt(text, prompt)
    # result = re.sub(r"(\w+):", r'"\1":', result)  # キーをJSON形式にする
    result = result.replace("json", "").replace("```", "")  # シングルクォートをダブルクォートに置換
    print(result)
    result = json.loads(result)
# JSON形式として読み込み
    result_list.append(result)

    with open("events.json", "w", encoding="utf-8") as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)

print(result_list)
with open("events.json", "w", encoding="utf-8") as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)
    