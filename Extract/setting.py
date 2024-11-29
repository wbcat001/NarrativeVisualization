
prompt = """

あなたは脚本制作者です。物語の文章から脚本を制作する補助をしてもらいます。

# 概要
次の文章に基づいて、時間と場所の範囲を抽出して列挙してください。時間や場所が変更されるタイミングを正確に捉え、その変更の開始と終了の位置を特定してください。入力されるtextにはインデックスが振られています。
場所に関しては大分類と小分類があり、脚本で見られる表現のように、Castle - hole 

# 出力形式
以下の情報をjson形式で出力してもらいます
1. 時間: "day" または "night"
2. 場所の種類: "INT"か"EXT"
3. 場所
    - 大分類: 家、森、庭など。必ず選択してください
    - 小分類: 玄関、屋内、木の下。必要があれば記述してください
4. 範囲
    - 最初のインデックスと最後のインデックスを答えてください
5. 根拠の説明: 

# 注意事項
範囲に関して考えるときには、映像化した際に舞台が変わると思うところで区切ってください。

# Example

text: 
1 ...
2 ...
3 ...

output:   
[{
    time: "day",
    location: {
        "general": "house",
        "specific": "kitchen"
    },
    "locationtype": "INT",
    "range": [1, 10],
    "reason": "It is clear that the time is daytime and the place is inside the house, as the text mentions 'a room with morning sun"
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
    df.assign(Location = "")
    df.assign(LocationType = "")
    df.assign(Time="")

    for i in range(1, 13):
        text = ""
        for index, row in df[df["Chapter"] == i].iterrows():
            text += str(row["Index"]) + " " + row["Content"] + "\n"
        result = extract_event(text, prompt)
    
        result = json.loads(result.replace("json", "").replace("```", ""))

        for scene in result:
            location = scene["location"]["general"]
            location_type = scene["locationtype"]
            time = scene["time"]
            range = scene["range"] # eval(scene["range"])
        
            start, end = range[0], range[1]

            df.loc[start:end, "Location"] = location
            df.loc[start:end, "LocationType"] = location_type
            df.loc[start:end, "Time"] = time        

    df.to_csv(file_path)

