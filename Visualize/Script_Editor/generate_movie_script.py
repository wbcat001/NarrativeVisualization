"""
小説の文章から脚本風のデータを生成する
generate_movie_script(text)を定義する

input: text
output:
[
{tag: "", content: "", name: ""},
]
"""



from openai import OpenAI
import re
# ChatGPT APIの設定
client = OpenAI()

""" Example1
Tag: LOCATION, Content: INT. FOREST - DAY
Tag: ACTION, Content: Alice was walking through the forest when she suddenly heard a noise.
Tag: ACTION, Content: She looked around but couldn't see anything.
Tag: ACTION, Content: Suddenly, a rabbit appeared in front of her.
Tag: DIALOGUE, Content: ALICE: What is that sound?
Tag: ACTION, Content: The rabbit was in a hurry, looking anxiously at his watch.

"""

"""2
/// example
    [{{
        "tag": "LOCATION", "content": "plane text"
        "name": ""
    }},
    {{
        "tag": "ACTION", "content": "place text",
        "name": ""
    }}
    {{
        "tag": "DIALOGUE", "content": "place text",
        "name": "Jhon"
    }}
    ]
"""


def generate_movie_script(text):
    # APIに送るプロンプトを作成
    prompt = f"""
    I will assist you with converting parts of a novel into a screenplay format.

    Each line of the screenplay includes a Tag and Content. The Tag should be chosen from the following:

    STATEMENT: Actions or movements, explanation, situation
    DIALOGUE: Spoken lines
    LOCATION: Scene location
    DEFAULT: Other descriptions

    When there is a change of location, insert a line with a LOCATION tag. Following this, use a STATEMENT tag to describe the situation or events in the new location.

    If DIALOGUE is selected, include the name of the speaker in the format NAME: name. If a conversation is taking place between characters, express it by repeating DIALOGUE.

    Content can be longer. The emphasis is on describing it clearly.

    It is not necessary to script all the text of the novel, but to describe the appropriate information as a visualisation.


    /// Example
    Tag: LOCATION, Content: INT. FOREST - DAY
    Tag: STATEMENT, Content: Alice was walking through the forest when she suddenly heard a noise. She looked around but couldn't see anything.Suddenly, a rabbit appeared in front of her.
    Tag: DIALOGUE, Name: ALICE, Content What is that sound?
    Tag: STATEMENT, Content: The rabbit was in a hurry, looking anxiously at his watch.

    小説の文章: {text}

    変換後の脚本：
    """




    # OpenAI APIを呼び出して、脚本風に変換されたテキストを取得
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデル
        messages = [{"role": "user", "content":prompt}],
        # max_tokens=500,
        temperature=0.7
    )
    print(response)

    # APIのレスポンスから生成された脚本
    script = response.choices[0].message.content
    
    
    # 脚本を行単位で分割し、タグに基づいて分類
    pattern = r"(?i)Tag: (\w+),(?: Name: (\w+),)? Content: (.+)"

    # 結果を格納するリスト
    result = []

    # 各行を処理
    for line in script.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            tag, name, content = match.groups()
            entry = {"tag": tag, "content": content}
            if name:
                entry["name"] = name
            result.append(entry)
        
    return result

# 使用例
sample_text1 = """
Alice was walking through the forest when she suddenly heard a noise. 
She looked around but couldn't see anything. 
Suddenly, a rabbit appeared in front of her. 

ALICE: What is that sound? 

The rabbit was in a hurry, looking anxiously at his watch. 
"""

sample_text2 = """

彼女がいつものカフェに入ると、店内は午後の陽光で満たされていた。静かな音楽が流れ、数人の客がそれぞれの時間を楽しんでいる。リナは窓際の席に座り、頼んだばかりのコーヒーをじっと眺めた。どこか落ち着かない様子でカップを手に取ると、向かいの席に誰かが座った。

「久しぶりだな、リナ。」
声の主は高校時代の友人、タカだった。彼は少し日焼けした肌に、相変わらずの穏やかな笑みを浮かべている。

「タカ？どうしてここに？」
「仕事で近くに来てさ。時間が空いたから寄ってみた。ここ、よく来るんだろ？」
リナは答える代わりに、カップに視線を落とした。しばらくの沈黙が流れる。

タカは小さく息をついて続けた。「あいつのこと、まだ気にしてるのか？」
その言葉にリナは顔を上げる。タカの目には変わらない優しさがあったが、その奥にある躊躇が彼の言葉の重みを感じさせた。

その夕方、リナは近くの公園を歩いていた。タカとの再会は嬉しかったが、彼が口にした「あいつ」の名前が心をざわつかせる。ベンチに腰掛けて空を見上げると、オレンジ色の空に鳩の群れが横切った。

"""

if __name__ == "__main__":

    ## Test Code
    # generate
    movie_script = generate_movie_script(sample_text2)

    # output
    for item in movie_script:
        print(f"Tag: {item['tag']}, Content: {item['content']}")
