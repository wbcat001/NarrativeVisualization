from openai import OpenAI
client = OpenAI()
import json
import pandas as pd


def get_scene_evaluation(scene_text):
    """
    LLMにシーンテキストを渡し、要約、シーン題名、適切に区切られているか、情報が足りているかを評価する
    """
    # 質問のテンプレート
    prompt = """
    次のシーンについて要約し、JSON形式で返答してください。以下のキーを持つJSON形式に従ってください。このテキストは、あるメトリクスを用いて機械的に分割したものです。evaluationではそれが、舞台や話題の転換点で適切に区切られているか評価してください:
    {
        "summary": "シーンの要約",
        "title": "シーンの題名",
        "evaluation": "シーンが適切に区切られているか、情報が足りているかについての評価"
    }

    シーンのテキスト:
    
    """
    prompt += "\n" + scene_text

    # LLMにリクエスト
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデル
        messages=[  
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        
    )

    # レスポンスから結果を抽出
    response_text = response.choices[0].message.content.replace("```", "").replace("json","")# .strip() #  response['choices'][0]
    print(response_text)
    # 結果を辞書形式で整理
    result = {}
    try:
        # JSONをパース
        response_data = json.loads(response_text)
        result = {
            "summary": response_data["summary"],
            "title": response_data["title"],
            "evaluation": response_data["evaluation"]
        }
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")

    return result


def process_clusters_and_evaluate(df, text_column, cluster_column):
    """
    クラスタごとにテキストを集約し、LLMで評価を得てJSON形式で保存する関数
    """
    # 各クラスタのテキストを集約
    cluster_texts = {}
    for cluster_id in df[cluster_column].unique():
        print(cluster_id)
        cluster_data = df[df[cluster_column] == cluster_id]
        if (len(cluster_data) > 1) & cluster_id != "-1": 

            cluster_text = " ".join(cluster_data[text_column].values)
            cluster_texts[cluster_id] = cluster_text

    # 各クラスタに対してLLMで評価
    cluster_results = {}
    for cluster_id, scene_text in cluster_texts.items():
        evaluation = get_scene_evaluation(scene_text)
        cluster_results[int(cluster_id)] = evaluation

    # 結果をJSON形式で保存
    output_file = "Experience/scene_evaluations.json"
    with open(output_file, "w",encoding="utf-8") as f:
        json.dump(cluster_results, f, ensure_ascii=False, indent=4) # json.dump(cluster_results, f, indent=4) # 

    print(f"Evaluation results saved to {output_file}")


# 使用例
# 仮のデータフレーム作成
data = {
    'Content': [
        "Alice saw a rabbit running. She followed it down the hole.",
        "The rabbit was talking about being late.",
        "Alice fell down the rabbit hole and encountered strange creatures.",
        "She was questioning where she was.",
        "Alice met the Cheshire cat, who gave her cryptic advice."
    ],
    'cluster': [0, 0, 1, 1, 2]  # 仮のクラスタID
}
df = pd.read_csv("Experience/df.csv")

# シーンごとに評価を行い、結果をJSONファイルとして保存
process_clusters_and_evaluate(df, text_column="Content", cluster_column="cluster")
