from openai import OpenAI
client = OpenAI()
import json
import pandas as pd


def get_scene_evaluation(scene_text):
    """
    LLMにシーンテキストを渡し、要約、シーン題名、適切に区切られているか、情報が足りているかを評価する
    """
    # 質問のテンプレート
    prompt = f"""
    次のシーンについて要約してください。シーンの題名をつけ、適切に区切られているか、情報が足りているかを評価してください。

    シーンのテキスト:
    {scene_text}

    要約:
    1. シーンの要約
    2. シーンの題名
    3. シーンは適切に区切られているか、情報は足りているか？
    """

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
    response_text = response['choices'][0]['message']['content'].strip()

    # 結果を辞書形式で整理
    result = {}
    try:
        summary, title, evaluation = response_text.split("\n")
        result = {
            "summary": summary.split(":")[1].strip(),
            "title": title.split(":")[1].strip(),
            "evaluation": evaluation.split(":")[1].strip()
        }
    except Exception as e:
        print("Error parsing LLM response:", e)

    return result


def process_clusters_and_evaluate(df, text_column, cluster_column):
    """
    クラスタごとにテキストを集約し、LLMで評価を得てJSON形式で保存する関数
    """
    # 各クラスタのテキストを集約
    cluster_texts = {}
    for cluster_id in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster_id]
        cluster_text = " ".join(cluster_data[text_column].values)
        cluster_texts[cluster_id] = cluster_text

    # 各クラスタに対してLLMで評価
    cluster_results = {}
    for cluster_id, scene_text in cluster_texts.items():
        evaluation = get_scene_evaluation(scene_text)
        cluster_results[cluster_id] = evaluation

    # 結果をJSON形式で保存
    output_file = "scene_evaluations.json"
    with open(output_file, "w") as f:
        json.dump(cluster_results, f, indent=4)

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
df = pd.DataFrame(data)

# シーンごとに評価を行い、結果をJSONファイルとして保存
process_clusters_and_evaluate(df, text_column="Content", cluster_column="cluster")
