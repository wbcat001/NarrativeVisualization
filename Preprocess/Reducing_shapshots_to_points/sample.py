import numpy as np
import pandas as pd
import pickle

"""
人工ネットワークデータの前処理
- 正規化
- pickle形式で保存
- アノテーションデータの作成

"""

def load_data(file_path: str) -> np.ndarray:

    """data format
    最初の行はノード数、タイムスタンプ数
    二行目以降はタイムスタンプ数文の行
    """
    df = pd.read_csv(file_path, header=None, skiprows=1)
    data_np_array = df.to_numpy()
    print(len(df))
    
    return data_np_array

def normalize(data: np.ndarray) -> np.ndarray:
    stds = np.std(data, axis=0)
    means = np.mean(data, axis=0)
    # 標準偏差が 0 の列はそのまま
    normalized_data = np.where(stds == 0, 0, (data - means) / stds)
    return normalized_data

def get_annotation(data: np.ndarray) -> np.ndarray:

    data_index = np.arange(len(data))
    print(data_index.shape)
    data_relation_num = np.sum(data, axis=1)
    print(data_relation_num.shape)
    # relation_numのmin, max をもとに値を10のレベルに分類、カテゴリカルデータを作成
    data_active_level = pd.cut(data_relation_num, 10, labels=False)
    print(len(data_active_level))

    df = pd.DataFrame([data_index, data_relation_num, data_active_level]).T
    df.columns = ["_Index", "relation_num", "active_level"]

    return df

if __name__ == "__main__":
    file_path = "data/dynamic_network/evolution-graph-100-900-recurr.csv"

    data = load_data(file_path)
    print(data.shape)

    normalized_data = normalize(data)
    print(normalized_data.shape)

    with open("data/network/0/paragraph_embedding.pkl", "wb") as f:
        pickle.dump(normalized_data, f)


    annotation = get_annotation(data)
    print(annotation[:10])

    # min maxを表示
    print(annotation["active_level"].min())
    print(annotation["active_level"].max())
    annotation.to_csv("data/network/0/df.csv")