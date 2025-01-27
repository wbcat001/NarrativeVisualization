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
    # stds = np.std(data, axis=0)
    # means = np.mean(data, axis=0)
    # # 標準偏差が 0 の列はそのまま
    # normalized_data = np.where(stds == 0, 0, (data - means) / stds)

    # print(np.amax(normalized_data, axis=0)[:10])
    row_norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / row_norms
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

# 行列を増やす
def add_line_matrix(data: np.ndarray) -> np.ndarray:
    """
    行列を受け取り、元の行列ににた行列を追加する （2倍の行列を返す）
    """

    add_data = np.zeros_like(data)
    add_data[:,:9000] = data[:,:9000]
   
    # add_data[:, 2] = data
    # add_data[:, :5000] = 0
    # 結合
    add_data = np.concatenate([data, add_data], axis=0)
    # add_data = np.concatenate([add_data, add_data2], axis=0)
    print("added", add_data.shape)

    return add_data

if __name__ == "__main__":
    file_path = "data/dynamic_network/evolution-graph-100-900-recurr.csv"

    data = load_data(file_path)
    print(data.shape)

    data = add_line_matrix(data)

    normalized_data = normalize(data)
    print(normalized_data.shape)
    

    with open("data/network/1/paragraph_embedding.pkl", "wb") as f:
        pickle.dump(normalized_data, f)


    annotation = get_annotation(data)
    print(annotation[:10])

    # min maxを表示
    print(annotation["active_level"].min())
    print(annotation["active_level"].max())
    annotation.to_csv("data/network/1/df.csv")