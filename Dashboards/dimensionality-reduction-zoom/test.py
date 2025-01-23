import pandas as pd

# サンプルデータフレームの作成
data = {
    "text": ["A", "B", "C", "D", "E", "F", "G"],
    "value": [10, 20, 30, 40, 50, 60, 70],
}
df = pd.DataFrame(data)

# ウィンドウ幅とストライド幅を指定
window = 3
stride = 2

# 畳み込み処理
result = []
for start in range(0, len(df), stride):
    end = start + window
    if end <= len(df):
        # 各ウィンドウの最初の行を参照
        first_row = df.iloc[start]
        result.append(first_row)

# 結果を新しいデータフレームとして構築
result_df = pd.DataFrame(result)

# 結果を表示
print(result_df)
