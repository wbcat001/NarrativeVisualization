import numpy as np
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import random

# テキストとその重要度のリスト
texts = ["Data Science", "Machine Learning", "AI", "Deep Learning", "Neural Networks", 
         "Natural Language Processing", "Computer Vision", "Reinforcement Learning", 
         "Big Data", "Robotics"]
importances = [10, 8, 9, 7, 6, 5, 8, 7, 6, 9]  # 重要度リスト (重要度が高いほど中心に近づく)

# 配置する領域の設定 (ここでは簡単な2D空間で配置)
x_limit = 10  # X軸の最大範囲
y_limit = 10  # Y軸の最大範囲

# テキストのサイズを重要度に比例させる
max_importance = max(importances)
text_sizes = [importance / max_importance * 2 for importance in importances]  # サイズを0〜2の範囲にスケール

# 重なりを避けるためにテキストの配置を計算する関数
def calculate_positions(texts, importances, x_limit, y_limit):
    positions = []
    attempts = 0
    max_attempts = 100  # 再試行の最大回数

    for i, text in enumerate(texts):
        size = text_sizes[i]  # テキストのサイズ

        # 重なりを避けるために新しい位置をランダムに選ぶ
        while attempts < max_attempts:
            x = random.uniform(size, x_limit - size)  # サイズを考慮してランダム位置
            y = random.uniform(size, y_limit - size)

            # 他のテキストと重なっていないかチェック
            overlap = False
            for pos in positions:
                prev_x, prev_y, prev_size = pos
                distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                if distance < (size + prev_size):  # 重なっている場合
                    overlap = True
                    break

            if not overlap:  # 重なりがなければ配置する
                positions.append((x, y, size))
                break
            else:
                attempts += 1
                if attempts >= max_attempts:
                    print("最大再試行回数に達しました")
                    return positions

    return positions

# ポジション計算
positions = calculate_positions(texts, importances, x_limit, y_limit)

# 可視化
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, x_limit)
ax.set_ylim(0, y_limit)
ax.set_title("Wordcloud-like Layout with Non-overlapping Texts")

# テキストをプロット
for i, (x, y, size) in enumerate(positions):
    ax.text(x, y, texts[i], fontsize=size*10, ha='center', va='center')

plt.show()
