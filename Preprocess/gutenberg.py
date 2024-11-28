import pandas as pd
import re

# ファイルを読み込んでテキストを取得
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# テキストを段落ごとに分割
def split_into_paragraphs(text):
    # 2行以上の改行で段落を分割
    paragraphs = re.split(r'\n\s*\n', text)
    # 各段落をトリムして空白を除去
    # paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraphs = [' '.join(p.splitlines()).strip() for p in paragraphs if p.strip()]
    return paragraphs

# DataFrameに変換
def create_dataframe(paragraphs):
    df = pd.DataFrame(paragraphs, columns=['Content'])
    df["Word_Count"] = df["Content"].apply(lambda x:  len(x.split(" ")))
    return df

# メイン処理
def main():
    file_path = "Preprocess/macbeth.txt"  # ローカルのテキストファイルのパスを指定
    
    # テキストを読み込む
    text = read_text_file(file_path)
    
    # 段落に分割
    paragraphs = split_into_paragraphs(text)
    
    # DataFrameに変換
    df = create_dataframe(paragraphs)
    
    # CSVとして保存
    df.to_csv('gutenberg_text.csv', index=True)
    print("Saved as 'gutenberg_text.csv'")
    print(df.head())  # デバッグ用に最初の数行を表示

if __name__ == "__main__":
    main()
