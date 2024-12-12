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
    chapter_pattern = re.compile(r'^(Chapter|CHAPTER|第[一二三四五六七八九十]+章)[^\\n]*', re.MULTILINE)
    p = re.compile('[\u2160-\u217F]+')
    is_body = False
  
    data = []
    current_chapter = 0

    for para in paragraphs:
        # if re.search(footer_end_pattern, para):
        #     is_body = False

    
        # 空行や短い行を除外
        para = para.strip()
        if not para or len(para) < 10:
            continue

        # 章のタイトルを更新
        chapter_match = chapter_pattern.match(para)
        if chapter_match:
            # current_chapter = para.strip()
            current_chapter += 1
        
        # データリストに追加
        data.append({"Chapter": current_chapter, "Content": para})

        # if re.search(header_start_pattern, para):
        #     is_body = True
        
    # DataFrame作成
    df = pd.DataFrame(data)
    print(current_chapter)

    return df

# メイン処理
def main():
    name = "80days"
    file_path = f"data/data_gutenberg/{name}.txt"  # ローカルのテキストファイルのパスを指定
    
    # テキストを読み込む
    text = read_text_file(file_path)
    
    # 段落に分割
    paragraphs = split_into_paragraphs(text)
    
    # DataFrameに変換
    df = create_dataframe(paragraphs)
    df["Index"] = df.index
    
    # CSVとして保存
    df.to_csv(f'data/{name}s_df.csv', index=True)
    print("Saved as 'gutenberg_text.csv'")
    print(df.head())  # デバッグ用に最初の数行を表示

if __name__ == "__main__":
    main()
