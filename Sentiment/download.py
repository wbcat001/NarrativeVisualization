## python
import re
# import nltk

def extract_body_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ヘッダーとフッターを示すキーワード
    header_start_pattern = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*'
    footer_end_pattern = r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*'

    # ヘッダーとフッターの行を見つける
    start_idx, end_idx = 0, len(lines)
    for i, line in enumerate(lines):
        if re.search(header_start_pattern, line):
            start_idx = i + 1  # ヘッダー終了行の次から本文開始
        if re.search(footer_end_pattern, line):
            end_idx = i  # フッター開始行までで本文終了
            break

    # 本文を抽出
    body_lines = lines[start_idx:end_idx]
    body_text = ''.join(body_lines)
    return body_text


def preprocess_gutenberg_text(filename):
   

    text = extract_body_text(filename)
    text = text.strip()

    # 小文字化と句読点の削除
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 句読点の削除

    # 単語にトークン化
    #words = nltk.word_tokenize(text)
    words = re.findall(r'\b\w+\b', text)

    return words


