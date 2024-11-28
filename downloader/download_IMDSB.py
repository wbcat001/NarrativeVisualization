import requests
from bs4 import BeautifulSoup

def download_script(movie_title):
    # IMSDBのURL
    base_url = "https://www.imsdb.com/scripts/"
    movie_title_formatted = movie_title.replace(" ", "-")  # 映画タイトルをURL形式に変換
    script_url = f"{base_url}{movie_title_formatted}.html"
    
    # スクリプトページを取得
    response = requests.get(script_url)
    
    # スクリプトページが存在するか確認
    if response.status_code != 200:
        print(f"Error: Could not find script for '{movie_title}'. Status code: {response.status_code}")
        return
    
    # BeautifulSoupでページを解析
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # スクリプトのコンテンツを抽出
    script_section = soup.find('pre')
    
    if script_section is None:
        print(f"Error: Script not found on the page for '{movie_title}'.")
        return
    
    script_text = script_section.get_text()
    
    # スクリプトをファイルに保存
    with open(f"{movie_title_formatted}.txt", 'w', encoding='utf-8') as file:
        file.write(script_text)
    
    print(f"Script for '{movie_title}' has been downloaded successfully!")

# 使用例
movie_title = "Star-Wars-A-New-Hope"  # ここに映画のタイトルを入力
download_script(movie_title)
