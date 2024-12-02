from openai import OpenAI
import requests
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="A serene summer day, with a beautifully detailed backdrop of a grassy bank by a river. A little girl named Alice, with blonde hair in a blue dress, sits beside her sister. Alice looks frustrated and bored, glancing at a book without pictures. Her expression shows her desire for adventure, surrounded by lush greenery and flowers, with an air of stillness and quiet longing for excitement.",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url


# 画像を保存
image_response = requests.get(image_url)
if image_response.status_code == 200:
    with open("Extract/img/generated_image.png", "wb") as f:
        f.write(image_response.content)
    print("画像を保存しました: generated_image.png")
else:
    print("画像のダウンロードに失敗しました。")
