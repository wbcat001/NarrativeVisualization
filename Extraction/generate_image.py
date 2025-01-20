from openai import OpenAI
import requests
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="A vibrant jungle scene in 19th-century colonial India, featuring a tall, refined European man wearing a formal suit with a top hat, seated calmly on a large Asian elephant. Next to him, a shorter, animated man with a lively expression and dressed in simpler 19th-century attire gestures energetically. The elephant is adorned with an ornate saddle-like structure, and a local guide dressed in traditional Indian clothing leads the elephant with a rope. The jungle is lush with tall trees, thick undergrowth, and vines hanging from branches. Rays of sunlight filter through the dense canopy, illuminating the travelers as they make their way through the serene yet adventurous wilderness",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url


# 画像を保存
image_response = requests.get(image_url)
if image_response.status_code == 200:
    with open("Extract/img/80_raw.png", "wb") as f:
        f.write(image_response.content)
    print("画像を保存しました: generated_image.png")
else:
    print("画像のダウンロードに失敗しました。")
