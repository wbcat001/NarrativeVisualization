import json

file_path = "Experience/scene_evaluations.json"
file_path2= "Experience/scene_evaluations3.json"
with open(file_path, "r") as file:
    data = json.load(file)

with open(file_path2, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)