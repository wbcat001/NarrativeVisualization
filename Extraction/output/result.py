import json
file_path = "Extract/output/event.json"
with open(file_path, "r") as f:
    data = json.load(f)
count = 0
for d in data:
    count += len(d)
print(f"event data length: {count}")