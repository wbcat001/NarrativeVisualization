import re

def parse_location(text):
    match = re.match(r"^(INT|EXT)\.\s*(.+?)\s*-\s*(.+?)(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?$", text)

    if match:
        group = match.groups()
        print(group)
        exteror = group[0]
        location = group[1]
        time = None
        for element in group[0:]:
            if (element == "Day"):
                time = element
        return exteror, location, time
    else:
        return None


text = "INT. REBEL BLOCKADE RUNNER - MAIN PASSAGEWAY - HOGE HOGE - Day - hoge"

print(parse_location(text))