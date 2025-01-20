def count_indent(line):
    # 行の先頭の空白を削除して、残りの長さを返す
    return len(line) - len(line.lstrip())

with open("data/Star-Wars-A-New-Hope.txt", "r") as f:
    text = f.read()

lines = text.splitlines()

# sample_line = "    This is an indented line    "
# indentation = count_indent(sample_line)
# print(f"Indentation: {indentation} spaces")

for i in range(400):
    if not lines[i]: 
        # reset
        continue

    indent = count_indent(lines[i])
    print(f"Indent: {indent} \n Content: {lines[i]}")



