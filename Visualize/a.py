"""
ガントチャートの描画のためにデータを加工する

{
"tag" : "", 
"content" : "",
"name" : "",
"sentiment?"
}
-> 
df = [
    dict(Task="Job A", Start=convert_to_datetime(0), Finish=convert_to_datetime(4)),
    dict(Task="Job B", Start=convert_to_datetime(3), Finish=convert_to_datetime(6)),
    dict(Task="Job C", Start=convert_to_datetime(6), Finish=convert_to_datetime(10)),
    dict(Task="Job A", Start=convert_to_datetime(17), Finish=convert_to_datetime(20)),
    dict(Task="Job C", Start=convert_to_datetime(19), Finish=convert_to_datetime(30))
]に変換する.


"""

from typing import List, Optional
from movie_parser import MovieScriptParser, TaggedContent, Tag
import re
import pandas as pd
import random

from timeline import draw_gantt
def parse_location(text):
    match = re.match(r"^(INT|EXT)\.\s*(.+?)\s*-\s*(.+?)(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?(?:\s*-\s*(.+?))?$", text)

    if match:
        group = match.groups()
        exterior = group[0]
        location = group[1]
        time = None
        for element in group[0:]:
            if (element == "Day"):
                time = element
    
        return exterior, location, time
    else:
        print(f"unmatch text:{text}")
        return "", "", ""
        

                
    

    # text -> location, exterior, day/night


## data
lines : List[TaggedContent] = []

with open("data/Star-Wars-A-New-Hope.txt", "r") as f:
    text = f.read()
movie_parser = MovieScriptParser(text)
movie_parser.parse()

lines = movie_parser.lines


time = "Init"
exterior = "Init"
location = "Init"
tag = "Init"
entities = set()
timeline_data = []
length = 0

for line in lines:

    ## 
    tmp_tag = line.tag
    tmp_content = line.content
    tmp_name = line.name
    tmp_length = len(tmp_content) # line.length

    if tmp_tag == Tag.LOCATION:
        
        ## reset
        if location != "":
            timeline_data.append({
                "location": location,
                "time": time,
                "exterior": exterior,
                "entities": list[entities],
                "length": length // 100,
            })
            time = ""
            exterior = ""
            location = ""
            entities = set()
            length = 0
        
        ## new scene
        # location, exterior, time

        exterior, location, time = parse_location(tmp_content)

        

    elif tmp_tag == Tag.STATEMENT:
        # add content
        length += tmp_length

    elif tmp_tag == Tag.DIALOGUE:
        # get name, add content

        length += tmp_length
        entities.add(tmp_name)

    else:
        # add content
        length += tmp_length


def assign_color(strings, color_map=None):
    if color_map is None:
        color_map = ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(len(strings))]

    return {string: color_map[i % len(color_map)] for i, string in enumerate(strings)}

  

print(f"timeine_data's length: {len(timeline_data)}")
        
for i in timeline_data[:10]:
    print(i)


######### Visulaization Test: Location TimeLine Chart
processed_data = []
prev_time = 0
colors = {}

for i, d in enumerate(timeline_data):
    row = dict(
        Task = d["location"],
        Location = d["location"],
        Time = d["time"],
        Exterior = d["exterior"],
        Start = prev_time ,
        Finish = prev_time + d["length"],

    )
    print(row)
    processed_data.append(row)

    prev_time = d["length"] + prev_time + 1





df = pd.DataFrame(processed_data)
colors = assign_color(df["Location"].unique())
draw_gantt(df, "Task",colors=colors)
