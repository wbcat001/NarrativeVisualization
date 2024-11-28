from typing import List, Optional
import re

class Tag:
    ACTION = "Action"
    SCENERY = "View"
    DIALOGUE = "Dialogue"
    # CHARACTER = "Name"
    LOCATION = "Location"
    DEFAULT = "default"
    STATEMENT = "Statement"

class TaggedContent:
    def __init__(self, tag: Tag, content: str, name: str):
        self.tag = tag
        self.content = content
        self.other = {}
        self.name = name

    def __repr__(self):
        return f"TaggedContent(tag={self.tag}, content='{self.content}')"

def is_scene(text):

    pattern = r'^(INT\.|EXT\.)\s+(.+)$'
    match = re.match(pattern, text)

    return match

class MovieScriptParser:
    

    def __init__(self, script: str):
        self.script = script
        self.lines: List[TaggedContent] = []
        # self.parser = ""

        self.sentiment_analyzer = None

        

    def parse(self):
        # init
        self.lines = []

        """
        indent: 15 or 20
        name: 37
        """
        lines = self.script.splitlines()
        print(f"script lines: {len(lines)}")
        tmp_content = ""
        tmp_tag = Tag.DEFAULT
        tmp_name = ""
        blank_count = 0

        for line in lines:
            # line = line.strip()

            if not line:
                if tmp_content != "":
                    blank_count += 1
                    # reset
                    if blank_count >= 3:
                        blank_count = 0
                        self.lines.append(TaggedContent(tmp_tag, tmp_content, tmp_name))
                        tmp_content = ""
                        tmp_tag = Tag.DEFAULT
                        tmp_name = ""
                    # print(blank_count)
                continue
            else:
                blank_count = 0

            
            indent = len(line) - len(line.lstrip())
            striped_text = line.strip()
            

            if tmp_tag != Tag.DEFAULT:
                tmp_content += " " + striped_text
            
            else:
                if indent == 37:
                    tmp_tag = Tag.DIALOGUE
                    tmp_name = striped_text
                    # tmp_content = " "
                
                elif indent == 25:
                    tmp_tag  = Tag.DIALOGUE
                    tmp_content += striped_text

                elif indent == 15:
                    match = is_scene(striped_text)
                    if match:
                        # match.groups() -> ( "INT.", "Location")
                        tmp_tag = Tag.LOCATION
                        tmp_content += striped_text
                    else:
                        tmp_tag = Tag.STATEMENT
                        tmp_content += striped_text
                else:
                    print(("?", striped_text))
                    tmp_tag = Tag.DEFAULT
                    tmp_content += striped_text

            

            
            
        
        print(f"length of script is : {len(self.lines)}")
        print("parse is finished")

        return
   
    def set_sentiment(self, sentiment_analyzer):
        # List[str] -> List[number]
        self.sentiment_analyzer = sentiment_analyzer
    def add_sentiment(self):
        result = self.sentiment_analyzer([line.content for line in self.lines])
           
        for index, r in enumerate(result):
            self.lines[index].other["sentiment"] = r

        return

    def get_element(self):
        return
    
    def show_lines(self, num):
        num = min(num, len(self.lines))
        for i in range(num):
            print(self.lines[i])

"""
EXT. NAME

DIALOG
ACTION
"""
from transformers import pipeline
if __name__ == "__main__":
    with open("data/Star-Wars-A-New-Hope.txt", "r") as f:
        text = f.read()

    

    parser = MovieScriptParser(text)

    parser.parse()
    parser.show_lines(100)