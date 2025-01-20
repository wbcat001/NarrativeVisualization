from labMTsimple.storyLab import emotionFileReader, emotion, stopper
from download import preprocess_gutenberg_text
import pandas as pd

# text length N must > N_w
def calc_stride_size(N, n, N_w=10000 ):
    N_s = (N -N_w + 1) // n
    return N_s

# text: List<str>
def calc_sentiment(text, score_dict):
    word_dict = {}
    t_length = 0 # score_dictにある単語のみにするべき
    for w in text:
        if w not in word_dict:
            word_dict[w] = 1
        else:
            word_dict[w] += 1

    sentiment_score = 0
    t_length = 0
    for w, count in word_dict.items():
        if w in score_dict:
            sentiment_score += score_dict[w] * count
            t_length += count
    print(f"valid word length: {t_length}")
    
    return sentiment_score / t_length



# Get labMT object(to make dict)
labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0,returnVector=True)

df = pd.read_csv("Sentiment/NRC-VAD-Lexicon.txt",sep='\t')

score_dict = dict(zip(labMTwordList, labMTvector))

score_dict = df[["Word", "Dominance"]].set_index("Word")["Dominance"].to_dict()



## あとは文章をきれいに単語に分けられれば良い。
# lower, strip, re.subで記号削除, 

class MyIterator:
    def __init__(self, text_list, window_size: int, stride:int ):
        self.text_list = text_list
        self.window_size = window_size
        self.stride = stride
        self.current = 0
        self.length = len(self.text_list)
    def __iter__(self):
        return self
    def __next__(self):
        if self.current + self.window_size > self.length:
            raise StopIteration
        result = self.text_list[self.current:self.current + self.window_size]
        self.current += self.stride
        
        return result
    
text_list = preprocess_gutenberg_text("Sentiment/pg11.txt")
print(f"text length: {len(text_list)}")
window_size = 10000
sequence_num = 100
stride = calc_stride_size(len(text_list), sequence_num)
myIterator = MyIterator(text_list, window_size, stride)
scores = [calc_sentiment(t, score_dict) for t in myIterator]

print(scores)

import plotly.express as px

px.line(scores).show()