import numpy as np
import re
from labMTsimple.storyLab import emotionFileReader
import pandas as pd
class SentimentCalculator:
# load lexicon(NRC-VAD)
    def __init__(self):
        self.lexicon = pd.read_csv("Visualize/Sentiment/Lexicon/NRC-VAD-Lexicon.txt", sep='\t')
        
        self.arousal_dict = self.lexicon[["Word", "Arousal"]].set_index("Word")["Arousal"].to_dict()

        self.valence_dict =  self.lexicon[["Word", "Valence"]].set_index("Word")["Valence"].to_dict()

        self.dominance_dict = self.lexicon[["Word", "Dominance"]].set_index("Word")["Dominance"].to_dict()

        labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0,returnVector=True)
        self.labMT_dict = dict(zip(labMTwordList, labMTvector))

    def calculate_sentitment_score(self, sentence, lexicon_dict):
        ## get list of words in sentence

        words = re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower().split(" ")
        # words = [re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in words]
       
        sentiment_values = [ lexicon_dict[word] for word in words if word in lexicon_dict]
      
 
        if sentiment_values:
            return np.mean(sentiment_values)
        else:
            return None
        
    def calculate_arousal(self, sentence):
        return self.calculate_sentitment_score(sentence, self.arousal_dict)
    
    def calculate_dominance(self, sentence):
        return self.calculate_sentitment_score(sentence, self.dominance_dict)
    
    def calculate_valence(self, sentence):
        return self.calculate_sentitment_score(sentence, self.valence_dict)

    def calculate_labMT(self, sentence):
        return self.calculate_sentitment_score(sentence, self.labMT_dict)


if __name__ == "__main__":
    import json
    import pandas as pd

    file_path = "data/paragraph_alice.csv"
    
    sentiment_calculator = SentimentCalculator()
    df = pd.read_csv(file_path, index_col=0)
    df["Sentiment"] = df["Content"].apply(sentiment_calculator.calculate_labMT)
    
    df["Location"].apply(str)
    import plotly.express as px
    
    df["Sentiment_mean"] = df["Sentiment"].dropna().rolling(window=50, min_periods=1).mean()
    df["customdata"] = df["Event"]
    px.line(df, y="Sentiment_mean", hover_data=["Event"]).show()
