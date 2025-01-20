import pandas as pd
import re
import numpy as np

class SentimentCalculator:
# load lexicon(NRC-VAD)
    def __init__(self):
        self.lexicon = pd.read_csv("Visualize/Sentiment/Lexicon/NRC-VAD-Lexicon.txt", sep='\t')
        self.lexicon = self.lexicon[["Word", "Arousal"]]
        self.arousal_dict = self.lexicon.set_index("Word")["Arousal"].to_dict()

    def calculate_sentitment_score(self, sentence, lexicon_dict):
        ## get list of words in sentence
       
        words = sentence.lower().split(" ")
        # words = [re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in words]
       
        sentiment_values = [ lexicon_dict[word] for word in words if word in lexicon_dict]
 
        if sentiment_values:
            return np.mean(sentiment_values)
        else:
            return None
        
    def calculate_arousal(self, sentence):
        return self.calculate_sentitment_score(sentence, self.arousal_dict)
        


if __name__ == "__main__":
    sentence = "I am excited and happy to see you!"
    sentiment_calculator = SentimentCalculator()
    print(sentiment_calculator.calculate_arousal(sentence))


