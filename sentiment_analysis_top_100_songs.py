import pandas as pd
import time
from nltk.sentiment import SentimentIntensityAnalyzer

#t0 = time.time()

top_100 = pd.read_csv('/Users/brit.cava/Desktop/top100.csv')

text = top_100['Word']
scores = []
words = []
sid = SentimentIntensityAnalyzer()
word_score_dict = {}

for word in text:
    words.append(word)
    ss = sid.polarity_scores(word)
    scores.append(ss['compound'])
    
for i in range(len(words)):
    word_score_dict[words[i]] = scores[i]

#t1 = time.time()

#total = t1-t0

print(total)
    
df = pd.DataFrame(list(word_score_dict.iteritems()), columns=['word','score'])

df.to_csv("/Users/brit.cava/Desktop/Scores.csv", index=False, cols=('word','score'))
