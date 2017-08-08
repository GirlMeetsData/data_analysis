from nltk.sentiment import SentimentIntensityAnalyzer

text = ["This lunch was not good"]
scores = []
sid = SentimentIntensityAnalyzer()

for word in text:
    ss = sid.polarity_scores(word)
    scores.append(ss['compound'])
    print(ss)
    
# the compound score is a normalized score of sum_s and

# sum_s is the sum of valence computed based on some heuristics and a 
# sentiment lexicon (aka. Sentiment Intensity) and

# the normalized score is simply the sum_s divided by its square plus an alpha parameter that 
# increases the denominator of the normalization function.
