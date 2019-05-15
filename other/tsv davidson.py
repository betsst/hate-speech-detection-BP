import csv
import pandas as pd

corpus = pd.read_csv("../data/davidson.csv")
davidson_tweets = corpus['tweet']
tweet_class = corpus['class']

with open('../data/davidson.tsv', 'w', newline='', encoding='UTF-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    for tweet, c in zip(davidson_tweets, tweet_class):
        writer.writerow([c, tweet])
