import operator

from pyphen import Pyphen
import pandas as pd
import re
import spacy
from math import log
import csv
import html
import numpy as np
from torchtext import data
import os
cwd = os.getcwd()
print(cwd)

# x_vals = data.TabularDataset('..\\..\\data\\davidson.csv', 'CSV', skip_header=True,
#                              fields=[('index', data.Field()),
#                                      ('count', data.Field()),
#                                      ('hate_speech', data.Field()),
#                                      ('offensive_language', data.Field()),
#                                      ('neither', data.Field()),
#                                      ('class', data.Field()),
#                                      ('tweet', data.Field())])

corpus = pd.read_csv("../../data/davidson.csv")
tweets_done = 0
# ways to get columns
# print(data.columns.values.tolist())
# columns = list(data.columns.values)
# columns2 = list(data)
# print(columns2)

# spacy_en = spacy.load('en', disable=['parser', 'ner'])
spacy_en = spacy.load('en')
tweets = corpus['tweet']

pos_tweets = []


documents_count = len(tweets)


def text_tokenizer(text): # create a tokenizer function
    tokens = spacy_en.tokenizer(text)
    arrayOfTokens = []
    for tok in tokens:
        if not tok.is_punct:
            arrayOfTokens.append(tok)
    return [tok.text for tok in arrayOfTokens]

def text_pos_tokenizer(text): # create a tokenizer function
    arrayOfTokens = []
    for tok in spacy_en(text):
        # print(tok.text, tok.lemma_, tok.pos_, tok.tag_, tok.dep_, tok.shape_, tok.is_alpha, tok.is_stop)
        if not tok.is_punct:
            arrayOfTokens.append(tok)
    return [tok.pos_ for tok in arrayOfTokens]

def count_syllables(word):
    pyphen_dic = Pyphen(lang='en')
    syllabled_word = pyphen_dic.inserted(word)
    return syllabled_word.count('-') + 1

def flesch_reading_ease(words_count, sentences_count, syllables_count):
    if not sentences_count or not words_count:
        return 100
    return  206.835 - 1.015 * (words_count / sentences_count) - 84.6 * (syllables_count / words_count)

def flesch_kincaid_grade_level(words_count, sentences_count, syllables_count):
    if not sentences_count or not words_count:
        return -3.40
    return  0.39 * (words_count / sentences_count) + 11.8 * (syllables_count / words_count) - 15.59

def is_retweeted(t):
    retweet_flags = len(re.findall("!*\sRT\s@\S+", t))
    t = re.sub("!*\sRT\s@\S+", "", t)
    is_retweeted = 1 if retweet_flags != 0 else 0
    return t, is_retweeted

def count_hashtags(t):
    hashtag_re = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
    hashtags_count = len(hashtag_re.findall(t))
    t = hashtag_re.sub("", t)
    return t, hashtags_count

def count_mentions(t):
    # mention_re = re.compile("(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", re.UNICODE)
    mention_re = re.compile("(?<!RT\s)@\S+", re.UNICODE)
    mentions_count = len(mention_re.findall(t))
    t = mention_re.sub("", t)
    return t, mentions_count

def count_url(t):
    urls_count = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', t))
    t = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", t)   # possibly to substitu with HEREURL
    return t, urls_count

def count_tweet_syllables(tokens):
    syllables_count = 0
    for w in tokens:
        syllables_count += count_syllables(w)
    return syllables_count

def count_sentences(t):
    sentences_count = len(re.split(r'[(?|!|.)]+', t))
    return sentences_count - 1 if sentences_count > 1 else 1

def tfidf(tokens):
    # TODO n grams not only unigrams
    unique_tokens = {}
    count_document_containing_tokens = []
    for tok in tokens:
        if tok not in unique_tokens.keys():
            unique_tokens[tok] = 1
            count_document_containing_tokens.append(corpus.tweet.str.count(tok).sum())
        else:  # TF
            unique_tokens[tok] += 1

    inverse_doc_freq = [log(documents_count/n) for n in count_document_containing_tokens ]
    term_freq = list(unique_tokens.values())
    # get the most three frequent words in document (tweet)
    three_frequent_words = [item[1] for item in sorted(((v,k) for k,v in unique_tokens.items()))[-3:]]
    return three_frequent_words, [tf * idf for tf, idf in zip(term_freq, inverse_doc_freq)]

def process_html_entities(tweet):
    tweet = html.unescape(tweet)
    return tweet

def add_embedding_vectors(words):
    v = []
    for w in words:
        token = spacy_en(w)
        if token.has_vector:
            # TODO only 300 dimensions
            v = v + token.vector.tolist()
    return v

def get_tweet_features(t, tweet_index):
    one_var = []
    # one_var.append(tweet_index)
    print(f'Tweet index: {tweet_index}')
    t, is_tweet_retweeted = is_retweeted(t)
    one_var.append(is_tweet_retweeted)

    t, hashtags_count = count_hashtags(t)
    one_var.append(hashtags_count)

    t, mentions_count = count_mentions(t)
    one_var.append(mentions_count)

    t, urls_count = count_url(t)
    one_var.append(urls_count)

    # count of uppercase characters
    upper_count = sum(map(str.isupper, t))
    one_var.append(upper_count)

    t = re.sub(r"^\s+", "", t)
    tokens = text_tokenizer(t)
    words_count = len(tokens)
    one_var.append(words_count)

    sentences_count = count_sentences(t)
    syllables_count = count_tweet_syllables(tokens)

    # Flesch reading ease
    one_var.append(flesch_reading_ease(words_count, sentences_count, syllables_count))
    # Flesch–Kincaid grade level
    one_var.append(flesch_kincaid_grade_level(words_count, sentences_count, syllables_count))

    three_frequent_words, tfidf_values = tfidf(tokens)
    print(tfidf_values)
    one_var = one_var + tfidf_values

    one_var = one_var + add_embedding_vectors(three_frequent_words)
    # TODO pos tf idf
    pos = text_pos_tokenizer(t)
    pos_tweets.append(' '.join(pos))
    print(t)
    print(one_var)
    return one_var

reduce_to = 500
def get_x_values():
    x_values = []
    for t, i in zip(tweets[:reduce_to], range(documents_count)):
        x_values.append(get_tweet_features(t, i))
        if i == reduce_to:
            break
    return x_values

def get_y_values():
    classes = corpus['class'].tolist()
    # for i in range(len(classes)):
    c = []
    # one hot encoding
    for i in range(0, reduce_to):
        if classes[i] == 0:  # if offensive then be hate or offensive
            c.append([1, 0, 0])
        if classes[i] == 1:  # if offensive then be hate or offensive
            c.append([0, 1, 0])
        if classes[i] == 2:  # if offensive then be hate or offensive
            c.append([0, 0, 1])
    print("Y vals done\n")
    return c

def classes_summmary():
    neither_count = len(corpus[corpus['class'] == 2])
    hatered_count = len(corpus[corpus['class'] == 0])
    offensive_count = len(corpus[corpus['class'] == 1])
    hatered_percentage = hatered_count / documents_count
    offensive_percentage = offensive_count / documents_count
    neither_percentage = neither_count / documents_count
    print(f"Count of hateful tweets: {hatered_count}/{documents_count}         {hatered_percentage} ({hatered_percentage * 100}%)")
    print(f"Count of offensive tweets: {offensive_count}/{documents_count}      {offensive_percentage} ({offensive_percentage * 100}%)")
    print(f"Count of neither_count tweets: {neither_count}/{documents_count}   {neither_percentage} ({neither_percentage * 100}%)")

    weights = [hatered_count, offensive_count, neither_count]
    max_count = max(weights)
    print([w / max_count for w in weights])
    return [w / max_count for w in weights]

def save_x_values(l):
    with open("../../data/x_vals.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(l)

for index, row in corpus.iterrows():
    corpus.at[index, 'tweet'] = process_html_entities(row['tweet'])

# classes_summmary()
# x_vals = get_x_values()
# save_x_values(x_vals)
# y_vals = get_y_values()
