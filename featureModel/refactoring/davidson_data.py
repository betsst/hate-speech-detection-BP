from pyphen import Pyphen
import pandas as pd
import re
import spacy
from math import log
import html
import numpy as np
from torchtext import data as torchdata
from sklearn.feature_extraction.text import TfidfVectorizer

reduce = {'do_reducing': True, 'reduce_to': 10}   # for debugging
do_feature_extraction = False
save_features = False
pos_ngrams = (1, 3)
word_ngrams = (1, 3)

corpus = pd.read_csv("../data/davidson.csv")
davidson_tweets = corpus['tweet']
documents_count = len(davidson_tweets)
# spacy_en = spacy.load('en', disable=['parser', 'ner'])
spacy_en = spacy.load('en')

tweets_done = 0


if do_feature_extraction:
    tabular_dataset = torchdata.TabularDataset('../data/tweets_with_html_ent.tsv', format='tsv',
                                               fields=[('tweet', torchdata.Field(tokenize="spacy", sequential=True,
                                                                                 lower=True))])
    tweets = [t.tweet for t in tabular_dataset.examples]

    pos_tweets = []
    processed_tweets = []
else:
    with open("../data/processed_tweets.csv", encoding="utf-8") as f:
        processed_tweets = f.read().splitlines()
    with open("../data/tweets_pos.csv", encoding="utf-8") as f:
        pos_tweets = f.read().splitlines()


def text_tokenizer(text):  # create a tokenizer function
    tokens = spacy_en.tokenizer(text)
    array_of_tokens = []
    for tok in tokens:
        if tok.text == ' ' or tok.lower == ' ':
            continue
        if not tok.is_punct:
            array_of_tokens.append(tok)
    return [tok.text.lower() for tok in array_of_tokens]


def text_pos_tokenizer(text): # create a tokenizer function
    array_of_tokens = []
    for tok in spacy_en(text):
        # print(tok.text, tok.lemma_, tok.pos_, tok.tag_, tok.dep_, tok.shape_, tok.is_alpha, tok.is_stop)
        if not tok.is_punct:
            array_of_tokens.append(tok)
    return [tok.pos_ for tok in array_of_tokens]


def count_syllables(word):
    pyphen_dic = Pyphen(lang='en')
    syllabled_word = pyphen_dic.inserted(word)
    return syllabled_word.count('-') + 1


def flesch_reading_ease(words_count, sentences_count, syllables_count):
    if not sentences_count or not words_count:
        return 100
    sentences_count = 1
    return 206.835 - 1.015 * (words_count / sentences_count) - 84.6 * (syllables_count / words_count)


def flesch_kincaid_grade_level(words_count, sentences_count, syllables_count):
    if not sentences_count or not words_count:
        return -3.40
    # sentences_count = 1  #
    return 0.39 * (words_count / sentences_count) + 11.8 * (syllables_count / words_count) - 15.59


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
    t = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", t)
    return t, urls_count


def count_tweet_syllables(tokens):
    syllables_count = 0
    for w in tokens:
        syllables_count += count_syllables(w)
    return syllables_count


def count_sentences(t):
    sentences_count = len(re.split(r'[(?|!|.)]+', t))
    return sentences_count - 1 if sentences_count > 1 else 1


def count_token_occurrences(token):
    token_in_docs = 0
    for tweet in tweets:
        if token in tweet:
            token_in_docs += 1
    return token_in_docs if token_in_docs != 0 else 1


def tfidf_unigram(tokens):
    term_freq = {}
    token_in_docs = {}
    for tok in tokens:
        if tok not in term_freq.keys():
            term_freq[tok] = 1
            token_in_docs[tok] = count_token_occurrences(tok)
        else:  # TF
            term_freq[tok] += 1

    inverse_doc_freq = {}
    for n in token_in_docs:
        inverse_doc_freq[n] = log(documents_count/token_in_docs[n])
    # term_freq = list(term_freq.values())
    # get the most three frequent words in document (tweet)
    three_frequent_words = [item[1] for item in sorted(((v,k) for k,v in term_freq.items()))[-3:]]
    tf_idf = [term_freq[tok] * inverse_doc_freq[tok] for tok in tokens]
    return three_frequent_words, tf_idf


def process_html_entities(tweet):
    tweet = html.unescape(tweet)
    return tweet


def add_embedding_vectors(words):
    v = []
    for w in words:
        token = spacy_en(w)
        if token.has_vector:
            v = v + token.vector.tolist()
    return v


def get_tweet_features(t, tweet_index):
    one_var = []
    if tweet_index % 100 == 0:
        import datetime
        print(f'Tweet index: {tweet_index} {datetime.datetime.now()}')

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
    t = re.sub(r"^\"\s+", "\"", t)
    tokens = text_tokenizer(t)
    words_count = len(tokens)
    one_var.append(words_count)

    sentences_count = count_sentences(t)
    syllables_count = count_tweet_syllables(tokens)

    # Flesch reading ease
    one_var.append(flesch_reading_ease(words_count, sentences_count, syllables_count))
    # Flesch–Kincaid grade level
    one_var.append(flesch_kincaid_grade_level(words_count, sentences_count, syllables_count))

    three_frequent_words, tfidf_values = tfidf_unigram(tokens)
    # one_var = one_var + tfidf_values

    one_var = one_var + add_embedding_vectors(three_frequent_words)

    pos = text_pos_tokenizer(t)
    pos_tweets.append(' '.join(pos))
    processed_tweets.append(t)

    # print(t)
    return one_var


def other_features():
    # word n-grams
    tfidf = TfidfVectorizer(ngram_range=word_ngrams)
    tfidf_ngrams = tfidf.fit_transform(processed_tweets)

    # pos n-grams
    pos_tfidf = TfidfVectorizer(ngram_range=pos_ngrams)
    pos_ngrams = pos_tfidf.fit_transform(pos_tweets)

    if save_features:
        np.savetxt("../data/tfidf.csv", tfidf_ngrams.toarray(), delimiter=",")
        np.savetxt("../data/pos_tfidf.csv", pos_ngrams.toarray(), delimiter=",")

    return tfidf_ngrams, pos_ngrams


def get_x_values(do_reduce=reduce['do_reducing'], reduce_to=reduce['reduce_to']):
    processed_tweets = reduce_to if do_reduce else davidson_tweets

    text_features = []
    for i, t in enumerate(processed_tweets):
        text_features.append(get_tweet_features(t, i))
    if save_features:
        np.savetxt("../data/text_features.csv", text_features, delimiter=",")

    tfidf_features = other_features()
    return text_features, tfidf_features


def get_y_values(do_reduce=reduce['do_reducing'], reduce_to=reduce['reduce_to']):
    classes = corpus['class'].tolist()
    c = []
    # one hot encoding
    process_tweets = reduce_to if do_reduce else documents_count

    for i in range(0, process_tweets):
        if classes[i] == 0:  # if offensive then be hate or offensive
            c.append([1, 0, 0])
        if classes[i] == 1:  # if offensive then be hate or offensive
            c.append([0, 1, 0])
        if classes[i] == 2:  # if offensive then be hate or offensive
            c.append([0, 0, 1])

    print(f"Labels hot encoding done ({process_tweets})\n")
    return c


def classes_summary(do_reduce=reduce['do_reducing'], reduce_to=reduce['reduce_to']):
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
    weights[weights.index(max_count)] = 1
    print([w / max_count for w in weights])
    return [w / max_count for w in weights]


if __name__ == "__main__":
    # for index, row in corpus.iterrows():
    #     corpus.at[index, 'tweet'] = process_html_entities(row['tweet'])

    if do_feature_extraction:
        get_x_values()
