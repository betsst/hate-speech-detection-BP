import csv
import json
import re

from random import choice
from tqdm import tqdm
from math import log10
from pyphen import Pyphen
import pandas as pd
import re
import spacy
from math import log
import html
import numpy as np
from torchtext import data as torchdata
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torchtext import data as torchdata
from torchtext.data import TabularDataset
from nltk.tokenize.punkt import PunktSentenceTokenizer


def tokenizer(text):
    return text


class FeatureField(torchdata.Field):
    def __init__(self, device, features, ngrams_range):
        super(FeatureField, self).__init__(tokenize=tokenizer, lower=True)
        self.features = features
        self.ngrams_range = ngrams_range
        self.device = device
        self.sentence_tokenizer = PunktSentenceTokenizer()
        # spacy_en = spacy.load('en', disable=['parser', 'ner'])
        self.spacy_en = spacy.load('en')
        self.words_tfidf_features = {}
        self.pos_features = {}
        self.pos_features_count = 0
        self.words_tfidf_features_count = 0

        if 'pos_ngrams' in self.features:
            self.pos_tfidf = TfidfVectorizer(ngram_range=tuple(self.ngrams_range['pos']))
        if 'word_ngrams' in self.features:
            self.tfidf = TfidfVectorizer(ngram_range=tuple(self.ngrams_range['word']))

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            tmp_list = []
            tmp_list.append(x[-max_len:] if self.truncate_first else x[:max_len])
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                tmp_list +
                ([] if self.eos_token is None else [self.eos_token]))
        return padded

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=self.device)

        var = [self.extract_features(doc) for doc in arr]
        var = torch.FloatTensor(var).to(self.device).float()

        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def get_ngram_features(self, examples):
        pos_tweets = {}
        tweets = []
        pbar = tqdm(total=len(examples))
        pbar.set_description('Extracting ngram features')
        for example in examples:
            if 'pos_ngrams' in self.features:
                tweet_pos = self.pos_tokenizer(example)
                pos_tweets[example] = ' '.join(tweet_pos)
            if 'word_ngrams' in self.features:  # word n-grams
                tweets.append(example)
            pbar.update(1)

        if 'pos_ngrams' in self.features:
            pos_ngrams = self.pos_tfidf.fit_transform(pos_tweets.values())
            print(f'pos features {pos_ngrams.shape[1]}')
            for tweet_pos_ngrams, tweet in zip(pos_ngrams, pos_tweets.keys()):
                self.pos_features[tweet] = tweet_pos_ngrams.toarray()[0].tolist()
            self.pos_features_count = len(self.pos_features.get(choice(list(self.pos_features))))
            self.pos_features_vocab = self.pos_tfidf.vocabulary_
        if 'word_ngrams' in self.features:
            tfidf_ngrams = self.tfidf.fit_transform(tweets)
            print(f'word tfidf features {tfidf_ngrams.shape[1]}')
            for tweet_tfidf_ngrams, tweet in zip(tfidf_ngrams, tweets):
                self.words_tfidf_features[tweet] = tweet_tfidf_ngrams.toarray()[0].tolist()
            self.words_tfidf_features_count = len(self.words_tfidf_features[choice(list(self.words_tfidf_features))])
            self.pos_features_vocab = self.tfidf.vocabulary_

    def extract_features(self, doc):
        var = []
        document = doc[0]

        if 'is_retweeted' in self.features:
            modified_document, is_retweeted = self.is_retweeted(document)
            var.append(is_retweeted)
        if 'count_mentions' in self.features:
            modified_document, mentions_count = self.count_mentions(modified_document)
            var.append(mentions_count)
        if 'count_urls' in self.features:
            modified_document, urls_count = self.count_url(modified_document)
            var.append(urls_count)
        if 'count_hashtags' in self.features:
            modified_document, hashtags_count = self.count_hashtags(modified_document)
            var.append(hashtags_count)
        if 'count_uppercase' in self.features:
            var.append(sum(map(str.isupper, modified_document)))  # count of uppercase characters

        sentences_count = len(self.sentence_tokenizer.tokenize(document))

        # document = re.sub(r"^\s+", "", document)
        # document = re.sub(r"^\"\s+", "\"", document)
        tokens = self.text_tokenizer(document)
        words_count = len(tokens)
        if 'count_words' in self.features:
            var.append(words_count)   # words count
        syllables_count = self.count_tweet_syllables(tokens)

        if 'flesch_reading_ease' in self.features:  # Flesch reading ease
            var.append(self.flesch_reading_ease(words_count, sentences_count, syllables_count))
        if 'flesch_kincaid_grade_level' in self.features:   # Flesch–Kincaid grade level
            var.append(self.flesch_kincaid_grade_level(words_count, sentences_count, syllables_count))

        if 'pos_ngrams' in self.features:
            try:
                var += self.pos_features[document]
            except KeyError:
                tweet_pos = self.pos_tokenizer(document)
                var += self.pos_tfidf.transform(tweet_pos).toarray()[0].tolist()

        if 'word_ngrams' in self.features:  # word n-grams
            try:
                var += self.words_tfidf_features[document]
            except KeyError:
                var += self.tfidf.transform(tweet_pos).toarray()[0].tolist()
        return var

    def pos_tokenizer(self, text):
        array_of_tokens = []
        for tok in self.spacy_en(text):
            # print(tok.text, tok.lemma_, tok.pos_, tok.tag_, tok.dep_, tok.shape_, tok.is_alpha, tok.is_stop)
            if not tok.is_punct:
                array_of_tokens.append(tok)
        return [tok.pos_ for tok in array_of_tokens]

    def text_tokenizer(self, text):  # create a tokenizer function
        tokens = self.spacy_en.tokenizer(text)
        array_of_tokens = []
        for tok in tokens:
            if tok.text == ' ' or tok.lower == ' ':
                continue
            if not tok.is_punct:
                array_of_tokens.append(tok)
        return [tok.text.lower() for tok in array_of_tokens]

    @staticmethod
    def is_retweeted(t):
        retweet_flags = len(re.findall("!*\sRT\s@\S+", t))
        t = re.sub("!*\sRT\s@\S+", "", t)
        is_retweeted = 1 if retweet_flags != 0 else 0
        return t, is_retweeted

    @staticmethod
    def flesch_reading_ease(words_count, sentences_count, syllables_count):
        if not sentences_count or not words_count:
            return 100
        sentences_count = 1
        return 206.835 - 1.015 * (words_count / sentences_count) - 84.6 * (syllables_count / words_count)

    @staticmethod
    def flesch_kincaid_grade_level(words_count, sentences_count, syllables_count):
        if not sentences_count or not words_count:
            return -3.40
        # sentences_count = 1  #
        return 0.39 * (words_count / sentences_count) + 11.8 * (syllables_count / words_count) - 15.59

    @staticmethod
    def count_syllables(word):
        pyphen_dic = Pyphen(lang='en')
        syllabled_word = pyphen_dic.inserted(word)
        return syllabled_word.count('-') + 1

    def count_tweet_syllables(self, tokens):
        syllables_count = 0
        for w in tokens:
            syllables_count += self.count_syllables(w)
        return syllables_count

    @staticmethod
    def count_hashtags(t):
        hashtag_re = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
        hashtags_count = len(hashtag_re.findall(t))
        t = hashtag_re.sub("", t)
        return t, hashtags_count

    @staticmethod
    def count_mentions(t):
        # mention_re = re.compile("(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", re.UNICODE)
        mention_re = re.compile("(?<!RT\s)@\S+", re.UNICODE)
        mentions_count = len(mention_re.findall(t))
        t = mention_re.sub("", t)
        return t, mentions_count

    @staticmethod
    def count_url(t):
        urls_count = len(
            re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', t))
        t = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", t)
        return t, urls_count

    def get_features_count(self):
        features_count = 0
        if 'is_retweeted' in self.features:
            features_count += 1
        if 'count_mentions' in self.features:
            features_count += 1
        if 'count_urls' in self.features:
            features_count += 1
        if 'count_hashtags' in self.features:
            features_count += 1
        if 'count_uppercase' in self.features:
            features_count += 1
        if 'count_words' in self.features:
            features_count += 1
        if 'flesch_reading_ease' in self.features:
            features_count += 1
        if 'flesch_kincaid_grade_level' in self.features:
            features_count += 1
        if 'pos_ngrams' in self.features:
            features_count += self.pos_features_count
        if 'word_ngrams' in self.features:
            features_count += self.words_tfidf_features_count

        return features_count
