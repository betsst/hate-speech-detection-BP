import codecs
import csv
from tqdm import tqdm

import spacy
import torchtext


spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def save_tfs(vocab, filenamepath):
    with codecs.open(filenamepath, "w", "utf-8") as file:
        w = csv.writer(file, delimiter='\t', lineterminator='\n')
        for key, val in vocab.items():
            w.writerow([key, val])


def get_tfs(save_to_file, filename):
    TEXT = torchtext.data.Field(tokenize='spacy', sequential=True, lower=True)
    tabular_dataset = torchtext.data.TabularDataset(path='../data/tweets_with_html_ent.tsv', format='tsv',
                                                    fields=[('tweet', TEXT)])
    vocab = {}
    pbar = tqdm(total=len(tabular_dataset))

    for example in tabular_dataset:
        tweet = example.tweet
        tokens_in_tweet = []
        for token in tweet:
            if token in vocab.keys():
                if token not in tokens_in_tweet:
                    vocab[token] += 1     # document count which contain item
            else:  # new token
                vocab[token] = 1
            tokens_in_tweet.append(token)
        pbar.update(1)

    if save_to_file:
        save_tfs(vocab, filename)

    return vocab
