import codecs
import csv
from tqdm import tqdm

import spacy
import torchtext

reduce = {'do_reducing': False, 'reduce_to': 10}
spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


TEXT = torchtext.data.Field(tokenize='spacy', sequential=True, lower=True)
LABELS = torchtext.data.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)
tabular_dataset = torchtext.data.TabularDataset(path='../data/tweets_with_html_ent.tsv', format='tsv',
                                                fields=[('tweet', TEXT)])
labels_dataset = torchtext.data.TabularDataset(path='../data/davidson.tsv', format='tsv',
                                               fields=[('label', LABELS)])

vocab = {}
pbar = tqdm(total=len(tabular_dataset))
i = 0
# with codecs.open('../data/davidson_labels_html.tsv', "w", "utf-8") as file:
#     w = csv.writer(file, delimiter='\t', lineterminator='\n')
for example, example_label in zip(tabular_dataset, labels_dataset):
    pbar.set_description(f'{i}')
    pbar.update(1)
    tweet = example.tweet
    print(f'{tweet}')
    if reduce['do_reducing'] and i + 1 == reduce['reduce_to']:
        break

    tokens_in_tweet = []
    for token in tweet:
        if token in vocab.keys():
            if token not in tokens_in_tweet:
                vocab[token][1] += 1     #df
            vocab[token][0] += 1  #tf
        else:
            vocab[token] = [1, 1]
        tokens_in_tweet.append(token)
    i += 1
        # w.writerow([example_label.label, example.tweet])


with codecs.open('../data/bow_tf.tsv', "w", "utf-8") as file:
    w = csv.writer(file, delimiter='\t', lineterminator='\n')
    for key, val in vocab.items():
        w.writerow([key, val[1]])
