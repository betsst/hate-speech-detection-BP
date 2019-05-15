import codecs
import csv
from torchtext import data as torchdata

dataset_path = ''
TEXT = torchdata.Field(tokenize=lambda x: x, sequential=True, lower=True, batch_first=True)
LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

tabular_dataset = torchdata.TabularDataset(dataset_path, format='tsv', fields=[('label', LABEL), ('text', TEXT)])

for dataset, post in zip(tabular_dataset.split(split_ratio=[0.7, 0.1, 0.2]), ['train', 'valid', 'test']):
    with codecs.open(dataset_path + post + '.tsv', "w", "utf-8") as file:
        w = csv.writer(file, delimiter='\t', lineterminator='\n')
        for example in dataset.examples:
            w.writerow([example.label, example.text])
