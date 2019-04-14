import csv
import json
from math import log10

import torch
from torchtext import data as torchdata
from torchtext.data import TabularDataset

with open('config.json', 'r') as f:
    config = json.load(f)


class BOWField(torchdata.Field):
    def __init__(self, device, term_freqs=None):
        super(BOWField, self).__init__(tokenize='spacy', lower=True)
        self.device = device
        # TERM = torchdata.Field()
        # COUNT = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)
        # self.df_dataset = TabularDataset(path=config['dataset_file'], format='tsv',
        #                                  fields=[('count', COUNT), ('term', TERM)])
        if term_freqs is None:
            self.df_counts = {}
            with open(config['doc_freq_file'], 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    self.df_counts[row[0]] = int(row[1])
        else:
            self.df_counts = term_freqs
        self.feature_count = len(self.df_counts)

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=self.device)

        var = []
        for tokens in arr:
            tokens_count = dict([(t, tokens.count(t)) for t in set(tokens)])  # term freq
            bow = dict.fromkeys(self.df_counts, 0.0)
            for token, count in tokens_count.items():
                if token not in self.df_counts.keys():
                    continue
                bow[token] = count * log10(self.feature_count / self.df_counts[token])
            var.append(list(bow.values()))

        var = torch.FloatTensor(var).to(self.device).float()

        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def get_features_count(self):
        return self.feature_count
