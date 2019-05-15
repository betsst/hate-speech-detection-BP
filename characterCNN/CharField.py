import json

import torch
from torchtext import data as torchdata
from torchtext.data import get_tokenizer

with open('config.json', 'r') as f:
    config = json.load(f)


class CharField(torchdata.Field):
    def __init__(self, fix_length, lower, tokenize=(lambda s: list(s))):
        self.fix_length = fix_length
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.alphabet = config['alphabet']
        # self.alphabet.append("'")
        super(CharField, self).__init__(fix_length=self.fix_length, lower=self.lower, tokenize=self.tokenize)

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        var = []
        for elem in arr:  # process all in batch
            elem_matix = []
            for char in elem:
                v = [0] * len(self.alphabet)
                if char in self.alphabet:
                    v[self.alphabet.index(char)] = 1
                # elem_matix.insert(0, v)  # backward order
                elem_matix.append(v)  # in-order
            var.append(elem_matix)

        var = torch.tensor(var, dtype=self.dtype, device=device).float()

        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var
