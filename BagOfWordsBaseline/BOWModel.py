import json

import torch
import torch.nn as nn


class BOWModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(BOWModel, self).__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(input_size, out_features=self.num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
