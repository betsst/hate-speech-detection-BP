import torch
import torch.nn as nn


class FeatureModel(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(FeatureModel, self).__init__()
        self.pred = torch.nn.Linear(input_size, out_features=output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        # with CrossEntropy does not need to be softmaxed values because of combining nn.NLLLoss() and nn.LogSoftmax()

    def forward(self, x):
        # normalization
        x_mean = torch.mean(x, dim=1).view(-1, 1)
        x_std = torch.std(x, dim=1).view(-1, 1)
        x_normalized = (x - x_mean) / x_std
        output = self.pred(x_normalized)
        return output
