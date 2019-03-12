import torch
import torch.nn as nn


class BaselineModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineModel, self).__init__()
        # simple linear layer
        self.linear = torch.nn.Linear(input_size, out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # normalization
        x_mean = torch.mean(x, dim=1).view(-1, 1)
        x_std = torch.std(x, dim=1).view(-1, 1)
        x_normalized = (x - x_mean) / x_std
        output = self.linear(x_normalized)
        # with CrossEntropy does not need to be softmaxed values because of combining nn.NLLLoss() and nn.LogSoftmax()
        # output = self.softmax(output)
        return output
