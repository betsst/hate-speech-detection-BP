import json
import torch
import torch.nn as nn


with open('config.json', 'r') as f:
    config = json.load(f)
    config_model = config['model']

class CharCNNModel(nn.Module):
    # Our batch shape for input x is (batch_size, 70, 1024)
    def __init__(self, l0=config_model['l0'], batch_size=config_model['batch_size']):
        super(CharCNNModel, self).__init__()
        self.l0 = l0
        self.batch_size = batch_size
        self.dropout_prob = config_model['dropout']
        self.in_channels = 1
        if config_model['feature_size'] == 'small':
            self.conv_feature = 256
            self.output_units = 1024
        elif config_model['feature_size'] == 'large':
            self.conv_feature = 1024
            self.output_units = 2048
        else:
            raise Exception('feature_size in config should be either small or large')

        # 6 convolutional layers
        self.conv11d = nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=7, stride=1)
        self.conv1pool = nn.MaxPool1d(kernel_size=3)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.conv_feature, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        # 3 fully connected layers
        self.lin1 = nn.Sequential(
            nn.Linear(self.compute_dim(), self.output_units),
            # maybe relu?
            nn.Dropout(p=self.dropout_prob)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(self.output_units, self.output_units),
            # maybe relu?
            nn.Dropout(p=self.dropout_prob)
        )
        self.lin3 = nn.Linear(self.output_units, config_model['num_classes'])

    # dimension computation for layer 7
    def compute_dim(self):
        return ((self.l0 - 96) // 27) * self.conv_feature

    def forward(self, x):
        x = torch.randn(self.batch_size, len(config['data_processing']['alphabet']), self.l0)
        xt = x.transpose(1, 2)
        out = self.conv11d(xt)
        out = self.conv1pool(out)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.lin1(x)
        x = self.lin2(x)
        out = self.lin3(x)

        return out
