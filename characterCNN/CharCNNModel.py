import json
import torch
import torch.nn as nn


with open('config.json', 'r') as f:
    config = json.load(f)
    config_model = config['model']


class CharCNNModel(nn.Module):
    def __init__(self, l0=config_model['l0'], batch_size=config_model['batch_size']):
        super(CharCNNModel, self).__init__()
        self.l0 = l0
        self.alphabet_size = len(config['data_processing']['alphabet'])
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
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.alphabet_size, self.conv_feature, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1),
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
        # batch shape for model input x should be (batch_size, 70, 1014)
        # x = torch.randn(self.batch_size, len(config['data_processing']['alphabet']), self.l0) # for debug
        xt = x.transpose(1, 2)

        out = self.conv1(xt)
        print('shape after CONV1 ' + str(out.shape))
        out = self.conv2(out)
        print('shape after CONV2 ' + str(out.shape))
        out = self.conv3(out)
        print('shape after CONV3 ' + str(out.shape))
        out = self.conv4(out)
        print('shape after CONV4 ' + str(out.shape))
        out = self.conv5(out)
        print('shape after CONV5 ' + str(out.shape))
        out = self.conv6(out)
        print('shape after CONV6 ' + str(out.shape))

        out = out.view(self.batch_size, out.shape[1] * out.shape[2])
        print('shape to dense ' + str(out.shape))
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)

        return out
