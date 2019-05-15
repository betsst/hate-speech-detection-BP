import json
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


with open('config.json', 'r') as f:
    config = json.load(f)


class CharCNNModel(nn.Module):
    def __init__(self, num_classes, alphabet, l0=config['l0'], batch_size=config['batch_size']):
        super(CharCNNModel, self).__init__()
        self.l0 = l0
        self.alphabet_size = len(alphabet)
        self.batch_size = batch_size
        self.dropout_prob = config['dropout']
        self.in_channels = 1
        self.dropout = torch.nn.Dropout(config['dropout'])
        self.activation = torch.nn.ReLU()
        self.model_size = config['feature_size']

        # self.conv1_activation = None

        if self.model_size == 'small':
            self.conv_feature = 256
            self.output_units = 1024
        elif self.model_size == 'large':
            self.conv_feature = 1024
            self.output_units = 2048
        else:
            raise Exception('feature_size in config should be either small or large')

        # 6 convolutional layers
        self.conv1 = nn.Conv1d(self.alphabet_size, self.conv_feature, kernel_size=7, stride=1)
        self.conv1seq = nn.Sequential(
            self.conv1,
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=7, stride=1)
        self.conv2seq = nn.Sequential(
            self.conv2,
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        self.conv3 = nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1)
        self.conv3seq = nn.Sequential(
            self.conv3,
            nn.ReLU()
        )
        self.conv4 = nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1)
        self.conv4seq = nn.Sequential(
            self.conv4,
            nn.ReLU()
        )
        self.conv5 = nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1)
        self.conv5seq = nn.Sequential(
            self.conv5,
            nn.ReLU()
        )

        self.conv6 = nn.Conv1d(self.conv_feature, self.conv_feature, kernel_size=3, stride=1)
        self.conv6seq = nn.Sequential(
            self.conv6,
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )

        # 3 fully connected layers
        self.lin1 = nn.Linear(self.compute_dim(), self.output_units)
        self.lin1seq = nn.Sequential(
            self.lin1,
            # maybe relu?
            # nn.ReLU(),
            nn.Dropout(p=self.dropout_prob)
        )
        self.lin2 = nn.Linear(self.output_units, self.output_units)
        self.lin2seq = nn.Sequential(
            self.lin2,
            # maybe relu?
            # nn.ReLU(),
            nn.Dropout(p=self.dropout_prob)
        )
        self.lin3 = nn.Linear(self.output_units, num_classes)
        self.apply(self.init_weights)

    # applied weights recursivelly across model layers
    def init_weights(self, l):
        mean = 0
        if self.model_size == 'small':
            std = 0.05
        elif self.model_size == 'large':
            std = 0.02

        if type(l) == nn.Linear or type(l) == nn.Conv1d:
            torch.nn.init.normal_(l.weight, mean=mean, std=std)

    # dimension computation for layer 7
    def compute_dim(self):
        return ((self.l0 - 96) // 27) * self.conv_feature

    def forward(self, x):
        # batch shape for model input x should be (batch_size, 70, 1014)
        # x = torch.randn(self.batch_size, len(config['data_processing']['alphabet']), self.l0) # for debug
        xt = x.transpose(1, 2)
        # print('input shape  ' + str(x.shape) + ' after traspose ' + str(xt.shape))

        out = self.conv1seq(xt)
        # self.conv1_activation = self.conv1.weight
        # print('shape after CONV1 ' + str(out.shape))
        out = self.conv2seq(out)
        # print('shape after CONV2 ' + str(out.shape))
        out = self.conv3seq(out)
        # print('shape after CONV3 ' + str(out.shape))
        out = self.conv4seq(out)
        # print('shape after CONV4 ' + str(out.shape))
        out = self.conv5seq(out)
        # print('shape after CONV5 ' + str(out.shape))
        out = self.conv6seq(out)
        # print('shape after CONV6 ' + str(out.shape))

        out = out.view(out.shape[0], out.shape[1] * out.shape[2])
        # print('shape to dense ' + str(out.shape))
        out = self.lin1seq(self.dropout(out))
        out = self.lin2seq(self.dropout(out))
        out = self.lin3(out)

        return out

    # def visualize_conv1(self):
    #     act = self.conv1_activation.squeeze().detach().cpu()
    #     for idx in range(act.size(0)):
    #         fig, axarr = plt.subplots()
    #         axarr.imshow(act.numpy()[idx])
    #         plt.show()
