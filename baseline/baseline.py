import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from thinc.neural._classes import softmax
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchtext import data
import pandas as pd

from davidson_data import get_y_values, classes_summmary

batch_size = 20
classes = 3
class_weights = torch.Tensor(classes_summmary()).float()

with open("..\\..\\data\\x_vals.csv", "r") as ins:
    x_vals = []
    ids = []
    for i, line in enumerate(ins, 0):
        x_vals.append([float(x) for x in line.split(',')])
        ids.append(i)

max_length = len(max(x_vals, key=len))

# df = pd.read_table('..\\..\\data\\x_vals.csv', sep=',', header=None, error_bad_lines=False)
# df.fillna(0, inplace=True)
# print(df)
# my_data = genfromtxt('..\\..\\data\\x_vals.csv', delimiter=',', filling_values=0)
# df = dask.dataframe.read_csv("..\\..\\data\\x_vals.csv", encoding = "UTF-8")
# x_vals = []
# with open('..\\..\\data\\x_vals.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         x_vals.append(row)
#         print(row)


# x_vals = torch.Tensor(df.values)
# X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.33, random_state=42)
# # x_train_np = np.asarray(X_train,dtype=np.float32)
# # x_train_np = np.asarray(X_train,dtype=np.float32).reshape(-1,2)
# x_train = torch.Tensor(x_vals)

y_vals = get_y_values()
y_train_np = np.asarray(y_vals, dtype=np.float32).reshape(-1,3)
y_train = torch.from_numpy(y_train_np)

# alpha = np.random.rand(1)
# beta = np.random.rand(1)

class BaselineModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineModel, self).__init__()
        # simple linear layer
        self.linear = torch.nn.Linear(input_size, out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

    def num_features(self):
        return 1180

    def forward(self, x):
        # normalization
        x_mean = torch.mean(x, dim=1).view(-1, 1)
        x_std = torch.std(x, dim=1).view(-1, 1)
        x_normalized = (x - x_mean) / x_std
        output = self.linear(x_normalized)
        output = self.softmax(output)
        return output

# linear_reg_model = BaselineModel(len(x_vals[0]), 1)
# TODO  what value should be set if it varies
linear_reg_model = BaselineModel(max_length, classes)

criterion = nn.CrossEntropyLoss(weight=class_weights)
learning_rate = 0.1
# optimiser = torch.optim.SGD(linear_reg_model.parameters(), lr = learning_rate)
optimiser = torch.optim.Adam(linear_reg_model.parameters(), lr = learning_rate)

def set_index_target(hot_one_encoding):
    target_list = []
    for l in hot_one_encoding:
        if l[0] == 1:
            target_list.append(0)
        elif l[1] == 1:
            target_list.append(1)
        else:  # l[2] == 1:
            target_list.append(2)
        # OR
        # target_list.append(np.nonzero(l)[0])
    return target_list

def pad(seq, target_length, padding=0):
    for i, s in enumerate(seq, 0):
        if len(s) < target_length:
            seq[i].extend([padding] * (target_length - len(s)))
    return seq

# training
epochs = 500
linear_reg_model.train()
for epoch in range(epochs):
    batch_ids = np.random.choice(ids, batch_size)
    batch_data = [x_vals[i] for i in batch_ids]
    batch_labels = [y_vals[i] for i in batch_ids]
    seq_lengths = [len(b) for b in batch_data]
    batch_data = pad(batch_data, max(seq_lengths))
    batch_data = torch.Tensor(batch_data)

    inputs = Variable(batch_data)
    labels = Variable(y_train)
    target = torch.Tensor(set_index_target(batch_labels)).long()

    optimiser.zero_grad()
    outputs = linear_reg_model.forward(inputs.float())
    for o in outputs:
        print(o)
    loss = criterion(outputs, target)
    # loss = criterion(outputs.long(), labels.long())
    print("Loss data ", loss.item())
    loss.backward()
    optimiser.step() # update


# predicted = linear_reg_model(Variable(x_train)).data.numpy()
# plt.plot(x_train_np, y_train_np, 'ro', label='Original Data')
# plt.plot(x_train_np, predicted, label='Fitted Line')
# plt.legend()
# plt.show()

# testing

# exp = torch.Tensor([25.0])
# print("Length of sentence is: 25\n Class is:", linear_reg_model(exp).data[0][0].item())
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28 * 28))
#     outputs = model(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Accuracy of the model: %d %%' % (100 * correct / total))

