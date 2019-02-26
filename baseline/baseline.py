import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from baselineModel import BaselineModel
from davidson_data import classes_summary, get_y_values


random_seed = 42
reduce = {'do_reducing': True, 'reduce_to': 10}   # for debugging
save_model = True
model_testing = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_size = 0.2
batch_size = 3
learning_rate = 0.003
classes = 3
weighted_classes = True
if weighted_classes:
    class_weights = torch.Tensor(classes_summary(reduce['do_reducing'], reduce['reduce_to'])).float()
    class_weights = class_weights.to(device)
else:
    class_weights = torch.as_tensor([1, 1, 1], device=device).float()

# data loading
x_vals = []
with open('data\\features_1gram.csv') as f:
    for (idx, line) in enumerate(f, 1):
        if (idx % 1000) == 0:
            print('On line ' + str(idx))
        item = [float(x) for x in line.split(',')]
        x_vals.append(item)
        if reduce['do_reducing'] and idx == reduce['reduce_to']:
            break
    print('Dataset loaded.')

y_vals = get_y_values(reduce['do_reducing'], reduce['reduce_to'])
x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=test_size, random_state=random_seed)
features_count = len(x_train[0])

log_reg_model = BaselineModel(features_count, classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimiser = torch.optim.SGD(linear_reg_model.parameters(), lr = learning_rate)
optimiser = torch.optim.Adam(log_reg_model.parameters(), lr=learning_rate)


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


# model training
epochs = 10
avg_loss = 0
batches_done = 0
y_true = []
y_predictions = []

log_reg_model.train()
for e in range(epochs):
    pbar = tqdm(total=len(range(len(x_train) // batch_size)))
    # TODO add different batches because now are always same
    for step, epoch in enumerate(range(len(x_train) // batch_size)):
        # batch_ids = np.random.choice(ids, batch_size)
        batch_data = x_train[batch_size * step:(step + 1) * batch_size]
        batch_labels = y_train[batch_size * step:(step + 1) * batch_size]
        inputs = torch.Tensor(batch_data).to(device)
        target = torch.Tensor(set_index_target(batch_labels)).long().to(device)

        optimiser.zero_grad()
        outputs = log_reg_model.forward(inputs.float())

        loss = criterion(outputs, target)
        batches_done += 1
        avg_loss += loss.item()

        pbar.set_description(f"Loss data {avg_loss / batches_done}")
        pbar.update(1)

        y_test_batch = [np.argmax(t) for t in batch_labels]
        y_predict_batch = [np.argmax(t) for t in outputs.cpu().detach().numpy()]
        # print(f'F1Score for batch: {f1_score(y_test_batch, y_predict_batch, average="macro")}')

        y_true = y_true + y_test_batch
        y_predictions = y_predictions + y_predict_batch
        print(f'F1Score so far: {f1_score(y_true, y_predictions, average="macro")}')

        loss.backward()
        optimiser.step()  # update


# model testing
if model_testing:
    log_reg_model.eval()
    test_true = [np.argmax(t) for t in y_test]
    test_pred = []
    for x, label in zip(x_test, y_test):
        input = torch.Tensor([x]).to(device)
        out = log_reg_model(input.float())
        test_pred.append(np.argmax(outputs.cpu().detach().numpy()))
    print(f'Model has F1Score: {f1_score(y_true, y_predictions, average="macro")}')

# save model
if save_model:
    torch.save(log_reg_model.state_dict(), 'baselineModelTrained.pt')
