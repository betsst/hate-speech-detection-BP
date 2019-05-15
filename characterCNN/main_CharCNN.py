import json
import sys

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim
from torchtext import data as torchdata
from tqdm import tqdm

from CharCNNModel import CharCNNModel
from CharField import CharField

sys.path.append("..")
from utils import utils

with open('config.json', 'r') as f:
    config = json.load(f)
halved_times = 0
prev_lr = config['learning_rate']


def train(model, criterion, optimiser, train_iterator):
    model.train()

    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        pbar = tqdm(total=total_batches)
        train_loss = 0
        epoch_correct = 0
        train_loss = 0
        epoch_predictions = 0
        for i, batch in enumerate(train_iterator):
            predictions = model(batch.chars)  # forward pass

            loss = criterion(predictions, batch.label)
            train_loss += loss.item()

            label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
            true_labels = true_labels + batch.label.cpu().detach().tolist()
            model_predictions = model_predictions + label_pred
            for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
                epoch_predictions += 1
                if p == tp:
                    total_correct += 1
                    epoch_correct += 1

            pbar.set_description(
                f'{str(optimiser.param_groups[0]["lr"])} - ' +
                f'{epoch + 1}/{config["num_epochs"]} ' +
                f'Loss: {train_loss / (i + 1):.7f} ' +
                f'Acc: {epoch_correct / epoch_predictions:.7f} ' +
                f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
                f'Total correct {total_correct} out of {len(model_predictions)}'
            )

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            pbar.update(1)
        # print(f'{optimiser.param_groups["lr"]}')
        # optimiser = adjust_learning_rate(optimiser, epoch)

        if (epoch + 1) % 10 == 0:
            utils.save_model(f'modelCharCNN_large_lr_{epoch}.ckpt', charCNNModel)


def test(model, test_iterator):
    global num_classes
    model.eval()
    print('Testing model ...')

    total_correct = 0
    true_labels = []
    model_predictions = []
    true_predictions = []

    for i, batch in enumerate(test_iterator):
        predictions = model(batch.chars)  # forward pass
        # model.visualize_conv1()

        label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
        true_labels = true_labels + batch.label.cpu().detach().tolist()
        model_predictions = model_predictions + label_pred

        for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
            if p == tp:
                total_correct += 1
                true_predictions.append(p)

    print(
        f'\n\n\nAcc: {total_correct / len(model_predictions):.7f} ' +
        f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
        f'F1 by classes: {" ".join(str(f1)for f1 in f1_score(true_labels, model_predictions, average=None).tolist())}' +
        f'Total correct {total_correct} out of {len(model_predictions)}\n'
        f'Correct by classes: {[true_predictions.count(c) for c in list(range(num_classes))]} /' +
        f'{[true_labels.count(c) for c in list(range(num_classes))]}\n'
    )


def adjust_learning_rate(optimiser, epoch):
    """Sets the learning rate to the initial LR halved every 3 epochs"""
    global prev_lr
    global config
    global halved_times

    lr = config['learning_rate'] * (0.9 ** ((epoch + 1) // 3))
    if prev_lr != lr:
        if halved_times < 10:
            halved_times += 1
            prev_lr = lr
        else:
            lr = prev_lr
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
    return optimiser


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHARS = CharField(fix_length=config['l0'], lower=not config['differ_uppercase'])
    LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

    train_dataset, test_dataset = torchdata.TabularDataset.splits(path=config['dataset_path'],
                                                                  train=config['dataset_train'],
                                                                  test=config['dataset_test'],
                                                                  format='tsv',
                                                                  fields=[('label', LABEL), ('chars', CHARS)])

    train_iterator = torchdata.BucketIterator(train_dataset, batch_size=config['batch_size'], device=device)
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['test_batch_size'], device=device)

    num_classes, weights = utils.get_weights([e.label for e in train_dataset.examples], config)

    alphabet = config['alphabet']
    # alphabet.append("'")
    CHARS.build_vocab(alphabet)
    LABEL.build_vocab(train_dataset)

    charCNNModel = CharCNNModel(num_classes, alphabet=alphabet).to(device)
    if config['load_model']:
        charCNNModel.load_state_dict(torch.load(config['model_path'], map_location=device))
    print(f'Model has {utils.count_parameters(charCNNModel)} trainable parameters')

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())
    # optimiser = torch.optim.SGD(charCNNModel.parameters(), lr=config['learning_rate'], momentum=0.9,
    #                             weight_decay=config['weight_decay'])
    optimiser = torch.optim.Adam(charCNNModel.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    if not config['do_train'] and not config['do_test']:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if config['do_train']:
        train(charCNNModel, criterion, optimiser, train_iterator)
    if config['do_test']:
        test(charCNNModel, test_iterator)

    # save model
    if config['save_model']:
        utils.save_model('modelCharCNN.ckpt', charCNNModel)
