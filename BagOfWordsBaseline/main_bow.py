# Model based on bag-of-words approach with its TF-IDF values
import csv
import json
import sys

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim
from torchtext import data as torchdata
from torchtext.data import TabularDataset
from tqdm import tqdm

from BOWField import BOWField
from BOWModel import BOWModel
from get_tf import get_tfs
sys.path.append("..")
from utils import utils


def get_weights(labels):
    dataset_size = len(labels)
    classes = set(labels)
    num_classes = len(classes)

    if config['balanced_weights']:
        return num_classes, [[1.0] * num_classes]

    classes_occurrences = [0] * num_classes
    for label in labels:
        classes_occurrences[label] += 1

    max_count = max(classes_occurrences)
    class_weights = [max_count / n for n in classes_occurrences]

    return num_classes, class_weights


def train(model, criterion, optimiser, train_iterator):
    model.train()

    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        pbar = tqdm(total=total_batches)
        train_loss = 0
        epoch_predictions = 0
        epoch_correct = 0
        for i, batch in enumerate(train_iterator):
            predictions = model(batch.text)  # forward pass

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
                f'{epoch + 1}/{config["num_epochs"]} ' +
                f'Loss: {train_loss / (i + 1):.7f} ' +
                f'Acc: {epoch_correct / epoch_predictions:.7f} ' +
                f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
                f'Total correct {total_correct} out of {len(model_predictions)}'
            )
            pbar.update(1)

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


def test(model, test_iterator):
    global num_classes
    model.eval()
    print('Testing model ...')

    total_correct = 0
    true_labels = []
    model_predictions = []
    true_predictions = []

    for i, batch in enumerate(test_iterator):
        predictions = model(batch.text)  # forward pass

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
        f'F1 by classes: {" ".join(str(f1) for f1 in f1_score(true_labels, model_predictions, average=None).tolist())} ' +
        f'Total correct {total_correct} out of {len(model_predictions)}' +
        f'Correct by classes: {[true_predictions.count(c) for c in list(range(num_classes))]} /' +
        f'{[true_labels.count(c) for c in list(range(num_classes))]}\n'
    )


def save_model(model):
    torch.save(model.state_dict(),)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    tfs = None
    if config['get_tfs']:
        tfs = get_tfs(config['save_tfs'], config['doc_freq_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT = BOWField(device, config['cut_most_freq'], term_freqs=tfs)
    LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

    train_dataset, test_dataset = torchdata.TabularDataset.splits(path=config['dataset_path'],
                                                                  train=config['dataset_train'],
                                                                  test=config['dataset_test'],
                                                                  format='tsv',
                                                                  fields=[('label', LABEL), ('text', TEXT)])
    train_iterator = torchdata.BucketIterator(train_dataset, batch_size=config['batch_size'], device=device)
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['batch_size'], device=device)

    TEXT.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    num_classes, weights = get_weights([e.label for e in train_dataset.examples])
    feature_size = TEXT.get_features_count()

    BOWModel = BOWModel(input_size=feature_size, num_classes=num_classes, dropout=config['dropout']).to(device)
    if config['load_checkpoint']:
        BOWModel.load_state_dict(torch.load(config['checkpoint'], map_location=device))
    print(f'Model has {utils.count_parameters(BOWModel)} trainable parameters')

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())
    optimiser = torch.optim.Adam(BOWModel.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    train(BOWModel, criterion, optimiser, train_iterator)
    test(BOWModel, test_iterator)

    if config['save_model']:
        utils.save_model('modelBOW.ckpt', BOWModel)
