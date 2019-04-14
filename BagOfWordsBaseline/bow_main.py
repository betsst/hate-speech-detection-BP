# Model based on bag-of-words approach with its TF-IDF values
import csv
import json

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim
from torchtext import data as torchdata
from torchtext.data import TabularDataset
from tqdm import tqdm

from BagOfWordsBaseline.BOWField import BOWField
from BagOfWordsBaseline.BOWModel import BOWModel
from get_tf import get_tfs
from utils.utils import count_parameters


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
        for i, batch in enumerate(train_iterator):
            predictions = model(batch.text)  # forward pass

            loss = criterion(predictions, batch.label)
            train_loss += loss.item()

            label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
            true_labels = true_labels + batch.label.cpu().detach().tolist()
            model_predictions = model_predictions + label_pred
            for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
                if p == tp:
                    total_correct += 1

            pbar.set_description(
                f'Loss: {train_loss / ((i + 1) * (epoch + 1)):.7f} ' +
                f'Acc: {total_correct / ((len(batch) * (i + 1)) * (epoch + 1)):.7f} ' +
                f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
                f'Total correct {total_correct} out of {len(model_predictions)}\n'
            )
            pbar.update(1)

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


def test(model, test_iterator):
    model.eval()
    # test_true = [np.argmax(t) for t in y_test]
    # test_pred = []
    # for x, label in zip(x_test, y_test):
    #     input = torch.Tensor([x]).to(device)
    #     out = model(input)
    #     test_pred.append(np.argmax(outputs.cpu().detach().numpy()))
    # print(f'Model has F1Score: {f1_score(test_true, test_pred, average="macro")}')


def save_model(model):
    torch.save(model.state_dict(), 'modelBOW.ckpt')


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    tfs = None
    if config['get_tfs']:
        tfs = get_tfs(config['save_tfs'], config['doc_freq_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT = BOWField(term_freqs=tfs)
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
    print(f'Model has {count_parameters(BOWModel)} trainable parameters')

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())
    optimiser = torch.optim.Adam(BOWModel.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    train(BOWModel, criterion, optimiser, train_iterator)
    test(BOWModel, test_iterator)

    if config['save_model']:
        save_model(BOWModel)
