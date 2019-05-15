import json
import sys

import numpy as np
from numpy.distutils.system_info import numarray_info
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
from torchtext import data as torchdata
from tqdm import tqdm

from FeatureField import FeatureField
from FeatureModel import FeatureModel

sys.path.append("..")
from utils import utils


def train(model, criterion, optimiser, train_iterator, device):
    model.train()

    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        pbar = tqdm(total=total_batches)
        epoch_correct = 0
        train_loss = 0
        epoch_predictions = 0
        for i, batch in enumerate(train_iterator):
            predictions = model(batch.features)  # forward pass

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

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pbar.update(1)

        if (epoch + 1)  % 5 == 0:
            utils.save_model(f'model_{epoch}.ckp', model)


def test(model, test_iterator):
    global num_classes
    model.eval()
    print('Testing model ...')

    total_correct = 0
    true_labels = []
    model_predictions = []
    true_predictions = []
    for i, batch in enumerate(test_iterator):
        predictions = model(batch.features)  # forward pass
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
        f'Total correct {total_correct} out of {len(model_predictions)}' +
        f'Correct by classes: {[true_predictions.count(c) for c in list(range(num_classes))]} /' +
        f'{[true_labels.count(c) for c in list(range(num_classes))]}\n'
    )


def weights_analysis(featureModel):
    for idx in range(featureModel.pred.weight.size(0)):
        plt.rcParams["figure.figsize"] = 5, 2
        y = featureModel.pred.weight.detach().numpy()[idx][:8]
        fig, ax = plt.subplots()

        # extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
        ax.imshow(y[np.newaxis, :], cmap="Blues", aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(config['features'][:8])))
        ax.set_xticklabels(config['features'][:8])
        print(ax.get_xticklabels())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FEATURES = FeatureField(features=config['features'], ngrams_range=config['ngrams_range'], device=device)
    LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)
    train_dataset, test_dataset = torchdata.TabularDataset.splits(path=config['dataset_path'],
                                                                  train=config['dataset_train'],
                                                                  test=config['dataset_test'],
                                                                  format='tsv',
                                                                  fields=[('label', LABEL), ('features', FEATURES)])
    train_iterator = torchdata.BucketIterator(train_dataset, batch_size=config['batch_size'],
                                              device=device,
                                              sort_within_batch=False)
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['batch_size'],
                                             device=device,
                                             sort_within_batch=False)
    FEATURES.get_ngram_features([example.features for example in train_dataset.examples])
    LABEL.build_vocab(train_dataset)
    FEATURES.build_vocab(train_dataset)
    num_classes, weights = utils.get_weights([e.label for e in train_dataset.examples], config)

    featureModel = FeatureModel(FEATURES.get_features_count(), num_classes, config['dropout']).to(device)
    # featureModel = FeatureModel(153099, 3, config['dropout']).to(device)
    if config['load_checkpoint']:
        featureModel.load_state_dict(torch.load(config['checkpoint'], map_location=device))

    print(f'Model has {utils.count_parameters(featureModel)} trainable parameters')

    # weights vectors analysis
    # weights_analysis(featureModel)

    optimiser = torch.optim.Adam(featureModel.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())

    train(featureModel, criterion, optimiser, train_iterator, device)

    if config['save_model']:
        utils.save_model('FeatureModel.ckpt', featureModel)

    test(featureModel, test_iterator)