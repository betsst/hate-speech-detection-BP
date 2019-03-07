import json

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim
from torchtext import data as torchdata
from tqdm import tqdm

from characterCNN.CharCNNModel import CharCNNModel
from characterCNN.CharField import CharField
from characterCNN.ChatCnnDataset import CharCnnDataset


def train(model, criterion, optimiser, train_iterator):
    model.train()

    train_loss = 0
    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    pbar = tqdm(total=total_batches)
    model_predictions = []
    true_labels = []

    for epoch in range(config_train['num_epochs']):
        for i, batch in enumerate(train_iterator):
            predictions = model(batch.text)  # forward pass

            loss = criterion(predictions, batch.labels)
            train_loss += loss.item()

            label_pred = [np.argmax(p) for p in predictions.detach().numpy()]
            true_labels = true_labels + batch.labels.detach().tolist()
            model_predictions = model_predictions + label_pred
            for p, tp in zip(label_pred, batch.labels.detach().tolist()):
                if p == tp:
                    total_correct += 1

            pbar.set_description(
                f'Loss: {train_loss / (i + 1)}, Acc: {total_correct / (len(batch) * (i + 1))},' +
                f'F1: {f1_score(true_labels, model_predictions, average="macro")}')

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            # TODO learning scheduler https://pytorch.org/docs/0.3.0/optim.html#how-to-adjust-learning-rate or determing step size
            optimiser.step()
            pbar.update(1)


def eval(model, eval_iterator):
    model.eval()


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
        config_data = config['data_processing']
        config_model = config['model']
        config_train = config['model']['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT = torchdata.Field(tokenize=lambda x: list(x), fix_length=config_model['l0'],
                           lower=not config_data['differ_uppercase'])
    CHARS = CharField(fix_length=config_model['l0'], lower=not config_data['differ_uppercase'])
    LABELS = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

    charcnn_dataset = CharCnnDataset(path=config_data['dataset_file'], format='tsv',
                                     fields=[('labels', LABELS), ('text', CHARS)])

    train_iterator = torchdata.Iterator(charcnn_dataset, batch_size=config_model['batch_size'], device=device)

    num_classes, weights = charcnn_dataset.get_class_weight()
    if config_model['balanced_weights']:
        weights = [1.0] * num_classes

    TEXT.build_vocab(config_data['alphabet'])
    LABELS.build_vocab(charcnn_dataset)

    charCNNModel = CharCNNModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())
    optimiser = torch.optim.SGD(charCNNModel.parameters(), lr=config_model['train']['learning_rate'], momentum=0.9)
    # optimiser = torch.optim.Adam(charCNNModel.parameters(), lr=config['learning_rate'])

    train(charCNNModel, criterion, optimiser, train_iterator)
    eval(charCNNModel, [])

    # save model
    torch.save(charCNNModel.state_dict(), 'modelCharCNN.ckpt')
