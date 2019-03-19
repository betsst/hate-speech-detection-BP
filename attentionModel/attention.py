import json
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data as torchdata
from torchtext.vocab import GloVe
from tqdm import tqdm

from SelfAttentionModel import SelfAttentionModel


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


def ids2text(batch, vocab):
    batch_text = []
    for doc in batch:
        doc_text = [vocab.itos[id] for id in doc]
        batch_text.append(doc_text)
    return batch_text


# visualize all extractions attentions for one document
def visualize_doc_all_extractions(doc_attention, doc_text, ):
    fig, ax = plt.subplots()
    im = ax.imshow(doc_attention.cpu().detach().numpy(), cmap='YlGn')

    ax.set_xticks(np.arange(len(doc_text)))
    ax.set_yticks(np.arange(doc_attention.shape[0]))
    ax.set_xticklabels(doc_text)
    ax.set_yticklabels([idx for idx in range(doc_attention.shape[0])])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    plt.show()


# visualize all joined attention extractions for batch documents
def visualize_joined_att(attention, docs_text):
    # join extraction
    sum_att = torch.sum(attention, dim=1)  # sum over annotations
    norm_att = F.softmax(sum_att, dim=1)  # normalize to sum up to one
    for doc_att, doc_text in zip(norm_att, docs_text):
        fig, ax = plt.subplots()
        im = ax.imshow([doc_att.cpu().detach().numpy()])
        ax.set_xticks(np.arange(len(doc_text)))
        ax.set_xticklabels(doc_text)

        ax.set_yticks([0])
        ax.set_yticklabels([1])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # fig.tight_layout()
        plt.show()


def visualize(batch, attention, vocab):
    docs_text = ids2text(batch, vocab)

    if config['visualize_all_extractions']:
        for doc_attention, doc_text in zip(attention, docs_text):
            visualize_doc_all_extractions(doc_attention, doc_text)
    else:
        visualize_joined_att(attention, docs_text)


def frobenius_norm(p):
    pen_loss = torch.sum(p ** 2) ** 0.5  # sum(p^2)^0.5
    return pen_loss


def train(model, criterion, optimiser, train_iterator, vocab, device):
    model.train()

    train_loss = 0
    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        pbar = tqdm(total=total_batches)
        for i, batch in enumerate(train_iterator):
            predictions, attention = model(batch.text)  # forward pass
            # visualize(batch.text, attention, vocab)

            loss = criterion(predictions, batch.label)

            # penalization term
            if config['penalization_form']:
                # A * A^T - identity matrix
                pen = torch.bmm(attention, attention.transpose(2, 1))
                pen -= torch.eye(config['extraction_count'], requires_grad=False, device=device)\
                    .expand(config['batch_size'], config['extraction_count'], config['extraction_count'])
                pen_loss = frobenius_norm(pen)  # frob_norm(penalization)
                loss += pen_loss * config['coefficient']

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

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pbar.update(1)


def test(model, test_iterator, vocab):
    model.eval()
    print('Testing model ...')

    total_correct = 0
    total_batches = len(test_iterator.data()) // test_iterator.batch_size
    pbar = tqdm(total=total_batches)
    for i, batch in enumerate(train_iterator):
        predictions, attention = model(batch.text)  # forward pass
        visualize(batch.text, attention, vocab)
        label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
        true_labels = true_labels + batch.label.cpu().detach().tolist()
        model_predictions = model_predictions + label_pred
        for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
            if p == tp:
                total_correct += 1
        pbar.update(1)

    print(
        f'Acc: {total_correct / (len(batch) * (i + 1)):.7f} ' +
        f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
        f'Total correct {total_correct} out of {len(model_predictions)}\n'
    )


def save_model(model):
    torch.save(model.state_dict(), 'modelSelfAtt.ckpt')


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT = torchdata.Field(tokenize="spacy", sequential=True, lower=True, batch_first=True)
    LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

    tabular_dataset = torchdata.TabularDataset(config['dataset'], format='tsv',
                                               fields=[('label', LABEL), ('text', TEXT)])
    train_dataset, test_dataset = tabular_dataset.split()

    train_iterator = torchdata.BucketIterator(train_dataset, batch_size=config['batch_size'],
                                              sort_key=lambda x: len(x.text),
                                              device=device,
                                              sort_within_batch=False)
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['batch_size'],
                                             sort_key=lambda x: len(x.text),
                                             device=device,
                                             sort_within_batch=False)
    if config['embeddings'] == 'glove':
        TEXT.build_vocab(tabular_dataset, vectors=GloVe(name='6B', dim=300))
    else:
        raise NotImplementedError('Embeddings should be either "glove"')

    LABEL.build_vocab(tabular_dataset)

    num_classes, weights = get_weights([e.label for e in tabular_dataset.examples])

    selfAttModel = SelfAttentionModel(vocabulary=TEXT.vocab, device=device, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())
    optimiser = torch.optim.Adam(selfAttModel.parameters(), lr=config['learning_rate'])

    train(selfAttModel, criterion, optimiser, train_iterator, TEXT.vocab, device)
    test(selfAttModel, test_iterator, TEXT.vocab)

    if config['save_model']:
        save_model(selfAttModel)
