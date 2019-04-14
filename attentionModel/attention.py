import json
import matplotlib
import matplotlib.pyplot as plt
import sys

from allennlp.data import Vocabulary, Tokenizer, DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.modules import TextFieldEmbedder
from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchtext import data as torchdata
from torchtext.vocab import GloVe
from tqdm import tqdm

from SelfAttentionModel import SelfAttentionModel

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


def ids2text(batch, vocab):
    batch_text = []
    for doc in batch:
        doc_text = [vocab.itos[id] for id in doc]
        batch_text.append(doc_text)
    return batch_text


# visualize all extractions attentions for one document
def visualize_doc_all_extractions(doc_attention, doc_text, ):
    fig, ax = plt.subplots()
    im = ax.imshow(doc_attention.cpu().detach().numpy(), cmap='Blues')

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
    sum_att = torch.sum(attention, dim=1)  # sum over annotations
    norm_att = F.softmax(sum_att, dim=1)  # normalize to sum up to one
    for doc_att, doc_text in zip(norm_att, docs_text):
        fig, ax = plt.subplots()
        im = ax.imshow([doc_att.cpu().detach().numpy()], cmap='Blues')
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
    if config['visualize_joined_extractions']:
        visualize_joined_att(attention, docs_text)


def frobenius_norm(p):
    pen_loss = (torch.sum(torch.sum((p ** 2), 1), 1).squeeze()) ** 0.5
    return torch.sum(pen_loss) / p.shape[0]


def train(model, criterion, optimiser, train_iterator, vocab, device):
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
            predictions, attention = model(batch.text)  # forward pass
            # visualize(batch.text, attention, vocab)

            loss = criterion(predictions, batch.label)

            # penalization term
            if config['penalization_form']:
                # A * A^T - identity matrix
                size = attention.shape
                pen = torch.bmm(attention, attention.transpose(2, 1))
                pen -= torch.eye(config['extraction_count'], requires_grad=False, device=device)\
                    .expand(size[0], config['extraction_count'], config['extraction_count'])
                pen_loss = frobenius_norm(pen)  # frob_norm(penalization)
                loss += pen_loss * config['coefficient']

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
    true_labels = []
    model_predictions = []
    # pbar = tqdm(total=total_batches)
    for i, batch in enumerate(test_iterator):
        predictions, attention = model(batch.text)  # forward pass
        if config['visualization']:
            visualize(batch.text, attention, vocab)
        label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
        true_labels = true_labels + batch.label.cpu().detach().tolist()
        model_predictions = model_predictions + label_pred
        for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
            if p == tp:
                total_correct += 1
        # pbar.update(1)

    print(
        f'\n\n\nAcc: {total_correct / len(model_predictions):.7f} ' +
        f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
        f'Total correct {total_correct} out of {len(model_predictions)}\n'
    )


def save_model(model):
    torch.save(model.state_dict(), 'modelSelfAtt.ckpt')


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
sentence_tokenizer = PunktSentenceTokenizer()


def bert_tokenize(text):
    sentences = sentence_tokenizer.tokenize(text)
    text = ''
    for sentence in sentences:
        text += sentence + ' [SEP] '
    tokens = []
    tokens.append("[CLS]")
    tokens += bert_tokenizer.tokenize(text)
    return tokens


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if config['embeddings'] == 'elmo':
    #     token_indexer = ELMoTokenCharactersIndexer()
    #     ds_reader = DatasetReader()
    #     train_ds, test_ds = (ds_reader.read(dataset) for dataset in [config['dataset_train'], config['dataset_test']])
    #     vocab = Vocabulary()
    #     train_iterator = BucketIterator(train_ds, batch_size=config['batch_size'])
    #     test_iterator = BucketIterator(test_ds, batch_size=config['batch_size'])

    # else:
    text_tokenizer = bert_tokenize if config["embeddings"] == "bert" else "spacy"
    TEXT = torchdata.Field(tokenize=text_tokenizer, sequential=True, lower=True, batch_first=True)

    LABEL = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)

    train_dataset, test_dataset = torchdata.TabularDataset.splits(path=config['dataset_path'],
                                                                  train=config['dataset_train'],
                                                                  test=config['dataset_test'],
                                                                  format='tsv',
                                                                  fields=[('label', LABEL), ('text', TEXT)])

    train_iterator = torchdata.BucketIterator(train_dataset, batch_size=config['batch_size'],
                                              sort_key=lambda x: len(x.text),
                                              device=device,
                                              sort_within_batch=False)
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['batch_size'],
                                             sort_key=lambda x: len(x.text),
                                             device=device,
                                             sort_within_batch=False)

    # vocab = Vocabulary.from_instances(train_dataset + test_dataset)

    LABEL.build_vocab(train_dataset)
    if config['embeddings'] == 'glove':
        TEXT.build_vocab(train_dataset, vectors=GloVe(name='6B', dim=300))
    elif config['embeddings'] == 'bert' or config['embeddings'] == 'elmo':
        TEXT.build_vocab(train_dataset)
    else:
        raise NotImplementedError('Embeddings should be either "glove", "bert", "elmo".')

    num_classes, weights = get_weights([e.label for e in train_dataset.examples])

    selfAttModel = SelfAttentionModel(vocabulary=TEXT.vocab, device=device, num_classes=num_classes).to(device)
    if config['load_checkpoint']:
        selfAttModel.load_state_dict(torch.load(config['checkpoint'], map_location=device))

    print(f'Model has {utils.count_parameters(selfAttModel)} trainable parameters')

    optimiser = torch.optim.Adam(selfAttModel.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())

    train(selfAttModel, criterion, optimiser, train_iterator, TEXT.vocab, device)

    if config['save_model']:
        save_model(selfAttModel)

    test(selfAttModel, test_iterator, TEXT.vocab)
