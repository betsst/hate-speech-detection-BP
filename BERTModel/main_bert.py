import json
import sys

from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torchtext import data as torchdata
from tqdm import tqdm

from BERTModel import BERTModel

sys.path.append("..")
from utils import utils
from utils.utils import get_weights, save_model


# fine-tuning model
def train(model, criterion, optimiser, train_iterator, device):
    model.train()

    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        train_loss = 0
        pbar = tqdm(total=total_batches)
        batch_correct = 0
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
                    batch_correct += 1

            pbar.set_description(
                f'Loss: {train_loss / ((i + 1) * (epoch + 1)):.7f} ' +
                f'Acc: {batch_correct / len(batch):.7f} ' +
                f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
                f'Total correct {total_correct} out of {len(model_predictions)}\n'
            )

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pbar.update(1)

            # if epoch + 1 == config['freeze_after']:
            #     model.freeze_bert_encoder()

        # test(bert_model, test_iterator)


def test(model, test_iterator):
    model.eval()
    print('Testing model ...')

    total_correct = 0
    total_batches = len(test_iterator.data()) // test_iterator.batch_size
    true_labels = []
    model_predictions = []

    for i, batch in enumerate(test_iterator):
        predictions = model(batch.text)  # forward pass
        label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
        true_labels = true_labels + batch.label.cpu().detach().tolist()
        model_predictions = model_predictions + label_pred
        for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
            if p == tp:
                total_correct += 1

    print(
        f'\n\n\nAcc: {total_correct / (len(batch) * (i + 1)):.7f} ' +
        f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
        f'Total correct {total_correct} out of {len(model_predictions)}\n'
    )


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
sentence_tokenizer = PunktSentenceTokenizer()


def tokenize(text):
    sentences = sentence_tokenizer.tokenize(text)
    text = ''
    for sentence in sentences:
        text += sentence + ' [SEP] '
    tokens = []
    tokens.append("[CLS]")
    tokens += tokenizer.tokenize(text)
    return tokens


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=config['lower_case'])
    TEXT = torchdata.Field(tokenize=tokenize, sequential=True, lower=True, batch_first=True)
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

    TEXT.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    num_classes, weights = get_weights([e.label for e in train_dataset.examples], config)
    bert_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                             num_attention_heads=12, intermediate_size=3072)
    bert_model = BERTModel(bert_config, num_classes).to(device)
    print(f'Model has {utils.count_parameters(bert_model)} trainable parameters')
    if config['load_model']:
        bert_model.load_state_dict(torch.load(config['checkpoint']))

    if config['optimiser'] == 'adam':
        optimiser = torch.optim.Adam(bert_model.parameters(), lr=config['learning_rate'])
    elif config['optimiser'] == 'bert_adam':
        optimiser = BertAdam(bert_model.parameters(), lr=config['learning_rate'])
    else:
        raise NotImplementedError('Optimiser should be set as "adam" or "bert_adam".')

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())

    train(bert_model, criterion, optimiser, train_iterator, device)
    test(bert_model, test_iterator)

    if config['save_model']:
        save_model('modelBert.ckpt', bert_model)
