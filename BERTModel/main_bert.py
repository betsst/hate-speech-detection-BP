import json
import sys
import warnings

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


with open('config.json', 'r') as f:
    config = json.load(f)


# fine-tuning model
def train(model, criterion, optimiser, train_iterator, vocab):
    model.train()
    model.freeze_bert_encoder()

    total_correct = 0
    total_batches = len(train_iterator.data()) // train_iterator.batch_size
    model_predictions = []
    true_labels = []

    for epoch in range(config['num_epochs']):
        train_loss = 0
        pbar = tqdm(total=total_batches)
        epoch_predictions = 0
        epoch_correct = 0
        for i, batch in enumerate(train_iterator):
            # logits = model(input_ids, segment_ids, input_mask)
            segment_ids, input_mask = extract_features(batch.text, vocab)
            predictions = model(batch.text, segment_ids, input_mask)  # forward pass
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

            # if epoch + 1 == config['freeze_after']:
            #     model.freeze_bert_encoder()

        # if epoch == config['unfreeze_after_epoch']:
        #     model.unfreeze_bert_encoder()


def test(model, test_iterator, vocab):
    global num_classes
    model.eval()
    print('Testing model ...')

    total_correct = 0
    total_batches = len(test_iterator.data()) // test_iterator.batch_size
    true_labels = []
    model_predictions = []
    true_predictions = []

    for i, batch in enumerate(test_iterator):
        segment_ids, input_mask = extract_features(batch.text, vocab)
        predictions = model(batch.text, segment_ids, input_mask)  # forward pass
        label_pred = [np.argmax(p) for p in predictions.cpu().detach().numpy()]
        true_labels = true_labels + batch.label.cpu().detach().tolist()
        model_predictions = model_predictions + label_pred
        for p, tp in zip(label_pred, batch.label.cpu().detach().tolist()):
            if p == tp:
                total_correct += 1
                true_predictions.append(p)

    print(
        f'\n\n\nAcc: {total_correct / (len(batch) * (i + 1)):.7f} ' +
        f'F1: {f1_score(true_labels, model_predictions, average="macro"):.7f} ' +
        f'Total correct {total_correct} out of {len(model_predictions)}' +
        f'Correct by classes: {[true_predictions.count(c) for c in list(range(num_classes))]} /' +
        f'{[true_labels.count(c) for c in list(range(num_classes))]}\n'
    )


tokenizer = BertTokenizer.from_pretrained(config['bert_model'], do_lower_case=True, max_len=config['max_seq_length'])
sentence_tokenizer = PunktSentenceTokenizer()


def tokenize(text):
    if not config['one_seq']:
        sentences = sentence_tokenizer.tokenize(text)
        text = ''
        for sentence in sentences:
            text += sentence + ' [SEP] '
    else:
        text += ' [SEP] '

    tokens = []
    tokens.append("[CLS]")
    tokens += tokenizer.tokenize(text)
    return tokens


def extract_features(batch, vocab):
    batch_segment_ids = []
    batch_input_mask = []
    for example in batch:
        # example == input_ids
        segment_ids = [0] * len(example)
        batch_segment_ids.append(segment_ids)
        input_mask = [1] * len(example)
        batch_input_mask.append(input_mask)
        # padding = [0] * (config['max_seq_length'] - len(example))
        # input_mask += padding
        # segment_ids += padding
    return torch.tensor(batch_segment_ids).to(device), torch.tensor(batch_input_mask).to(device)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=config['lower_case'])
    if not config['lower_case'] and 'uncased' in config['bert_model']:
        warnings.warn('Using uncased bert model should be lower casting characters.')
        config['lower_case'] = True

    TEXT = torchdata.Field(tokenize=tokenize, sequential=True, lower=config['lower_case'], batch_first=True,
                           fix_length=config['max_seq_length'])
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
    test_iterator = torchdata.BucketIterator(test_dataset, batch_size=config['test_batch_size'],
                                             sort_key=lambda x: len(x.text),
                                             device=device,
                                             sort_within_batch=False)

    TEXT.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    num_classes, weights = get_weights([e.label for e in train_dataset.examples], config)
    bert_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                             num_attention_heads=12, intermediate_size=3072)
    bert_model = BERTModel(bert_config, num_classes, config['bert_model']).to(device)
    print(f'Model has {utils.count_parameters(bert_model)} trainable parameters')
    if config['load_model']:
        bert_model.load_state_dict(torch.load(config['checkpoint']))

    if config['optimiser'] == 'adam':
        optimiser = torch.optim.Adam(bert_model.parameters(), lr=config['learning_rate'])
    elif config['optimiser'] == 'bert_adam':
        optimiser = BertAdam(bert_model.parameters(), lr=config['learning_rate'], warmup=0.1,
                             t_total=int(len(train_dataset) / config['batch_size'] / 1) * config['num_epochs'])
    else:
        raise NotImplementedError('Optimiser should be set as either "adam" or "bert_adam".')

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device).float())

    if not config['do_train'] and not config['do_test']:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if config['do_train']:
        train(bert_model, criterion, optimiser, train_iterator, TEXT.vocab)
    if config['do_test']:
        test(bert_model, test_iterator, TEXT.vocab)

    if config['save_model']:
        save_model('modelBert.ckpt', bert_model)
