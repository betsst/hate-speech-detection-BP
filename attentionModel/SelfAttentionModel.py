# model based on Structured Self-attentive Sentence Embedding - https://arxiv.org/pdf/1703.03130.pdf
import json
import sys

from allennlp import training
from allennlp.commands.elmo import ElmoEmbedder, batch_to_ids
from allennlp.modules import TextFieldEmbedder, Elmo
from allennlp.modules.token_embedders import Embedding
from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertEmbeddings
import torch
import torch.nn as nn

sys.path.append("..")
from utils.utils import ids2str


class SelfAttentionModel(nn.Module):
    with open('config.json', 'r') as f:
        config = json.load(f)

    def __init__(self, vocabulary, num_classes, device, u=config['hidden_units'], da=config['d_a'],
                 r=config['extraction_count'], pad_masking=config['pad_masking'], embeddings=config['embeddings'],
                 hidden_classifier_units=config['hidden_classifier_units'],
                 classifier_dropout=config['classifier_dropout']):
        super(SelfAttentionModel, self).__init__()
        self.device = device
        self.u = u  # hidden unit for each unidirectional LSTM
        self.r = r  # number of different parts to be extracted from the sentence.

        self.embeddings = embeddings
        self.embeddings_dim = 300
        self.vocab_size = len(vocabulary.stoi)
        self.vocab = vocabulary
        if self.embeddings == 'bert':
            self.bert_config = BertConfig(vocab_size_or_config_json_file=self.vocab_size, hidden_size=768,
                                          num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
            self.bert = BertModel(self.bert_config).from_pretrained('bert-base-uncased')
            self.embeddings_dim = self.bert.embeddings.word_embeddings.embedding_dim
        elif self.embeddings == 'elmo':
            # self.elmo_embedding = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embeddings_dim)
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo_embedder = ElmoEmbedder(options_file, weight_file)
            self.elmo = Elmo(options_file, weight_file, 1)
            self.embeddings_dim = 1024
        elif self.embeddings == 'glove':
            self.embeddings_dim = vocabulary.vectors.shape[1]
            self.embeddings = nn.Embedding(len(vocabulary), self.embeddings_dim)
            if pad_masking:
                self.vocab.vectors[self.vocab.stoi['<pad>']] = -1e8 * self.embeddings_dim  # pad token masking
            self.embeddings.weight.data.copy_(vocabulary.vectors.to(device))

        self.bilstm = nn.LSTM(input_size=self.embeddings_dim, hidden_size=u, num_layers=1, bidirectional=True)  # batch_first=True
        self.Ws1 = nn.Linear(2 * u, da, bias=False)
        self.Ws2 = nn.Linear(da, r, bias=False)
        self.lin3 = nn.Linear(r * 2 * u, hidden_classifier_units)  # MLP
        self.pred = nn.Linear(hidden_classifier_units, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.relu = torch.nn.ReLU()
        # softmax is included in CrossEntropyLoss


    def self_attention(self, H):
        out_ws1 = self.Ws1(H)               # Ws1^T * H
        out_tahn = torch.tanh(out_ws1)      # tahn (Ws1^T * Ht)
        out_ws2 = self.Ws2(out_tahn)        # tahn (Ws1 * Ht) * Ws2^T
        out_a = self.softmax(out_ws2)       # softmax( Ws2 * tahn (Ws1 * Ht) )
        return out_a

    def get_embeddings(self, w):
        if self.embeddings == 'bert':
            embedding_output = self.bert.embeddings(w)
        elif self.embeddings == 'elmo':
            character_ids = batch_to_ids(ids2str(w, self.vocab))
            # embeddings = self.elmo_embedder.batch_to_embeddings()
            embeddings = self.elmo(character_ids.to(device=self.device))
            embedding_output = embeddings['elmo_representations'][0].to(self.device)
        else:
            embedding_output = self.embeddings(w)
        return embedding_output

    def forward(self, w):
        embedding_output = self.get_embeddings(w)

        # bidirectional pass of LSTM - H
        out_lstm, _ = self.bilstm(embedding_output)
        # self-attention mechanism - A
        out_a = self.self_attention(out_lstm)

        out_a = out_a.transpose(2, 1)
        out_m = torch.bmm(out_a, out_lstm)  # M = A * H - visualization
        out = out_m.view(out_m.shape[0], self.r * 2 * self.u)

        # MLP
        # out = self.lin3(self.relu(self.dropout(out)))
        out = self.relu(self.lin3(self.dropout(out)))
        pred = self.pred(self.dropout(out))

        return pred, out_a
