# model based on Structured Self-attentive Sentence Embedding - https://arxiv.org/pdf/1703.03130.pdf
import json

import torch
import torch.nn as nn


class SelfAttentionModel(nn.Module):
    with open('config.json', 'r') as f:
        config = json.load(f)

    def __init__(self, vocabulary, num_classes, device, u=config['hidden_units'], da=config['d_a'], r=config['extraction_count']):
        super(SelfAttentionModel, self).__init__()
        self.vocab = vocabulary
        self.embeddings_dim = vocabulary.vectors.shape[1]
        self.embeddings = nn.Embedding(len(vocabulary), self.embeddings_dim)
        self.vocab.vectors[self.vocab.stoi['<pad>']] = -1e8 * self.embeddings_dim  # pad token masking
        self.embeddings.weight.data.copy_(vocabulary.vectors.to(device))

        self.u = u  # hidden unit for each unidirectional LSTM
        self.r = r  # number of different parts to be extracted from the sentence.

        self.bilstm = nn.LSTM(input_size=self.embeddings_dim, hidden_size=u, num_layers=1, bidirectional=True)  # batch_first=True
        self.Ws1 = nn.Linear(2 * u, da, bias=False)
        self.Ws2 = nn.Linear(da, r, bias=False)
        # self.lin3 = nn.Linear(r, num_classes)
        self.lin3 = nn.Linear(r * 2 * u, num_classes)
        self.softmax = nn.Softmax(dim=1)
        # softmax is included in CrossEntropyLoss

    def self_attention(self, H):
        out_ws1 = self.Ws1(H)           # Ws1^T * H
        out_tahn = torch.tanh(out_ws1)      # tahn (Ws1^T * Ht)
        out_ws2 = self.Ws2(out_tahn)    # tahn (Ws1 * Ht) * Ws2^T
        out_a = self.softmax(out_ws2)      # softmax( Ws2 * tahn (Ws1 * Ht) )
        return out_a

    def forward(self, w):
        out = self.embeddings(w)
        # bidirectional pass of LSTM - H
        out_lstm, _ = self.bilstm(out)
        # self-attention mechanism - A
        out_a = self.self_attention(out_lstm)

        out_a = out_a.transpose(2, 1)
        out_m = torch.bmm(out_a, out_lstm)  # M = A * H - visualization
        out = out_m.view(out_m.shape[0], self.r * 2 * self.u)
        out = self.lin3(out)

        return out, out_a
