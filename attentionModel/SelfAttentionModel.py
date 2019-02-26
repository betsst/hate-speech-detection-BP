# model based on Structured Self-attentive Sentence Embedding - https://arxiv.org/pdf/1703.03130.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 3
batch_size = 16
hidden_units = 299     # hidden unit for each unidirectional LSTM
d_a = 32
extraction_count = 16      # number of different parts to be extracted from the sentence.


class SelfAttentionModel(nn.Module):
    def __init__(self, vocabulary, u=hidden_units, da=d_a, r=extraction_count, batch_size=batch_size):
        super(SelfAttentionModel, self).__init__()
        self.embeddings_dim = vocabulary.vectors.shape[1]
        self.embeddings = nn.Embedding(len(vocabulary), self.embeddings_dim)
        self.embeddings.weight.data.copy_(vocabulary.vectors)
        self.batch_size = batch_size
        self.u = u
        self.r = r

        self.bilstm = nn.LSTM(input_size=self.embeddings_dim, hidden_size=u, num_layers=1, bidirectional=True)  # batch_first=True
        self.Ws1 = nn.Linear(2 * u, da, bias=False)
        self.Ws2 = nn.Linear(da, r, bias=False)
        # self.lin3 = nn.Linear(r, num_classes)
        self.lin3 = nn.Linear(r * 2 * u, num_classes)
        self.softmax = nn.Softmax(dim=1)
        # softmax is included in CrossEntropyLoss

    # def init_hidden(self):
    #     return (Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)),
    #             Variable(torch.zeros(1, self.batch_size, self.lstm_hid_dim)))

    def self_attention(self, H):
        out_ws1 = self.Ws1(H)           # Ws1^T * H
        out_tahn = torch.tanh(out_ws1)      # tahn (Ws1^T * Ht)
        out_ws2 = self.Ws2(out_tahn)    # tahn (Ws1 * Ht) * Ws2^T
        out_a = self.softmax(out_ws2)      # softmax( Ws2 * tahn (Ws1 * Ht) )
        return out_a

    def forward(self, w):
        self.input_shape = w.shape
        out = self.embeddings(w)
        # bidirectional pass of LSTM - H
        out_lstm, _ = self.bilstm(out)

        # self-attention mechanism - A
        out_a = self.self_attention(out_lstm)

        out_a = out_a.transpose(2, 1)
        out_m = torch.bmm(out_a, out_lstm)  # M = A * H - visualization
        out = out_m.view(self.batch_size, self.r * 2 * self.u)
        out = self.lin3(out)

        return out, out_m
