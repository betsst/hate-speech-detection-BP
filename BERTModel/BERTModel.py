import torch
import torch.nn as nn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertModel, BertPreTrainedModel


class BERTModel(BertPreTrainedModel):
    def __init__(self, config, num_classes, bert_model):
        super(BERTModel, self).__init__(config)
        self.num_classes = num_classes
        self.bert = BertModel(config).from_pretrained(bert_model)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.linear = torch.nn.Linear(config.hidden_size, num_classes)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output
        pooled_output = self.dropout(pooled_output)
        out = self.linear(pooled_output)
        return out

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
