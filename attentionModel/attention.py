import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torchtext import data as torchdata
from torchtext.vocab import GloVe
from tqdm import tqdm

from attentionModel.SelfAttentionModel import SelfAttentionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
learning_rate = 0.003

TEXT = torchdata.Field(tokenize="spacy", sequential=True, lower=True, batch_first=True)
LABELS = torchdata.Field(use_vocab=False, sequential=False, preprocessing=lambda x: int(x), is_target=True)
tabular_dataset = torchdata.TabularDataset('../data/davidson2.tsv', format='tsv', fields=[('labels', LABELS),
                                                                                          ('text', TEXT)])

train_iterator = torchdata.BucketIterator(tabular_dataset, batch_size=16, sort_key=lambda x: len(x.text), device=device,
                                          sort_within_batch=False)
TEXT.build_vocab(tabular_dataset, vectors=GloVe(name='6B', dim=300))
LABELS.build_vocab(tabular_dataset)

selfAttModel = SelfAttentionModel(vocabulary=TEXT.vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(selfAttModel.parameters(), lr=learning_rate)

train_loss = 0
total_correct = 0
total_batches = len(train_iterator.data()) // train_iterator.batch_size
pbar = tqdm(total=total_batches)
model_predictions = []
true_labels = []

selfAttModel.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_iterator):
        predictions, attention = selfAttModel(batch.text)  # forward pass

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(1)

# eval
selfAttModel.eval()

# save model
torch.save(selfAttModel.state_dict(), 'modelSelfAtt.ckpt')
