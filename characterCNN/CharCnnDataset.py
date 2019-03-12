import json
import torch
import pandas as pd

import torchtext
from torchtext.data import Example, TabularDataset

from characterCNN.CharField import CharField

with open('config.json', 'r') as f:
    config = json.load(f)
    config_data = config['data_processing']
    config_model = config['model']
    config_train = config['model']['train']


class CharCnnDataset(TabularDataset):
    def __init__(self, path=config_data['dataset_file'], format='tsv', fields=[], **kwargs):
        super(CharCnnDataset, self).__init__(path, format, fields, **kwargs)

    def get_class_weight(self):
        dataset_size = self.__len__()
        classes = set(self.labels)
        num_classes = len(classes)

        if config['balanced_weights']:
            return num_classes, [[1.0] * num_classes]

        classes_occurrences = [0] * num_classes
        for label in self.labels:
            classes_occurrences[label] += 1

        max_count = max(classes_occurrences)
        class_weights = [max_count / n for n in classes_occurrences]

        print(f'Dataset size: {dataset_size} N. classes: {num_classes}' +
              f'Count by classes: {classes_occurrences} Weights: {class_weights}')
        return num_classes, class_weights
