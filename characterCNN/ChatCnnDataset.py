import json
import torch
import pandas as pd

from torchtext.data import Example, TabularDataset

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

        classes_occurrences = [0] * num_classes
        for label in self.labels:
            classes_occurrences[label] += 1
        class_weights = [n/dataset_size for n in classes_occurrences]

        class_weights_modified = class_weights.copy()
        max_count = max(class_weights_modified)
        class_weights_modified[class_weights_modified.index(max(class_weights_modified))] = 1
        class_weights_modified = [max_count/w for w in class_weights_modified]

        print(f' : {[n/dataset_size for n in classes_occurrences]}')
        print(f'w: {[dataset_size/n for n in classes_occurrences]}')
        print(f'w mod: {class_weights_modified}')

        return num_classes, class_weights_modified
