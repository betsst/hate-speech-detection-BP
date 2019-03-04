import torch
import pandas as pd

from torch.utils.data import Dataset


class CharCnnDataset(Dataset):
    def __init__(self, filename, read_from_file=False):
        self.csv_file = pd.read_csv(filename)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[index, 0])
        image = io.imread(img_name)
        landmarks = self.csv_file.iloc[index, 1:]
        sample = {'label': label, 'one_hot_encoding': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
