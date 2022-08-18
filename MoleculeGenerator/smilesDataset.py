import numpy as np
import pandas as pd
import torch, json

from torch.utils.data import Dataset


class SMILESDataset(Dataset):

    def __init__(self,path):
        """
        Init method for the dataset
        :param path: Path to the datafile
        """
        super(SMILESDataset, self).__init__()

        self.smiles = pd.read_csv(path, header = None, squeeze = True)
        self.len = len(self.smiles)

        with open("data/int2char.txt", 'r') as f:
            self.int2char = json.load(f)
            self.int2char = {int(k): v for k, v in self.int2char.items()}

        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.uniqueCount = len(self.int2char)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.smiles[index]

    def split(self, batch):
        """
        Padds a batch of sequences and extracts their labels
        :param samples: a batch of seuqences
        :return: a padded batch of one hot vector sequences and their labels
        """
        samples = []
        targets = []
        for sample in batch:
            indeces = torch.tensor([self.char2int[c] for c in sample])
            sample = np.insert(indeces,0,1)
            target = torch.tensor(np.append(indeces,2))
            samples.append(sample)
            targets.append(target)
        samples = torch.nn.utils.rnn.pad_sequence(samples)
        targets = torch.nn.utils.rnn.pad_sequence(targets)
        samples = torch.stack([self.oneHotEncoding(index) for index in samples])
        return samples, targets

    def oneHotEncoding(self, index):
        index = index.long()
        ones = torch.sparse.torch.eye(self.getAmountOfUniqueChars())
        return ones.index_select(0,index.data)

    def getAmountOfUniqueChars(self):
        return self.uniqueCount

    def getOneHotDict(self):
        return self.oneHotDict