# -*- coding: utf-8 -*-
'''
This is an example for creating a dataset for PyTorch from the torch.utils.data.Dataset class.
Note:
- the Dataset class represents a map-style dataset, we will be using this for language modeling where our loss is cross-entropy (i.e. Vocab * Vocab mapping).
- the `__len__()` function and `__getitem__()` function should be overwritten.
- TODO: Write dataset modules for benchmark leaderboards (as written below)
'''

from torch.utils.data import Dataset
import torch.nn.utils.rnn
import torch

def ShakespeareDataset(Dataset):
    def __init__(self):
        '''
        Initialization dataset parameters
        '''
        self.path = './data/sentences.pickle'
        self.data = open(self.path, 'r')

    def __len__(self):
        '''
        The total number of words/sentences
        '''
        return len(self.data)

    def __getitem__(self, index):
        '''
        Get an example from data
        '''
        x = self.data[index][:-1]
        y = self.data[index][1:]
        return x, y

    def collate_fn(self, batch):
        '''
        Define batch size of dataset
        '''
        x = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch],
                                            batch_first=True,
                                            padding_value=pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=pad_token_id)

##############################
# TODO: Create Datasets
# - SNLI
# - SQuAD
# - CNN/DM
# - Translations
#    - EN/CN/JP/KR
# - BioMed
#    - CDR
#    - CPI
##############################