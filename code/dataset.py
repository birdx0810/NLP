# -*- coding: utf-8 -*-
'''
This is an example for creating a dataset for PyTorch from the torch.utils.data.Dataset class.
Note that the `__len__()` function and `__getitem__()` function should be overwritten.
'''

from torch.utils.data import Dataset
import torch.nn.utils.rnn
import torch

def ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
        pass

    def __len__(self):
        '''
        The total number of words/sentences
        '''
        return len(self.data)

    def __getitem__(self, index):
        '''
        Get an example from data
        '''
        return self.data[index][]