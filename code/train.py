# -*- coding: utf-8 -*-
'''
This is an example for implementating of some NLP models using PyTorch and TensorboardX
Our dataset would be Romeo and Juliet from the shakespeare corpus from nltk.
```
import nltk; nltk.download('shakespeare')
```
'''
# Import required modules
import numpy as np

# Import PyTorch module
import torch
import torch.nn as nn
import torch.functional as F

# Import PyTorch Ignite modules
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

# Import TensorboardX modules
from tensorboardX import SummaryWriter

# Import local modules
import models.rnn as rnn

##################################################
# Initialization
##################################################

# Configure random seed
seed = 34
ramdom.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 32
epoch = 10

embedding_dim = 512
hidden_dim = 256

grad_clip_value = 1

dropout = 0
learning_rate = 10e-4

min_count = 0

num_rnn_layers = 2
num_linear_layers = 1

##################################################
# Load Data
##################################################

from torch.utils.data import DataLoader

##################################################
# Build Model
##################################################



##################################################
# Train
##################################################

model = rnn.RNN(input_dim=vocab_size, hidden_dim=hidden_dim, output_dim=, rnn_layers=, linear_layers=)

optimizer = torch.optim.Adam()
criterion = torch.nn.CrossEntropyLoss()

def fit(self, data, epochs, batch_size, optimizer, learning_rate, criterion):

    # TODO: fit model like Keras
    self.train() # Set model mode to training
    for epoch in range(epochs):
        for x, y in tqdm(data):
            x = x.to(device)
            y = y.view(-1).to(device)

            pred_y = self.forward(x)
            pred_y = pred_y.view(-1, tokenizer.vocab_size())
            loss = criterion(pred_y, y)
            total_loss += float(loss) / len(dataset)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip_value)
            optimizer.step()