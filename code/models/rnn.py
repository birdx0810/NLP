# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import tqdm

class RNN(nn.Module):
    def __init__(self, rnn_layers, linear_layers, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        
        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=rnn_layers,
                                batch_first=True)

        # Linear layer
        self.linear = nn.ModuleList()

        for _ in range(linear_layers):
            self.linear.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linear.append(torch.nn.ReLU())
        
        self.linear.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):

        # RNN layer
        hidden_tensor, _ = self.rnn_layer(x)

        # Linear layer
        hidden_tensor = self.linear(hidden_tensor)

        # Output
        y = hidden_tensor.matmul(x.weight.transpose(0,1))

        return y

class LSTM(RNN):
    def __init__(self, rnn_layers, linear_layers, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()

    # Overload RNN layer
    self.rnn_layer = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

class GRU(RNN):
    def __init__(self, rnn_layers, linear_layers, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()

    # Overload RNN layer
    self.rnn_layer = nn.GRU(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)