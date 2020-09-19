# -*- coding: utf-8 -*-
"""
This is an example for implementing a RNN model class using PyTorch.
Notes:
- We will be using torch.nn modules here.
- torch.nn.functional is low-level, stateless functions that are used by the modules, you could use them for flexibility(?)
"""
import torch
import torch.nn as nn
import tqdm

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_layers, linear_layers, bidirectional, vocab_size, is_logits, embedding_weight=None):
        super(RNN, self).__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_dim,
                                            padding_idx=pad_token_id)

        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)
        else:
            with torch.no_grad():
                for parameter in self.embedding_layer.parameters():
                    parameter.normal_(mean=0.0, std=0.1)


        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=rnn_layers,
                                bidirectional=bidirectional,
                                batch_first=True)

        with torch.no_grad():
            for parameter in self.rnn_layer.parameters():
                parameter.normal_(mean=0.0, std=0.1)

        # Linear layer
        self.linear = nn.ModuleList()

        for _ in range(linear_layers):
            if bidirectional and _ == 0:
                self.linear.append(torch.nn.Linear(hidden_dim*2, hidden_dim))
            else:
                self.linear.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linear.append(torch.nn.ReLU())

        self.linear.append(torch.nn.Linear(hidden_dim, output_dim))

        if is_logits:
            self.linear.append(torch.nn.Softmax())

        with torch.no_grad():
            for parameter in self.sequential.parameters():
                parameter.normal_(mean=0.0, std=0.1)

    def forward(self, x):

        # Embedding layer
        # B x S x E
        embedding_tensor = self.embedding_layer(x)

        # RNN layer
        # B x S x H
        hidden_tensor, _ = self.rnn_layer(embedding_tensor)

        # Linear layer
        # B x S x O
        y = self.linear(hidden_tensor)

        # Inference layer
        # B x S x V
        # yt = ht.matmul(self.embedding_layer.weight.transpose(0, 1))

        return y

class LSTM(RNN):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_layers, linear_layers, bidirectional):
        super(RNN, self).__init__()

    # Overload RNN layer
    self.rnn_layer = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             bidirectional=bidirectional,
                             batch_first=True)

class GRU(RNN):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_layers, linear_layers, bidirectional):
        super(RNN, self).__init__()

    # Overload RNN layer
    self.rnn_layer = nn.GRU(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
