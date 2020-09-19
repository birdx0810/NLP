# -*- coding: utf-8 -*-
"""
This is an example for implementing a 'Pointer-Generator' model class using PyTorch.
Notes:
- We will be using torch.nn modules here.
- torch.nn.functional is low-level, stateless functions that are used by the modules, you could use them for flexibility(?)
"""

import torch
import torch.nn as nn

class Pointer_Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_layers, linear_layers, bidirectional, vocab_size, embedding_weight=None):
        super(Pointer_Generator, self).__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_dim,
                                            padding_idx=pad_token_id)

        # Bidirectional Encoder
        self.rnn_encoder = nn.RNN(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=encoder_layers,
                                  bidirectional=True,
                                  batch_first=True)

        # Encoder Linear for projection (reduce states)
        self.linear_encoder = nn.Linear(hidden_dim*2, hidden_dim)

        # Unidirectional Decoder
        self.rnn_decoder = nn.RNN(input_size=hidden_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=decoder_layers,
                                  bidirectional=False,
                                  batch_first=True)

        # Decoder Linear
        self.linear_decoder = nn.ModuleList()

        self.linear_decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_decoder.append(nn.Linear(hidden_dim, vocab_size))

    def forward(self, x):
        x = self.embedding_layer(x)

        representation = self
        pass

    def attention(self, decoder_state, coverage=None)):
        pass
