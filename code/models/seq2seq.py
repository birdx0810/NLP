# -*- coding: utf-8 -*-
'''
This is an example for implementing a seq2seq model class using PyTorch.
Notes:
- We will be using torch.nn modules here.
- torch.nn.functional is low-level, stateless functions that are used by the modules, you could use them for flexibility(?)
'''
import torch
import torch.nn as nn
import tqdm

class Seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, encoder_layers, decoder_layers, bidirectional, vocab_size, embedding_weight=None):
        super(Seq2seq, self).__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_dim,
                                            padding_idx=pad_token_id)

        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # Encoder
        self.Encoder(input_dim=embedding_size,
                     hidden_dim=hidden_dim,
                     num_layers=encoder_layers,
                     bidirectional=bidirectional)

        # Decoder
        self.Decoder(input_dim=hidden_dim,
                     hidden_dim=vocab_size,
                     num_layers=decoder_layers,
                     bidirectional=bidirectional)

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, bidirectional):
            super(Encoder, self).__init__()

        self.encoder = nn.RNN(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True)

        def forward(self, sequence):
            output, _ = self.encoder(sequence)
            return

    class Decoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, bidirectional):
            super(Decoder, self).__init__()

        if bidirectional:
            input_dim = input_dim*2

        self.decoder = nn.RNN(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=rnn_layers,
                              batch_first=True)
        
        self.linear = nn.Linear(torch.nn.Linear(hidden_dim, vocab_size))

        def forward(self, thought):
            output, _ = self.decoder(thought)
            logits = self.linear(output)
            pass

    def forward(self, x):

        # Embedding layer
        x = self.embedding_layer(x)

        # Encoder
        hidden_tensor, _ = self.encoder(x)

        # Decoder
        hidden_tensor, _ = self.decoder(hidden_tensor)

        # Output
        y = hidden_tensor.matmul(x.weight.transpose(0,1))

        return y

    def generator(self, y):
        #TODO: sentence generator with beam search
        pass


