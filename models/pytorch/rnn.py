import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, linear_layers, rnn_layers, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()

        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                batch_first=True)

        # Linear layer
        self.linear = []

        for _ in range(linear_layers):
            self.linear.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linear.append(torch.nn.ReLU())

