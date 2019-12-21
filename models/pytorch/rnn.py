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

    def fit(self, data, epochs, batch_size, optimizer, learning_rate, criterion):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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