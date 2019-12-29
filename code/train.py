# -*- coding:utf-8 -*-
'''
This is an example for implementating of some NLP models.
'''
import torch
import pytorch.rnn as rnn

model = rnn.RNN()

optimizer = 

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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip_value)
            optimizer.step()