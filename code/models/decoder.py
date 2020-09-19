# -*- coding: utf-8 -*-
"""
This is an example for implementing a decoder.
Notes:
- We will be using torch.nn modules here.
- torch.nn.functional is low-level, stateless functions that are used by the modules, you could use them for flexibility(?)
"""
import torch
import torch.nn as nn
import tqdm

def decoder(model, features, max_seq_len):
    results = [model.bos_token_idx]
    prob = []

    while len(results) < max_len and results[-1] != model.eos_token_idx:

        # 1 x 1 x V
        yt = model(
            features,
            torch.LongTensor(results).unsqueeze(0) # 1 x S
        )[:,-1,:].squeeze().argmax()

        pr = torch.nn.functional.softmax(model(
            features,
            torch.LongTensor(results).unsqueeze(0) # 1 x S
        )[:,-1,:].squeeze(), dim=0).max()

        results.append(yt)
        prob.append(pr)

    return model.tokenizer.decode([results]), prob