# NLP Deep Learning

## Recurrent Networks
> 

Could be thought of as multiple copies of the same network connected to each other, which output is based on the input of the previous state (context). This network could show dynamic temporal behavior for a time sequence.

The objective function for language modeling is usually Cross Entropy over the vocabulary. Or could be mapped to designated labels with a FFN/FCL/MLP.

![](https://i.imgur.com/ydKpjpR.jpg)

### Pseudocode
```python
def RNN(prev_hs, x_t):
    combine = prev_hs + x_t
    hidden_state = tanh(combine)
    return hidden_state

fnn = FNN() # feed-forward neural-network
hidden_state = np.array(len(sentence)) # length of words in input

for words in sentence:    # loop through words until fin
    output, hidden_state = rnn(word, hidden_state)
    
prediction = fnn(output) # final output should have data from all past hidden states
```

### Review
- Suffers from vanishing gradient problem
- Variants:
    - Bi-directional
    - LSTM, GRU
    - Memory Networks

### Bi-directional

![](https://i.imgur.com/SxudGs4.png)

## LSTM

- Forget Gate: Defines which information to forget
- Input Gate: Updates cell state
- Output Gate： Defines hidden state

### Pseudocode
```python
def LSTM(prev_cs, prev_hs, x_t):
    combine = concat(x_t, prev_hs)
    
    # Forget Gate
    forget_weight = sigmoid(combine) 
    
    # Input Gate
    input_weight = sigmoid(combine)
    candidate = tanh(combine) # Regulate network

    # Update Cell State
    new_cell_state = (prev_cs * forget_weight) + (candidate * input_weight) # Forget + Input
    
    # Output Gate
    output_weight = sigmoid(combine)
    new_hidden_state = output_weight * tanh(C_t)

    return new_hidden_state, new_cell_state
    
c_state = [0,0,0]
h_state = [0,0,0]

for word in sentence:
    h_state, c_state = LSTM(c_state, h_state, word)
```

## GRU

- Reset Gate: Decides past information to forget
- Update Gate: Updates information of hidden state (forget + input)

### Pseudocode
```python
def GRU(prev_hs, x_t):
    combine = concat(x_t, prev_hs)
    
    # Reset Gate
    reset_weight = sigmoid(combine)
    reset = prev_hs * reset_weight

    # Update Gate
    update_weight = sigmoid(combine)
    hidden_state = prev_hs * (1 - update_weight)

    # Output
    output = concat(reset, tanh(x_t))
    output = update_weight * output
    hidden_state = hidden_state + output

    return hidden_state
    
h_state = [0,0,0]

h_state = GRU(h_state, x_t)
```



## Convolution Networks

![](https://i.imgur.com/YdWcjGO.png)

### Pseudocode
```

```

## Pooling Layer

![](https://i.imgur.com/4XAPyVB.png)

### Pooling over filter

### Pooling over timesteps

### Pseudocode
```

```

## Attention

![](https://i.imgur.com/V0y4S2A.png)

### Pseudocode
```

```

## AutoEncoders
### Seq2seq



### Variational AE

### Denoising AE

### Pointer Networks

![](https://i.imgur.com/ORoovhQ.png)

## Memory Networks

![](https://i.imgur.com/EJIAwK4.png)