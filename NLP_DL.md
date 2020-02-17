# NLP Deep Learning

## Recurrent Neural Networks
> RNN Based Language Model (Mikolov et al., 2013)

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
Could be thought of as two independant RNNs, one starting from the first word of the sentence, the other from the back of the sentence, and the outputs of both forward and backward RNNs would be concatenated. 

It is considered that by doing so, we are obtaining information from the past, and also from the future, taking into context of a paragraph as a whole.

## LSTM
> Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling (Sak et al., 2014)

Was first proposed in 1997 by Sepp Hochreiter and Jürgen Schmidhuber for dealing the vanishing gradient problem.

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

![](https://i.imgur.com/SxudGs4.png)

## GRU
> Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (Cho et al., 2014)

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
> Convolutional Neural Networks for Sentence Classification (Yoon Kim, 2014)

![](https://i.imgur.com/YdWcjGO.png)

Hyperparameters to consider
- Padding (Narrow vs. Wide convolution)
![](https://i.imgur.com/ZYoGlzN.png)
- Stride
![](https://i.imgur.com/TZ6rqby.png)

**Notes:**
- Convolutions view a sentence as a bag-of-words, therefore tends to lose information of local order of words.

## Pooling Layer

Used for dimensionality reduction. Could be thought of as extracting the most relavent information (timestamp, word, pixel) from its input. 

![](https://i.imgur.com/4XAPyVB.png)

### Pooling over filter
> Character-Aware Neural Language Models (Kim et al., 2013)

Useful for classifiers as it could be directly fed into a feed-forward network (FCL/MLP).

### Pooling over timesteps
> Fully Character-Level Neural Machine Translation without Explicit Segmentation (Lee et al., 2016)

**Note:**
Pooling loses information about local order of words as it is meant for dimensionality reduction.

## Memory Networks
> End-to-end Memory Networks (Sukhbaatar et al., 2014)

![](https://i.imgur.com/TcFWgnJ.png)

Was first proposed as a network for that has a long-term external memory that could be read and written. It was redesigned as an extention of RNNsearch which has more flexibility and requires less supervision.

### Pseudocode
```

```

## Seq2seq
> Sequence-to-Sequence Learning With Neural Networks (Sutskever et al., 2014)

Encodes a sentence/image into a thought/percept

## Attention
> Attention Is All You Need (Vaswani et al, 2017)

![](https://i.imgur.com/V0y4S2A.png)

### Pseudocode
```

```

## Pointer Networks

![](https://i.imgur.com/ORoovhQ.png)


