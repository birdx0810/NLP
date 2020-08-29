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

linear = Linear()
hidden_state = np.array(len(sentence)) # length of words in input

for words in sentence:    # loop through words until fin
    output, hidden_state = RNN(word, hidden_state)
    
prediction = linear(output) # final output should have data from all past hidden states
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

### Gradient Explosion/Vanishing: The problem with RNNs
RNNs can learn long dependencies but inpractice they tend to be biased towards their most recent inputs in the sequence.

$$
\begin{aligned}
h_t &= \tanh(W_Ix_t + W_Rh_{t-1} + b_h) \\
y_t &= W_Oh_t + b_y
\end{aligned}
$$

To backpropagate, we need to compute the gradient of the cost function $C$ w.r.t the recurrent weights $W_R$. For step $t$, the using the chain rule, we could derive:

$$
\begin{aligned}\partial
\frac{\partial C_t}{\partial W_R} &= \sum_{i=0}^t
\frac{\partial C_t}{\partial y_t}
\frac{\partial y_t}{\partial h_t}
\frac{\partial h_t}{\partial h_i}
\frac{\partial h_i}{\partial W_R} \\
\frac{\partial h_t}{\partial h_i} &= \prod_{k=i}^{t-1}
\frac{\partial h_{k-1}}{\partial h_k} \\
\frac{\partial h_k}{\partial h_1} &= \prod_{i}^{k} \text{diag}(f'(W_Ix_i + W_Rh+{i-1}))W_R
\end{aligned}
$$

The derivative $\frac{\partial h_{k+1}}{\partial h_k}$ is essentially telling us how much our hidden state at $k+1$ will change if we change the hidden state at time $k$. If $W_R$ is too small the derivative would be 0, and $\infty$ if $W_R$ is too big.

This is known as the vanishing gradient ($W_R>1$) or exploding gradient ($W_R<1$). LSTMs uses gated operations to combat the Exploding/Vanishing Gradient problem.

#### Reference
- [Why LSTMs Stop Your Gradients From Vanishing](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)
- [Understanding, Deriving and Extending the LSTM ](https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)

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

for word in sentence:
    h_state = GRU(h_state, word)
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

MemNN was first proposed as a network for that has a long-term external memory that could be read and written. Consists of a $G(\cdot)$, which is used to update the memory; and $O(\cdot)$ which produces the output weights from the memory.

It was redesigned into (MemN2N) as an extention of RNNsearch which has more flexibility and requires less supervision. 

![](https://i.imgur.com/bwWOaIj.png)

There are two types of weighting:
1. Adjacent: $A^{k+1} = C^{k}$
2. Layer-wise: $A^1 = ... = A^k, C^1 = ... = C^k$

The pseudocode is the implementation of the Adjacent weighting model. 

### Pseudocode
```python
def MemN2N(prev_C, memory, x_t):
    # Adjacent Weighting
    next_A = [0,0,0]
    
    # Get probability vectors over inputs (Addressing)
    m_i = x_t * prev_C(memory)
    p_weight = softmax(m_i)

    # Generate weighted output vector (Read)
    c_i = next_A(sentences)
    output = p_weight * c_i

    return next_A, output

memory = []

h_state = [0,0,0]

for sentence in sentences:
    # sentence = [w_1, w_2, ..., w_n]
    question = check_if_question(sentence)
    
    if not question:
        # Write to memory `G()`, could be EMR
        memory.append(sentence)
    elif question:
        # Hops to iterate (RNN cells) 
        for hop in hops:
            # Generates output `O()`
            h_state, sentence = MemN2N(h_state, memory, sentence)
        
        # Flush memory
        memory = []

pred = Softmax(h_state(o+u))
```

## Seq2seq
> Sequence-to-Sequence Learning With Neural Networks (Sutskever et al., 2014)

![](https://i.imgur.com/SJXP9lB.png)

Encodes/extract a sentence/image into a thought/percept from a sequence via an bi-LSTM; and decodes it into a different sequence (img -> sent, en -> fr)

### Pseudocode
```python
def Seq2Seq():
    # TODO: Encoder & Decoder
```

## Attention
> Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al, 2014)

Attention mechanisms was first introduced into a seq2seq model for machine translation.
1. Do we need to translate word $\to V_t +1$
2. Which word do we need to translate (src) $\to$ Classifier
3. What does it need to be translated to (tgt) $\to$ Attention

> Attention Is All You Need (Vaswani et al, 2017)

Using an attention mechanism to train the model to focus on certain words in the sentence at different timesteps. The attention mechanism could be additive or multiplicative. In this paper, the author uses the dot-product (multiplicative) attention for self-attention. The larger the dot-product (1), the higher similarity.

![](https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s640/transform20fps.gif)

Query, Key and Value for all words should be the same size (vocab_size, emb_size).

### Pseudocode
```python
def SelfAttention(hidden_size, no_att_heads, att_mask):
    # Initialize Q, K, V with shape (BatchSize * SeqLen * NoAttHeads * AttHeadSize)
    att_head_size = hidden_size / no_att_heads
    query = key = value = linear(hidden_size, no_att_heads * att_head_size)
    d_k = sqrt(att_head_size)

    # Calculate Attention Score
    att_score = (query @ key.transpose())

    # Scale to prevent gradient exploding (d_k = query.size(-1))
    att_score = att_score / d_k

    # Apply Attention Mask
    att_score = att_score + att_mask

    # Normalize attention scores to probability
    att_score = softmax(att_score)

    # Calculate Context Layer (Value)
    context_layer = att_score @ value
    context_layer.reshape()

    return context_layer, att_score
```

### Different Attentions
> Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
![](https://i.imgur.com/i5bCXau.png)
- Global Attention
    - Global attention uses **ALL encoder hidden states** to define the attention based context vector for each decoder step
- Local Attention
    - Local attention attends to only a few hidden states that fall within a smaller window
- Hard Attention
    - Hard attention uses attention scores to select the $i$-th location (argmax)
- Soft Attention
    - Soft attention uses attention scores as the weighted average context vector calculation

## Pointer Networks

![](https://i.imgur.com/ORoovhQ.png)

## Siamese Networks
> Siamese Recurrent Architectures for Learning Sentence Similarity (Mueller and Thyagarajan, 2016)

Siamese Networks was first used for signature recognition. These networks are built with two encoders that share the same weights. The input of the encoders are two different images (or sentences), and the similarity (cosine or L2) between the representations is computed (ranking loss/contrastive loss).

### Pseudocode
```python
def Siamese(x_1, x_2):

    # Initialize encoder and initial weights
    encoder = LSTM()
    h_1, c_1 = encoder.init_weights()
    h_2, c_2 = encoder.init_weights()

    for word in x_1:
        h_1, c_1 = encoder_a(word, h1)
    for word in x_2:
        h_2, c_2 = encoder_b(word, h1)

    features = [ cosine_similarity(h_1, h_2), euclidean_distance(h_1, h_2) ]
    
    prediction = linear(features)

    return prediction
```
