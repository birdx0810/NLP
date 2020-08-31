# KALM: Knowledge-Augmented Language Model and its Application to Unsupervised Named-Entity Recognition

> Apr 2019
> Angli Liu, Jingfei Du, Veselin Stoyanov (NAACL 2019)
> Affiliates: Facebook
> Paper: [Link](https://arxiv.org/abs/1904.04458)
> Code: [Unofficial](https://github.com/raghavjajodia/unsupervised_nlu/blob/master/models.py) (in progress, using vanilla LSTM)

## Introduction
- Current language models are unable to encode and decode factual knowledge
- Language models only learn to represent the more popular named entities
- Inspired by [A Neural Knowledge Language Model](https://arxiv.org/abs/1608.00318) (Ahn et al., 2016)
- A.k.a Latent LM, Latent LM /w Type Representation

Proposed Knowledge Augmented Language Model (KALM), an end-to-end model using a gating mechanism that learns if a word is a reference to entity.

## Methodology
### Basic Model Architecture
KALM extends a traditional RNNLM, as it predicts the probability of words from a vocabulary and also named entities of a specific type. Each type has a separate vocabulary $\{V_0, ..., V_K\}$ collected from a [KB](https://github.com/uclanlp/NamedEntityLanguageModel).

![](https://i.imgur.com/W53NGWk.png =480x)

For a given word (context embedding) $c_t$, KALM computes a probability that the word represents an entity of a type. With the overall probability of a word being:
- $\Pr(\tau_{t+1} = j | c_t)$: the weighted sum of the type probabilities (`tag_logits`)
- $\Pr(y_{t+1}| \tau_{t+1} = j, c_t)$: the probability of the word under the given type (`word_distribution_logits`)

$$
\begin{equation}
\begin{aligned}\Pr(y_{t+1}|c_t) 
&= \sum_{j-0}^K \Pr(y_{t+1}, \tau_{t+1} = j |c_t) \\
&= \sum_{j-0}^K \Pr(y_{t+1} | \tau_{t+1} = j, c_t) \cdot \Pr(\tau_{t+1} = j | c_t)\\
\end{aligned}
\end{equation}\tag{1}\label{1}
$$

We first learn the probability of the entity type of the word from the hidden state $h_{t+1}$ of the LSTM output. The hidden state is first projected match the size of the entity type embeddings using $W^h$. $W^{e}$ is the type embedding matrix with size $W^h=100$

$$
\begin{equation}
\begin{aligned}
\Pr(\tau_{t+1} = j|c_t) &= \frac{\exp(W^e_j \cdot (W^h \cdot h_t))}{\sum_{k=0}^{K}\exp(W^e_j \cdot (W^h \cdot h_t))}
\end{aligned}
\end{equation}\tag{2}\label{2}
$$

After calculating the probability of the word being the entity type $\tau_{t+1}$, we could find $\Pr(y_{t+1}|\tau_{t+1}, c_t)$, the probability of each word given a particular type and context.

$$
\begin{equation}
\begin{aligned}
\Pr(y_{t+1} = i|\tau_{t+1}, c_t) &= \frac{\exp(W^{p, j}_{i,:} \cdot h_t)}{\sum_{w=1}^{|V_j|}\exp(W^{p, j}_{i,:} \cdot h_t)} \end{aligned}
\end{equation}\tag{3}\label{3}
$$

The hidden state of the LSTM is first passed through a linear projection $W^{p,j}$ and normalized through a softmax function. $W^{p,j}$ is a learnt type-specific projection matrix that determines if a word represents an entity of a type.

#### Pseudocode
1. Embedding Layer (+ Dropout)
2. Generate hidden state from LSTM
    - $h_t = \text{LSTM}(h_{t-1}, \text{word})$
3. Linear Projection of $h_t$ to Lower Dim
    - $h_l = W^h \cdot h_t$
4. Type Prediction Layer: Linear + Softmax over Types (`tag_logits`)
    - $\Pr(\text{TYPE}) = h_\text{tags} = W^e \cdot h_l$
5. Word Prediction Layer: Linear Softmax over Vocab
    - $h^* = W^p \cdot h_t$
6. Word Representation Layer: Linear Projection + Softmax of $h^*$ over each type (`word_distribution_logits`)
    - $\tilde{y} = \Pr(\text{word|TYPE}) = h^* \cdot h_{\text{tag} \in \text{tags}}$
7. Output: Tag Logits ($h_\text{tags}$) and Word Distribution Logits ($\tilde{y}$)

***TL;DR***:
- $W^p$ maps hidden state to words/entities
- $W^e$ maps hidden states to types

### Type Representation as Input
Adding type embedding representation of the previous word as input. A weighted sum of the type embeddings. Allowing more precise entity type predictions and learn latent types more accurately.

$$
\begin{equation}
\begin{aligned}
\nu_{t+1} &= \sum_{j=0}^K \Pr(\tau_{t+1}=j | c_t) \cdot W_{l,:}^e \\
\tilde{y}_{t+1} &= [y_{t+1};\nu_{t+1}]
\end{aligned}
\end{equation}\tag{4}\label{4}
$$

#### Pseudocode
1. Embedding Layer (+ Dropout)
2. Generate hidden state from LSTM
    - $h_t = \text{LSTM}(h_{t-1}, \text{word})$
3. Linear Projection of $h_t$ to Lower Dim
    - $h_l = W^h \cdot h_t$
4. Type Prediction Layer: Linear + Softmax over Types (`tag_logits`)
    - $\Pr(\text{TYPE}) = h_\text{tags} = W^e \cdot h_l$
5. Word Prediction Layer: Linear Softmax over Vocab
    - $h^* = W^p \cdot h_t$
6. Word Representation Layer: Linear Projection + Softmax of $h^*$ over each type
    - $\Pr(\text{word|TYPE}) = h^* \cdot h_{\text{tag} \in \text{tags}}$
6. Type Representation 
    - $\nu_{t+1} = \sum_{j=0}^K e_j$
    - $e_j = W^e \cdot h_o$
7. Output Layer: Tag Logits ($h_\text{tags}$) and Word Distribution Logits + Type Representation $\tilde{y}_{t+1} = [y_{t+1};\nu_{t+1}]$

### Implementation Details
Extended the AWD-LSTM ([Merity et al., 2017](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)).

- Vocabulary $V = {v_1,...,v_K}$
    - Identical words with different types have same input embeddings
- Model
    - Embedding size: $400$
    - Cell/hidden state size: $1,150$
    - LSTM layers: $3$
    - Entity type size: $100$
    - LSTM weight Dropout: $0$
- Optimization
    - Average SGD: $10$
    - Weight Decay: $1.2 \times 10^{-6}$
    - Gradient Clip: $0.25$

The type distributions (`type_logits`) learnt from KALM is latent, and can be used as the output for detemining the entity type. In terms of NER, we could use a bi-LSTM and concat the forward/backward vectors.

Due to the model predicting the entity type before predicting the word. It is unsufficient for even a bi-LSTM to decode the type of the word from the context of previous or following words ($c_l$ or $c_r$). Prior type information $\Pr(\tau_{t}|y_t)$ is used to calculate the entity population of the KBs. Two methods are proposed to deal with this matter:

#### Decoding with Type Priors
$$
\begin{equation}
\begin{aligned}
\Pr(\tau_t|c_l, c_r, y_t) &= \alpha \Pr(\tau_t|c_l, c_r) + \beta \Pr(\tau_t|y_t)
\end{aligned}
\end{equation}
$$

Learns coefficients $\alpha$ and $\beta$ from linear transformations of the predicted type $\tau_t$ given context and word.

#### Training with Type Priors
$$
\begin{equation}
\begin{aligned}
L = \text{CE}(\alpha \Pr(y_i|c_l, c_r), \Pr(\hat{y}_i|c_l, c_r)) + \lambda \cdot ||\text{KL}(\Pr(\tau_i|y_i))||^2
\end{aligned}
\end{equation}
$$

The objective is to learn type distributions $\Pr(y_i|c_l, c_r)$ to be close to the expected distribution $\Pr(\tau_i|y_i)$.

## Evaluation
The evaluation was done on Language Modeling and Unsupervised NER.

### Language Modeling
Language Modeling task was done one the Recipe dataset and CoNLL 2003 dataset.
![](https://i.imgur.com/RVTG01g.png)

### Unsupervised NER
The NER task was compared with some baseline supervised NER models.
![](https://i.imgur.com/9iP81GE.png)
