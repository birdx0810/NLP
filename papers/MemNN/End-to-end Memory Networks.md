# End-to-End Memory Networks
> Sukhbaatar et al., 2015 
> Affiliation: Facebook
> Paper: [Link](https://arxiv.org/abs/1503.08895)
> GitHub: [Original](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model) (written in Lua)

## Background
- Continuation of [Memory Networks](https://arxiv.org/abs/1410.3916) (Weston et. al, 2015).
- Motivation: 
    - not easy to train via backpropagation
    - required supervision at each layer

## TL;DR

[Memory Networks](https://arxiv.org/abs/1410.3916) + [RNNsearch](https://arxiv.org/abs/1409.0473)

## Approach
### Single Layer
![](https://i.imgur.com/LsfIt9X.png)

- Let,
    - $x_1,...,x_n$ are inputs to be stored in memory $M$ with $n$ fixed buffer size
    - $q$ and $a$ are an input query and output answer respectively
    - $x$, $q$ and $i$ contains symbols from dictionary with size $V$

- **Input Memory Representation**
    - $m_i$ is the sentence embedding
        - $m_i = A x_i^\intercal$ making it a $d \times 1$ vector
        - $A$ is an $d \times V$ matrix
        - $x_i$ is a $1 \times V$ vector
    - $u$ is the query embedding calculated via embedding matrix $B$ (similar to $m_i$)
    	- $u$ is a $d \times 1$
    - $p_i$ is the probability vector of all inputs $(i)$
        - $p_i$ is a probability weight (value of 0-1) for each sentence $i$

$$
p_i = \text{Softmax}(u^\intercal m_i)
$$

- **Output Memory Representation**
    - Each input $x_i$ has a corresponding output $c_i$
    - The formation of $c_i$ embeddings are also similar to $m_i$
    - $p_i$ is the calculated probability of vector over inputs $(i)$
    - $o$ is the weighted sum of $c_i$ with their respective probability

$$
o = \sum_i p_i c_i
$$

- **Generating Final Prediction**
    - Element-wise sum of input $u$ and output $o$ 
    - $W$ is a weighting matrix $V \times d$ matrix 
$$
\hat{a} = \text{Softmax}(W(o+u))
$$

**Objective**: Minimize cross-entropy loss between predicted output $\hat{a}$ and ground truth $a$.

**Hyperparameters**: 
- optimizer: Stochastic Gradient Descent

### Multiple layers
extends model to handle $K$ hop operations

Embedding weight tyings for multiple hops:
- **Adjacent**: $A^{k+1} = C^{k}$, and the final output embedding $W^T = C^K$, and question input embedding $B = A^1$
- **Layer-wise**: $A^1=A^2=...A^K$ and $C^1=C^2=...C^K$, and adding a linear mapping $u^{k+1} = Hu^k+o^k$ to update between layers

![](https://i.imgur.com/bwWOaIj.png)


