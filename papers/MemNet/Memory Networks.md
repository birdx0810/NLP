# Memory Networks

> Weston et. al, 2014 #Facebook
>
> Original Paper: [Link](https://arxiv.org/abs/1410.3916)
>
> [GitHub](https://github.com/facebook/MemNN) (Written in Lua Torch7), Python3 Implementation: [Link](https://github.com/jojonki/MemN2N-babi-python/commit/a10f53da08aa4fe431c592ed54c9e18ead64318d)

- Combines inference components with a **long-term memory** component
- Memory embedding serves as a knowledge base

---

## Model Architecture

- Memory $(m)$  - An array of vectors (embeddings) indexed by $m_i$

- Input Feature Map $(I)$ - Converts input to internal feature representation
  - Pre-process input data and encode input into vector
- Generalization $(G)$ - Updates old memories given new input
  - Store $I(x)$ in a "slot" within the memory
  - $m_{H(x)} = I(x)$
  - $H(\cdot)$ is a trainable function for selecting the slot index of $m$ for $G$ to replace
- Output Feature Map $(O)$ - Produces new output from feature representation
  - Responsible for reading memory and performing inference
- Response $(R)$ - Converts the output into response format
  - Produces final response $r$ given $o$
- **Note**: Components could be any existing machine learning models

---

## Process

Given input $x$, the flow of the model is as follows:

1. Convert $x$ into an internal feature representation $I(x)$
2. Update memories in $m_i$ given the new input: 
   - $m_i = G(m_i, I(x), m), \forall i$
3. Compute output features $o$ given new input and the memory: 
   - $o = O(I(x), m)$ 
4. Decode output features $o$ to give final response:
   - $r = R(o)$

## Model

- $I$ module takes an input text 
- Raw text for training (parameters will change), embedding for testing
- $S(x)$ returns next empty memory slot $N: m_N = x, \ N=N+1$
- $G$ is thus only used to store new memory and old memories are not updated
- $O$ module produces output features by finding $k$ supporting memories given $x$
  - $o_1 = O_1(x, m) = {argmax}_{i=1,...,N} s_O(x, m_i)$
  - Where $s_O$ is a function that scores the match between sentences from $x$ and $m_i$
  - $o_2 = O_2(x,m) = {argmax}_{i=1,...,N} s_O([x, m_{O_1}], m_i)$. (Where $k=2$)
  - Where $m_i$ is the candidate supporting memory being scored with respect to $x$ and $m_{O_1}$
- $R$ needs to produce a textual response $r$ from $o=[x, m_{O_1}, m_{O_2}]$
  - $r = argmax_{w \in W} s_R([x, m_{O_1}, m_{O_2}], w)$
  - Where $W$ is the set of all words in the dictionary and $s_R$ is a function that scores the match