# Memory Networks
###### tags: `MemNN`

> Weston et. al, 2014 
> Affiliation: Facebook
> Original Paper: [Link](https://arxiv.org/abs/1410.3916)

- Combines inference components with a **long-term memory** component
- Extending the neural network
- Memory embedding serves as a knowledge base

## Model Architecture

![](https://i.imgur.com/232TSy8.png)

- Memory $(m)$  - An array of vectors (embeddings) indexed by $m_i$
    - The memory saves the raw text for training and embedding for testing **as parameters will change during training**
- Input Feature Map $(I)$ - Converts input to internal feature representation
	- Pre-process input data and encode input into vector
    - Could be either a statement/fact, or a question to be answered
- Generalization $(G)$ - Updates old memories given new input
	- The network compresses and generalize its memories at this stage for future use
	- $G$ is **only used to store new memory** and old memories are not updated
	- Stores $I(x)$ in slot within the memory with index $H(x)$
	- $H(\cdot)$ is a trainable function for selecting the slot index of $m$ for $G$ to replace
	- $H$ could also be used to select which memory to replace
- $S(x)$ returns next empty memory slot $N: m_N = x, \ N=N+1$
- Output Feature Map $(O)$ - Produces new output from feature representation by finding **$k$ supporting memories** given $x$
	- Responsible for reading memory and performing inference (finding relevant memories)
	- $o_1 = O_1(x, m) = {argmax}_{i=1,...,N} s_O(x, m_i)$
	- $o_2 = O_2(x,m) = {argmax}_{i=1,...,N} s_O([x, m_{O_1}], m_i)$. Where $k=2$
	- $s_O$ is a function that scores the match between sentences from $x$ and $m_i$
		- $s_O (x,y) = x^\intercal U^\intercal_O \cdot U_O y$
	- Where $m_i$ is the candidate supporting memory being scored with respect to $x$ and $m_{O_1}$
- Response $(R)$ - Converts the output into response format
	- Produces final response $r$ given $o$
	- $R$ is used to produce actual wording of the answer
	- Could be a conditioned RNN (???
    - $R$ needs to produce a textual response $r$ from $o=[x, m_{O_1}, m_{O_2}]$
	- $r = argmax_{w \in W} s_R([x, m_{O_1}, m_{O_2}], w)$
	- Where $W$ is the set of all words in the dictionary and $s_R$ is a function that scores the match
	- $s_R$ is a function that scores the match between sentences from $x$ and $m_i$
		- $s_R (x,y) = x^\intercal U^\intercal_R \cdot U_R y$
    - The simplest form of the representation output is by returning $\mathbb{m}_{O_k}$. Could use an RNN as output decoder.
- The scoring functions $s_O$ and $s_R$ have the same form
    - $s(x,y) = \Phi_x(x)^\intercal U^\intercal U\Phi_y (y)$
    - $U$ is a $n \times D$ weight matrix of the embedding dimension $n$ and number of features $D = 3|W|$
    - Every word in the dictionary has 3 representations, two for $\Phi_x$ and one for $\Phi_y$, which are used to map the original text to the $D$-dimensional feature space

**Note**: 
- $I$, $G$, $O$ and $R$ components could be any existing machine learning models
- The core inference lies in $O$ and $R$

## Process
Given input $x$, the flow of the model is as follows:
1. Convert $x$ into an internal feature representation $I(x)$
2. Update memories in $m_i$ given the new input: 
	- $m_i = G(m_i, I(x), m), \forall i$
3. Compute output features $o$ given new input and the memory: 
	- $o = O(I(x), m)$ 
4. Decode output features $o$ to give final response:
	- $r = R(o)$

### Training Details
- Margin Ranking Loss & SGD

## Experiments
- Large-scale QA
- Simulated World QA
    - Considers: 
        1) Single Word Answers
        2) Simple Grammar Answers

## Conclusion
- Naive Architecture
    - Memory management 
        - FIFO writing
        - iterated over whole memory
        - reweight memory embeddings
    - Sentence representation
        - Single word as answer (no "decoder" mechanism)
- Not easy to train via backprop
- Requires supervision at each layer





