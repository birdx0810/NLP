# NLP Papers
###### tags: `NLP`

## Introduction

![](https://raw.githubusercontent.com/thunlp/PLMpapers/master/PLMfamily.jpg)

Word Vectors <=> Word Embeddings
- a set of language modeling techniques for mapping words to a vector of numbers (turns a text into numbers)
- a numeric vector represents a word
- comparitively sparse: more words == higher dimension

Key properties for Embeddings:
- Dimensionality Reduction: a more efficient representation
- Contextual Similarity: a more expressive representation
  - Syntax(syntactic): Grammatical structure
  - Semantics(Sentiment): Meaning of vocabulary

<!-- TODO: Read
[Neural Language Modeling](https://ofir.io/Neural-Language-Modeling-From-Scratch/)  
[Embed, Encode, Attend, Predict](https://explosion.ai/blog/deep-learning-formula-nlp)  
[What can we cram into a vector](https://arxiv.org/abs/1805.01070)  
 -->
 
---

## Text Pre-processing
- removing tags (HTML, XML)
- removing accented characters (é)
- expanding contractions (don't, i'd)
- removing special characters (!@#$%^&\*)
- stemming and lemmatization
    - remove affixes
    - root word/stem
- removing stopwords (a, an, the, and)
- remove whitespace, lowercasing, spelling/grammar corrections etc.
- replace special tokens (digits to `[NUM]` token)
- [Example Code](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text%20Normalization%20Demo.ipynb)

## Text Mining

## Tokenizers

### Byte Pair Encoding (BPE)
> Neural Machine Translation of Rare Words with Subword Units
> Sennrich et. al (2015)
> Paper: [Link](https://arxiv.org/abs/1508.07909)
> Code: [Link](https://github.com/rsennrich/subword-nmt)

- replaces the most frequent pair of characters in a sequence with a single (unused) character ngrams
- add frequent n-gram character pairs in to vocabulary (something like association rule)
- stop when vocabulary size has reached target
- do deterministic longest piece segmentation of words
- segmentation is only within words identified by some prior tokenizer
- Variants:
    - WordPiece/SentencePiece (Google)

Example Code:
```python
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Dictionary {'word': # of occurence}
vocab = {'l o w </w>':5, 'l o w e r </w>':2,
         'n e w e s t </w>':6, 'w i d e s t </w>':3}

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
```

### SentencePiece(WordPiece)
> Affiliation: Google
> Paper: [SentencePiece](https://arxiv.org/abs/1808.06226) & [WordPiece](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf)
> Code: [Code](https://github.com/google/sentencepiece)

- WordPiece tokenizes characters within words (BERT uses a variant of WP)
- SentencePiece tokenizes words and retaining whitespaces with a special token `_`

### Byte-to-Span
> Paper: [Link](https://arxiv.org/abs/1512.00103)
> Code: [Link]

---

## Word Embeddings

### Word2Vec: 
> Mikolov et. al (2013)
> Affiliates: Google
> Paper:
> - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781): CBOW & SkipGram
> - [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546): Hierarchical Softmax & Negative Sampling 
> Code: 
> - [GitHub](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) (Tensorflow)
> - [Original Project](https://code.google.com/archive/p/word2vec/)

Input: corpus
Framework: 2-layer (shallow) NN
Features = the number of neurons in the projection(hidden) layer
<!-- Other implementations: [CBOW](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#CBOW), [Skip-gram](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#Skip-gram) -->

[Word Embedding Visual Inspector](https://ronxin.github.io/wevi/)
[Word2Vec](https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)

![](https://i.imgur.com/veeC83b.png)

#### CBOW
Goal: Predict center word based on context words
![](https://i.imgur.com/0n4bJpZ.png)

$Q = (N \times D) + (D \times log_2(V))$
$Q$ is the model complexity
$(N \times D)$ is the complexity of the hidden layer
$(D \times log_2(V))$ is the output layer

Better syntactic performance

#### Skip-gram
Goal: Predict context words based on center word
![](https://i.imgur.com/9rbBGmp.png)
The input would be a one-hot vector of the center word. Since it is predicting the possibility of the next word, the output vector would be a probability distribution (Problem: High dimension output)
![](https://i.imgur.com/FWCkspb.png)

$Q = C \times D + (D \times log_2(V))$
$Q$ is the model complexity
$C$ is the size of the context window
$D$ is the complexity of the hidden layer
$(D \times log_2(V))$ is the output layer

Better overall performance

#### Hierarchical Softmax
- A Huffman binary tree, where the root node is the hidden layer activations (or context vector $C$ ), and the leaves are the probabilities of each word.
- The goal of this method in word2vec is to reduce the amount of updates required for each learning step in the word vector learning algorithm.
- Rather than update all words in the vocabulary during training, only $log(W)$ words are updated (where $W$ is the size of the vocabulary).
$$
\displaystyle p(w|w_I) = \prod_{j=1}^{L(w)-1}\sigma\bigg(\big[n(w,j+1) = ch(n(w,j))\big] \cdot v'_{n(w,j)} v_{w_I}^{\intercal} \bigg)
$$

![](https://i.imgur.com/JlTkpC1.png)

#### Negative Sampling
- Solves time complexity of Skip-gram.
- Sample part of vocabulary for negative values, rather than whole vocab. (for $k$ negative examples, we have 1 positive example)
- Determine is this word from context? or sampled randomly
- E.g. "the" "bird" might have high co-occurence values, but they might not mean anything useful.

$$
log\ \sigma(v'_{w_O} \top v_{w_I}) + \sum^{k}_{i=1} \mathbb{E}_{w_i \sim P_n(w)} \big[log\ \sigma (-v'_{w_i} \top v_{w_I} \big]
$$

$$
\text{Softmax} = p(t|c) = \frac{e^{\theta^{T}_{t} e_c}}{\sum^{10000}_{j=1} e^{\theta^{T}_{t} e_c}}
$$

Where $\theta_t$ is the target word, and $e_c$ is context word. If the target word is true has probabillity of 1. By reducing the context words from 10000 to $k$ we could reduce the model complexity and runtime.

$$
\begin{align*}
P(w_i) &= \bigg(\sqrt{\frac{z(w_i)}{0.001}}+1\bigg)⋅\frac{0.001}{z(wi)} \\
P(w_i) &= 1 - \sqrt{\frac{t}{f(w_i)}} \\
\end{align*}
$$

How to sample negative examples?
- according to empirical frequency
- $\frac{1}{|Vocabulary|}$
- $\frac {f(w_i)^{3/4}}{\sum^{10000}_{j=1}f(w_j)^{3/4}}$

### GloVe: Global Vectors for Word Representations
> Jeffrey Pennington, Richard Socher, Christopher Manning (2014) #ACL
> Affiliates: Stanford University
> Paper: [Link](https://nlp.stanford.edu/pubs/glove.pdf)
> Code: [Link](https://github.com/stanfordnlp/GloVe)
> Official Site: [Link](https://nlp.stanford.edu/projects/glove/)

: Capture global statistics directly through model

2 main models for learning word vectors:
- latent semantic analysis (Good use of statistics, bad anology)
    - e.g. TF-IDF, HAL, COALS
- local context window (Good anology, bad use of statistics)
    - e.g. CBOW, vLBL, PPMI

$X$ is a word-word co-occurence matrix

$X_{ij}$ is the number of times word $j$ occurs in the context of word $i$

$X_i = \sum_k X_{ik}$ is the number of times any word appears in the context of word $i$

$P_{ij} = P(i|j) = \frac{X_{ij}}{X_i}$ be the probability that the word $j$ appear in the context of word $i$

Error Function
$$
J = \sum^{V}_{i,j=1} f(X_{ij})(w^T_i \tilde{w}_j + b_i + \tilde{b_j} - logX_{ij})^2
$$

![](https://i.imgur.com/l2PN1lJ.png)

Although $i$ and $j$ is highly related (e.g. ice, steam), they might not frequently appear together $P_{ij}$. But, through observing neighbouring context words $k$, we could identify the similarity between them through $P_{ik}$ and $P_{ij}$. If $i$ and $j$ is similar, when $P_{ik}$ is small $P_{jk}$ would also be small, and vice versa. Thus, $\frac{P_{ik}}{P_{jk}} \approx 1$.

### fastText: Enriching Word Vectors with Subword Information
> Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov (2016) #TACL
> Affiliation: facebook
> Paper: [Link](https://arxiv.org/abs/1607.04606) & [A bag of tricks](https://arxiv.org/abs/1607.01759)
> Code: [Link](https://github.com/facebookresearch/fastText)

### ELMo: Embedding from Language Models
> Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer (2018) #NAACL
> Affiliation: AllenNLP
> Paper: [Link](https://arxiv.org/abs/1802.05365)
> Code: [Link](https://github.com/allenai/allennlp)
> Official Site: [Link](https://allennlp.org/elmo)

Looks at entire sentence before assigning each word in it an embedding
Bi-directional LSTM

### Google's Universal Sentence Encoder
> Paper: [USE](https://arxiv.org/abs/1803.11175) [MultiUSE](https://arxiv.org/abs/1907.04307)
> Code: [Link](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)
> Official Site: [Link](https://tfhub.dev/google/universal-sentence-encoder/4)

### Skip-Thought
> Paper: [Link](https://arxiv.org/abs/1506.06726)

---

## Language Models

### Class-Based $n$-gram Models of Natural Language
> n-gram Language Model (Statistical Language Model)
> Brown et al. (1992)
> Affiliates: IBM
> Paper: [Link](https://www.aclweb.org/anthology/J92-4003.pdf)

Language Models have been long used in:
- speech recognition (Bahl et al., 1983)
- machine translation (Brown et al, 1990)
- spelling correction (Mays et al, 1990)

Language models assign probabilities to sequence of words.
$P(w|h)$
- Given context $h$, calculate the probability of the next word being $w$
- An $n$-gram model is a model of the probability distribution of $n$-word (or character) sequences

We assume that production of English text can be characterized by a set of conditional probabilities:
$$
\begin{aligned}
P(w_1^k) &= P(w_1)P(w_2|w_1)...P(w_k|w_{1}^{k-1}) \\    
&\approx \prod_{i=1}^k P(w_i|w_{i-1})
\end{aligned}
$$
Where...
- $P(w_k|w_{1}^{k-1})$ is the conditional probability of predicted word $w_k$ given history $w_{1}^{k-1}$
- $w_{1}^{k-1}$ represents the string $w_1w_2...w_{k-1}$

A **trigram** model could be defined as:
$$
P(w_{1:n}) = \prod_{i=1}^k P(w_i|w_{i-2}^{i-1})
$$
Where...
- $w_{k-2}^{k-1}$ is the history taken into context (i.e. the two words before $w_k$)
- In practice, it's more common to use trigram models

Parameter estimation: sequential maximum likelihood estimation
$$
P(w_n|w_1^{n-1}) \approx \frac{C(w_1^{n-1}w_n)}{\sum_w C(w_1^{n-1}w)}
$$
Where...
- $C(w)$ is the occurrence of string $w$ in $t_1^T$
- Maximise $P(t_n^T|t_1^{n-1})$
- Could be thought of as the transition matrix of a Markov model from state $n-1$ to state $n$
- As $n$ increases, the model **accuracy** *increases*, but **reliability** of parameter estimate *decreases*


Besides predicting the probability of next word, we could also predict word classes (syntactic similarity)
$$
\begin{aligned}
P(w|c) &= \frac{C(w)}{C(c)} \\
P(c) &= \frac{C(c)}{V}     
\end{aligned}
$$

### A Neural Probabilistic Language Model
> Bengio et. al (2003) #JMLR  
> Affiliates: University of Montreal
> Paper: [Link](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Curse of Dimensionality: a word or sequence on which the model will be tested is likely to be different from all the word sequences seen during training

- Associate with each word in the vocabulary a distributed word feature vector
- Express the joint probability function of word sequences in terms of the feature word vectors of these words in the sequence
- Learn simultaneously the word feature vectors and the parameters of the probability function

$$
\hat{P}(W^T_1) = \prod_{t=1}^{T} \hat{P}(w_t|w_{1}^{t-1})
$$
Where...
- $w_t$ is the $t$-th word
- $w_i^j$ is the sequence $(w_i, w_{i+1},..., w_{j-1}, w_j)


Objective Function: Maximize Log-Likelihood

### RNN Based Language Model
> Mikolov et. al (2013) #ACL
> Affiliates: Microsoft
> Paper:  
> [ACL](https://www.aclweb.org/anthology/N13-1090/)  
> [INTERSPEECH](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)  

![](https://i.imgur.com/8LBdgID.png)

$w(t)$ is the vector of word at time $t$ (one-hot)
$U$ is a word matrix where each column represents a word
$s(t-1)$ is the history input, passed from the last $s(t)$

$$
\begin{align*}
w(t) &= v \times 1 \\
s(t) &= s(t-1) = d \times 1 \\
U &= d \times v \\
W &= d \times d \\
V &= V \times d \\
y(t) &= V \times 1 \\
\end{align*}
$$

### LSTM

### GRU

### ULMFiT: Universal Language Model Fine-tuning for Text Classification
> Affiliates: Fast.ai
> Paper: [Link](https://arxiv.org/abs/1801.06146)  
> Code: [Link](https://github.com/fastai/fastai/blob/master/examples/ULMFit.ipynb)  
> Official Site: [Link](http://nlp.fast.ai/ulmfit)  

### UNILM: Unified Language Model Pre-training for Natural Language Understanding and Generation
> Paper: [Link](https://arxiv.org/abs/1905.03197)  
> Code: [Link](https://github.com/microsoft/unilm)  

---

## Seq2Seq

### NMT & Seq2Seq Models: A tutorial...
> Paper: [Link](https://arxiv.org/abs/1703.01619)
> 
> Neural Machine Translation Code:
> - [Tensorflow](https://github.com/tensorflow/nmt)  
> - [Google](https://google.github.io/seq2seq/nmt/)  
> - [OpenNMT](http://opennmt.net/)  

### On the properties of Neural Machine Translation: Encoder-Decoder Approaches
> Cho et al. (2014)
> Affiliation: University of Montreal
> Paper: [Link](https://arxiv.org/pdf/1409.1259.pdf)

### Sequence to Sequence Learning with Neural Networks
> Affiliation: Google  
> Paper: [Link](https://arxiv.org/abs/1409.3215)  
> Code: [Link](https://github.com/google/seq2seq)  

Source Language $\to$ **encode** $\to$ compressed state (vector) $\to$ **decode** $\to$ Target Language
$V_{src} \text{: \{I love apple\} } \to V_{tgt} \text{: \{我喜歡蘋果\} }$

### Neural Machine Translation by Jointly Learning to Align and Translate
> A.k.a RNNencdec & RNNsearch  
> Paper: [Link](https://arxiv.org/abs/1409.0473)  

### Google's Neural Machine Translation System
> Affiliation: Google  
> Paper:  
> [Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)  
> [Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558)

- Human translation: Translating part by part (memorization)
- Attention weight (α parameters) in each hidden state. How much information should we pay attention to in each word in vocab.
- Mapping a query and a set of key-value pairs to an output.
- Reduced sequence computation with parallelization.

![](https://i.imgur.com/KmzSHjm.png)

The Problem with Machine Translation:
1. Do we need to translate $\to V_t +1$
2. Which word to translate (src) $\to$ Classifier
3. What should it be translated to (tgt) $\to$ Attention

$$
a^{<t'>} = \Big( \overrightarrow{a}^{<t'>}, \overleftarrow{a}^{<t'>} \Big) \\
\sum_{t'} \alpha^{<1, t'>} = 1 \\
C^{<1>} = \sum_{t'} \alpha^{<1, t'>} a^{<t'>} \\
$$

$C$ is the context weighted attention sum.
$\alpha^{<t,t'>}$ is the amount of attention $y^{<t>}$ should pay to $a^{<t'>}$.

A larger the scalar (dot) product ($\approx 1$) means higher similarity. Thus, leads to "more attention".

Problem: Slow and still limited by size of context vector of RNN; could we remove the continuous RNN states?

### Attention is all you need
> a.k.a Transformers  
> Vaswani et al. (2017)  
> Affiliates: Google  
> Paper: [Link](https://arxiv.org/abs/1706.03762)  
> Code: [Havard: The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)  

Loss Function: Cross Entropy

![](https://i.imgur.com/w5XmS2C.png)

Consists of an Encoder and Decoder
- Encoder (self attention)
    1. Self attention layer
    2. Feed-forward network

- Decoder
    1. Self attention layer
    2. Encoder-Decoder attention
    3. Feed-forward network

How does the model know the word sequence/order (without RNN)?
- Positional encoding
    - add a (one-hot) vector representing the sequence of each input word

Reference:
[Transformer - Heo Min-suk](https://www.youtube.com/watch?v=z1xs9jdZnuY)

#### Scaled Dot-Product Attention
$\text{Attention}(Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}}) \times V$

- $QK^T$ is the attention score where $Q$ (m $\times$ 1)
- $d_k$ is keys of dimension
- The output is the attention layer output (vector)

#### Multi-Head Attention
- Parallelization (8 attention layers)
$$
\begin{align*}
Multihead(Q,K,V) &= \text{concat}(head_1, ...,head_h)W^O \\
\text{where } head_i &= \text{Attention}(QW^Q_i,KW^K_i,VW^V_i)
\end{align*}
$$
References:

#### Encoder Layer (6 identical layers with different weights)
- Input: Word embeddings $\to$ Positional encoding $\to$ Attention Layer
- Concat Attention outputs * Weight Matrix -> FC Layer
- Output: Word embedding (same size as input)
* Residual Connection followed by normalization
    * for retaining the information in the positional encodings

#### Decoder Layer (6 identical layers)
**Masked MH Att. -> MH Att. -> FC Layer**
- Input word vectors one at a time
- Generates a word from input word
* Masked layer prevents future words to be part of the attention
* Second MH Att. layer $q$ are from the decoder and $k$, $v$ the are from encoder

#### Final Linear Layer & Softmax Layer
Linear: Generate logit
Softmax: Probability of word
Label smoothing (regularization for noisy labels)

---

## Transformers
Code: [Huggingface](https://github.com/huggingface/transformers)

### BERT (Biderectional Encoder Representation from Transformers)
###### Encoder
> Devlin et. al (2018) #NAACL
> Code: [Link](https://github.com/google-research/bert)
> Paper: [Link](https://arxiv.org/abs/1810.04805)

Reads entire sequence at once (non-directional)

$$
\text{Input: [CLS] } s_1 \text{ [SEP] } s_2 \text{ [SEP] }
$$

- Objective functions (Optional)
    - Masked Language Model (MLM): Cross Entropy
    - Next Sentence Prediction (NSP)
- Pre-train
- Fine-tune
  - GLUE: [CLS] for prediction
  - SQuAD: [CLS] Document [SEP] Question [SEP]
  - CNN-DM: BERT(Word Representation) + Downstream

#### Dataset
Wikipedia
BookCorpus

### SpanBERT
###### Encoder
> Joshi et al. (2019 July 24)
> Affiliation: facebook
> Paper: [Link](https://arxiv.org/abs/1907.10529)
> Code: [Link](https://github.com/facebookresearch/SpanBERT)

Span Masking (WWM)
Span Boundary Objective (SBO): Predict current words using front and back words
~~NSP~~

### RoBERTa
###### Encoder
> Liu et al. (2019 July 26)
> Paper: [Link](https://arxiv.org/abs/1907.11692)
> Code: [Link]

Dataset Size: 40GB
Batch Size: 2048, 4096
Dynamic Masking (BERT always mask same > overfit)
No NSP
Add sentences until fit sequence length 512


### ALBERT
###### Encoder
> Lan et al. (2019 Sept)
> Paper: [Link](https://arxiv.org/abs/1909.11942)
> Code: 

Weight Sharing
Sentence Order Prediction (SOP)

### GPT: Improving Language Understanding by Generative Pre-training
###### Decoder
> Paper: [Link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
> Code: [Link](https://github.com/openai/finetune-transformer-lm)

### GPT-II: Language Models are Unsupervised Multitask Learners
###### Decoder
> Paper: [Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
> Code: [Link](https://github.com/openai/gpt-2)

### TransformerXL: Attentive Language Models Beyond a Fixed-Length Context
###### AutoEncoder
> Paper: [Link](https://arxiv.org/abs/1901.02860)
> Code: [Link](https://github.com/kimiyoung/transformer-xl)

### XLNet: Generalized Autoregressive Pretraining for Language Understanding
###### AutoEncoder
> Paper: [Link](https://arxiv.org/abs/1906.08237)
> Code: [Link](https://github.com/zihangdai/xlnet)

### T5
###### AutoEncoders
> Paper: [Link](https://arxiv.org/abs/1910.10683)
> Code: [Link](https://github.com/google-research/text-to-text-transfer-transformer)

---

## Graph Embeddings

### TransE
> Paper: [Link](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)

### TransW
> Paper: [Link](https://arxiv.org/abs/1909.03794)

## Metrics and Evaluations
### Distance Metrics for Word Similarity
- symbol equivalency or semantic equality

### Objective Based Metrics
- Bilingual Evaluation Understudy Score (BLEU)
    - a score for comparing a candidate translation of text to one or more reference translations
- Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
    - used for evaluating automatic summarization and machine translation software
- General Language Understanding Evaluation (GLUE)
    - Single-sentence tasks
    - Similarity & paraphrase
    - Inference tasks
    - Variants: SuperGLUE

## Glossary
#### Word Embedding:
- word/phrases are represented as vectors of real numbers in an n-dimensional space, where n is the number of words in the corpus(vocabulary)
- same goes to character embedding and sentence embedding
- character embedding works better for bigger models/languages with with morphology

#### Corpus (p. Corpora):
- a collection of text documents

#### Fine-tuning (Transfer Learning):
- tune the weights of a pretrained model by continuing back-propagation
- general rule of thumb:
  - truncate last softmax layer (replace output layer that is more relevant to our problem)
  - use smaller learning rate (pre-trained weights tend to be better)
  - freeze first few layers (tend to capture universal features)
- [CS231n on Transfer Learning](https://cs231n.github.io/transfer-learning/)

#### Language Model
- A machine/deep learning model that learns to predict the probability of the next (or sequence) of words.
    - Statistical Language Models
        - N-grams, Hidden Markov Models (HMM)
    - Neural Language Models

#### Morphology
- The study of formation of words
- Prefix/Suffix
- Lemmatization/Stemming
- Spelling Check

#### Phonology
- The study of sounds in language

#### Pragmatics
- The study of idiomatic phrases

#### Normalizing (Pre-processing)
- [removing irrelevant noise](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#Text-Pre-processing) from the corpus

#### Pre-train:
- **To pre-train** is to train a model from scratch using a large dataset 
- A pre-trained model is a model that has been trained (e.g. Pre-trained BERT)

#### Stop Words:
- commonly used words, such as, 'the', 'a', 'this', 'in' etc.

#### Smoothing:
- prevents computational errors in n-gram models
- e.g. $n$ is the number of times a word appears in a corpus. The importance of the word is denoted by $\frac{1}{n}$, it the word doesn't appear, it would be a mathematical error. Hence, smoothing techniques are used to solve this problem.

#### Tokenizing:
- a.k.a lexical analysis, lexing
- separating sentences into words (or words to characters) and giving an integer id for each possible token

#### Vocabulary:
- unique words within learning corpus

## Appendix
### Word Mover's Distance
The distance between two text documents A and B is calculated by the minimum cumulative distance that words from the text document A needs to travel to match exactly the point cloud of text document B.

### Named Entity Relation (NER)
- Classifying named entities mentioned in unstructured text into pre-defined categories
	- Because mapping the whole vocabulary is too time consuming
	- Stress on certain keywords/entities
	- Extract boundaries of words
	- E.g. chemical, protein, drug, gene etc.
	- E.g. person, location, event etc.
- recoginze the entity the corpus needs
- E.g. extract **chemical** in biomedical corpus -> **chemical** is regarded as an entity

### IOB Tagging
- Usually used in NER for identifying words within entity phrase
- Tags:
  - **Inside**: token inside of chunk
  - **Outside**: token outside of chunk
  - **Beginning**: beginning of chunk
  - **End**: end of chunk
  - **Single**: represent a chunk containing a single token

[Reference](http://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf)

### Part-of-speech Tagging (POS)
- Parts of speech: noun, verb, pronoun, preposition, adverb, conjunction, participle, and article
|Symbol| Meaning     | Example   |
|------|-------------|-----------|
| S    | sentence    | the man walked |
| NP   | noun phrase | a dog |
| VP   | verb phrase | saw a park |
| PP   | prepositional phrase |	with a telescope |
| Det  | determiner  | the |
| N    | noun        | dog  |
| V    | verb        | walked |
| P    | preposition | in |

### Dependency Tree
Dependency tree parses two words in a sentence by dependency arc to express their syntactic(grammatical) relationship

![](https://www.nltk.org/images/depgraph0.png)

### Subword Modeling
- Turn words into subwords. E.g. subwords $\to$ sub, words
- Used in Language Models (e.g. fastText) and Tokenization (e.g. Byte-Pair Encoding)
![](https://i.imgur.com/udjUH6F.png =360x)
