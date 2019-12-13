# NLP Papers
###### tags: `NLP`

## Introduction

![](https://www.tensorflow.org/images/audio-image-text.png)

Word Vectors <=> Word Embeddings
- a set of language modeling techniques for mapping words to a vector of numbers (turns a text into numbers)
- a numeric vector represents a word
- more words == higher dimension

Key properties for Embeddings:
- Dimensionality Reduction: a more efficient representation
- Contextual Similarity: a more expressive representation

Syntax(syntactic): Grammatical structure
Semantics(Sentiment): Meaning of vocabulary

[Neural Language Modeling](https://ofir.io/Neural-Language-Modeling-From-Scratch/)
[Embed, Encode, Attend, Predict](https://explosion.ai/blog/deep-learning-formula-nlp)
[What can we cram into a vector](https://arxiv.org/abs/1805.01070)

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
- [Example Code](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text%20Normalization%20Demo.ipynb)

## Tokenizers

### Byte Pair Encoding (BPE)
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
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

# Output:
# ('e', 's')
# ('es', 't')
# ('est', '</w>')
# ('l', 'o')
# ('lo', 'w')
# ('w', 'est</w>')
# ('e', 'west</w>')
# ('n', 'ewest</w>')
# ('low', '</w>')
# ('d', 'est</w>')

```

### SentencePiece(WordPiece)
> Paper: [SentencePiece](https://arxiv.org/abs/1808.06226) & [WordPiece](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf)
> Github: [Code](https://github.com/google/sentencepiece)

- WordPiece tokenizes characters within words (BERT uses a variant of WP)
- SentencePiece tokenizes words and retaining whitespaces with a special token `_`

---

## Word Embeddings

### Word2Vec
input: corpus
framework: 2-layer (shallow) NN
[Example](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) (Tensorflow)
Features = the number of neurons in the projection(hidden) layer
Other implementations: [CBOW](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#CBOW), [Skip-gram](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#Skip-gram)

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

### GloVe
Global Vectors for Word Representations
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

## FastText

---
## Language Models

### n-gram: Probabilistic Language Model
> Bengio et. al (2003)
> Journal of Machine Learning Research
> Paper: [Link](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Derived from Markov Model
A sequence of N words
- unigram: "this" "that"
- bigram: "this word" "that boy"
- 3-gram: "this is tom" "he is gay"
- e.g. 3-gram model
>I would **not like them** here or there.
>I would **not like them** anywhere.
>I do not like green eggs and ham.
>I do **not like them**, Sam-I-Am.

$P(w|h)$
- Given context $h$, calculate the probability of the next word being $w$

$$
\begin{align*}
P(w_i|w_{i-1}w_{i-2}) &= P(them|not \ like) \\
&= \frac{C(w_{i-2}w_{i-1}w_{i})}{C(w_{i-2}w_{i-1})}\\
&= 3/4
\end{align*}
$$

Where $C(w_i)$ is the occurence count of word(or words) $w_i$ in corpus

Objective Function: Maximize Log-Likelihood

---

## RNN Based Language Model
> Paper
> [ACL](https://www.aclweb.org/anthology/N13-1090/)
> [INTERSPEECH](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)

![](https://i.imgur.com/8LBdgID.png)

$w(t)$ is the vector of word at time $t$ (one-hot)
$U$ is a word matrix where each column represents a word
$s(t-1)$ is the history input, passed from the last $s(t)$

$$\begin{align*}
w(t) &= v \times 1 \\
s(t) &= s(t-1) = d \times 1 \\
U &= d \times v \\
W &= d \times d \\
V &= V \times d \\
y(t) &= V \times 1 \\
\end{align*}
$$

---

## Sequence to Sequence Learning with Neural Networks
> Paper: [Link](https://arxiv.org/abs/1409.3215)
> GitHub: [Link](https://github.com/google/seq2seq)

Higly used in translations.
Source Language $\to$ **encode** $\to$ compressed state (vector) $\to$ **decode** $\to$ Target Language
$V_{src} \text{: \{I love apple\} } \to V_{tgt} \text{: \{我喜歡蘋果\} }$

![](https://i.imgur.com/2JsnR5F.png)

## Google's Neural Machine Translation System
> Paper: [Link](https://arxiv.org/pdf/1609.08144.pdf)
> GitHub:
> - [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)
> - [TensorFlow](https://github.com/tensorflow/nmt)

- Human translation: Translating part by part (memorization)
- Attention weight (α parameters) in each hidden state. How much information should we pay attention to in each word in vocab.
- Mapping a query and a set of key-value pairs to an output.
- Reduced sequence computation with parallelization.

![](https://i.imgur.com/IuaKvVQ.png)

The Problem with Machine Translation
1. 要不要翻譯 $\to V_t +1$
2. 翻譯哪個字(src) $\to$ Classifier
3. 翻譯成什麼(tgt) $\to$ Attention

$$
\begin{align*}
&a^{<t'>} = \Big( \overrightarrow{a}^{<t'>}, \overleftarrow{a}^{<t'>} \Big) \\ \\
&\sum_{t'} \alpha^{<1, t'>} = 1 \\ \\
&C^{<1>} = \sum_{t'} \alpha^{<1, t'>} a^{<t'>} \\ \\
\end{align*}
$$

$C$ is the context weighted attention sum.
$\alpha^{<t,t'>}$ is the amount of attention $y^{<t>}$ should pay to $a^{<t'>}$.

A larger the scalar (dot) product ($\approx 1$) means higher similarity. Thus, leads to "more attention".

Problem: Slow and still limited by size of context vector of RNN

## the Transformer model (Encoder-Decoder GAN)
Could we remove the continuous RNN states?
[Attention is all we need](https://arxiv.org/abs/1706.03762)

Consists of an Encoder and Decoder
- Encoder (self attention)
    1. Self attention layer
    2. Feed-forward network

![](https://i.imgur.com/M7DwjOl.png)

- Decoder
    1. Self attention layer
    2. Encoder-Decoder attention
    3. Feed-forward network
How does the model know the word sequence/order (without RNN)?
- Positional encoding
    - add a (one-hot) vector representing the sequence of each input word

![](https://i.imgur.com/WdSClBb.png)

Reference:
[Transformer - Heo Min-suk](https://www.youtube.com/watch?v=z1xs9jdZnuY)

### Scaled Dot-Product Attention
$\text{Attention}(Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}}) \times V$

- $QK^T$ is the attention score where $Q$ (m $\times$ 1)
- $d_k$ is keys of dimension
- The output is the attention layer output (vector)

### Multi-Head Attention
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

### Variational Autoencoder

---

## ELMo (Embedding from Language Models)
###### Pre-trained (2018)
Looks at entire sentence before assigning each word in it an embedding
Bi-directional LSTM

## BERT (Biderectional Encoder Representation from Transformers)
###### Pre-trained (2018)
Reads entire sequence at once (non-directional)
- Pre-train
    - Masked Language Model (MLM)
    - Next Sentence Prediction (NSP)
- Fine-Tuning

GitHub: [Code](https://github.com/google-research/bert)
Paper: [BERT](https://arxiv.org/abs/1810.04805)

## GPT-II
###### Pre-trained (2019)
GitHub: [Code](https://github.com/openai/gpt-2)
Paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

## TransformerXL
###### Pre-trained (2019)

GitHub: [Code](https://github.com/kimiyoung/transformer-xl)
Paper: [Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

## XLNet


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

---

## Glossary
#### Word Embedding:
- word/phrases are represented as vectors of real numbers in an n-dimensional space, where n is the number of words in the corpus(vocabulary)
- same goes to character embedding and sentence embedding
- character embedding works better for bigger models/languages with with morphology

#### Corpus:
- a collection of text documents

#### Counting:
- occurrences of tokens in each document

#### Normalizing (Pre-processing)
- [removing irrelevant noise](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#Text-Pre-processing) from the corpus

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
- **Inside**: token inside of chunk
- **Outside**: token outside of chunk
- **Beginning**: beginning of chunk
- **End**: end of chunk
- **Single**: represent a chunk containing a single token

[Reference](http://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf)

### Part-of-speech Tagging (POS)
parts of speech: noun, verb, pronoun, preposition, adverb, conjunction, participle, and article
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

### Language Model
- A machine/deep learning model that learns to predict the probability of the next (or sequence) of words.
    - Statistical Language Models
        - N-grams, Hidden Markov Models (HMM)
    - Neural Language Models

### Subword Modeling
- Turn words into subwords. E.g. subwords $\to$ sub, words
![](https://i.imgur.com/udjUH6F.png =360x)