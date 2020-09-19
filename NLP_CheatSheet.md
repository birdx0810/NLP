# NLP Cheat Sheet
###### tags: `NLP`
> "You shall know a word by the company it keeps" -- Firth (1957)  
> ![](https://imgs.xkcd.com/comics/machine_learning.png)  
> "I deeply believe that we do need more structure and modularity for language, memory, knowledge, and planning; it’ll just take some time..." -- Manning (2017)  

## Concepts
<!-- ![](https://i.imgur.com/iGMDwwC.png) -->
- The Imitation Game (Alan Turing, 1950)
- The Chinese Room Argument (John Searle, 1980)

## Conferences
[AAAI](https://www.aaai.org/Conferences/conferences.php)  
[ICLR](https://iclr.cc)  
[PAKDD](https://pakdd.org)  
[ACL](https://www.aclweb.org/anthology/)  
[IJCAI](https://www.ijcai.org)  
[KDD](https://www.kdd.org)  
[COLING](https://www.aclweb.org/anthology/venues/colin)  
[CIKM](www.cikmconference.org)  
[NIPS](https://nips.cc)  
[ACM](https://www.acm.org)  
[TREC](https://trec.nist.gov)  
[ICML](https://icml.cc)  

## NLP Levels
![](https://www.tensorflow.org/images/audio-image-text.png)

- Speech (Phonetics & Phonological Analysis)
- Text (Optical Character Recognition & Tokenization)
- Morphology (Def: the study of words, how they are formed, and their relationship to other words in the same language)
- Syntax & Semantics
- Discourse Processing

## NLP Tasks/Goals
- Voice & Speech
  - Automated Speech Recognition
  - Text-To-Speech
- Text Mining
  - Morphology
    - Prefix/Suffix
    - Lemmatization/Stemming
    - Spell Check
  - Syntactic (Parsing)
    - POS Tagging
    - Syntax Trees
    - Dependency Parsing
  - Semantic
    - NER Tagging
    - Relational Extraction
    - Similarity Analysis
    - Word Ranking
- Tokenization
  - BPE
  - WordPiece
  - Byte-to-Span
- Word Representations (Embeddings)
  - word2vec
  - GloVe
  - FastText
  - ELMo
  - Attention
    - BERT
    - RoBERTa
    - SpanBERT/WWM
    - DistilBERT
    - ALBERT
    - T5
    - NEZHA
- Language Models
  - n-gram models
  - RNN models
    - LSTM
    - GRU
- Topical Classification
  - Spam Detection
  - Fake News Detection
  - Stance Detection
- Machine Translation
- Natural Language Inference
  - Shallow Approach: Based on lexical overlap, pattern matching, distributional similarity etc.
  - Deep Approach: semantic analysis, lexical and world knowledge, logical inference
- Summarization
  - Extractive: Generates the summarization through extracting important information of the original document
  - Abstractive: Generates summarization after understanding the semantic meaning of document
- Question Answering
    - Open Domain: Deals with questions about nearly anything
    - Closed Domain: Deals with questions under a specific domain
- Conversational Agents
    - Rule-based
    - Retrieval-based
    - Generative-based

## Introduction

Word Vectors <=> Word Embeddings <=> Word Representations
- a set of language modeling techniques for mapping words to a vector of numbers (turns a text into numbers)
- a numeric vector represents a word
- comparitively sparse: more words == higher dimension

Key properties for Embeddings:
- Dimensionality Reduction: a more efficient representation
- Contextual Similarity: a more expressive representation
  - Syntax(syntactic): Grammatical structure
  - Semantics(Sentiment): Meaning of vocabulary

## Process
0. Define goal
1. Crawl/Prepare Data
2. Pre-process text
3. Stemming, Lemmatize, Tokenize
4. Build dictionary
    - {ID: WORD/SENT}
5. Embed words to vectors
6. Encode using Biderectional RNNs
7. Attend to compress
8. Inference: (Predict/Decode)

[Text Classification Flow Chart](https://developers.google.com/machine-learning/guides/text-classification/images/TextClassificationFlowchart.png) by Google Developers
[Language Processing Pipeline](https://spacy.io/usage/processing-pipelines)

## Modules (API)
- nltk
- gensim
- spacy
- stanza
- allennlp
- google-cloud-language
- nlp-architect
- flair

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

## Decoding

### Likelihood-maximizing decoding

- Greedy search: Take the most probable word at each step and feed it as input into the next step until the `EOS` token is produced
  - Usually ungrammatical, unnatural, or nonsensical
- Beam search: On each step, keep track of the top-$k$ most probable sequences (hypothesis) and choose the sequence with the highest probability
  - Higher beam size ($k$) is more computationally expensive, and leads to more genric or less relevant responses

### Sampling-based decoding

- Pure sampling: Randomly sample the next word from the probability distribution $P$ of the whole vocabulary at time $t$
  - $k = v$
- Top-k sampling: Randomly sample from $P_t$ with restriction of the top-$k$ most probable words
  - $k = 1$ is greedy search
  - Large $k$ is more diverse/risky, smaller $k$ is more generic/safe
- Neucleus sampling: For a given threshold $p$, at each timestep, we sample from the most probable words whose cumulative probability comprises the top-$p$ of the entire vocabulary

## Evaluation Metrics
Goal: Assign higher probability to "real"/"frequent" sentences

- Extrinsic Methods
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Intrinsic Methods
  - Perplexity

### Perplexity
- Perplexity evaluates the probability distribution of a word normalized over entire sentences or text (language model)
- Is the weighted equivalent branching factor
- Minimizing perplexity is the same as maximizing probability (lower is better), could be also thought of as entropy. The corpus that we fit is deemed as the gold truth. Thus, we would try to minimize the perplexity(entropy) of the corpus.
- Bad approximation, only used in pilot test

$$
\begin{align}
\text{PPL}(W) &= P(w_1 w_2 ... w_N)^{-\frac{1}{N}} \\
&= \sqrt[N]{\frac{1}{\Pr(w_1 w_2 ... w_N)}} \\
&= \sqrt[N]{\prod_{i=1}^N\frac{1}{\Pr(w_i| w_1 ... w_{i-1})}}
\end{align}
$$

Where
- $i-1$ is the $n$-gram the model takes into context.

### [BLEU](https://www.aclweb.org/anthology/P02-1040/)

- Bilingual Evaluation Understudy
- A method for Automatic Evaluation of **Machine Translation**
- Precision-based
- Range from 0.0 (perfect mismatch) to 1.0 (perfect match)
- Weighted cumulative n-grams
  - BLEU-1 (1,0,0,0)
  - BLEU-2 (0.5, 0.5, 0, 0)
  - BLEU-3 (0.33, 0.33, 0.33, 0)
  - BLEU-4 (0.25, 0.25, 0.25, 0.25)

$$
\log \text{BLEU} = \min(1 = \frac{r}{c}, 0) + \sum_{n=1}^N w_n \log p_n
$$

Where,
- $N$ or $n$ is the number of $n$-grams (default = 4)
- $w_n$ is the positive weights (1/n = 0.25)
- $p_n$ is the $n$-gram precisions (modified precision)
- $c$ is the length of the candidate translation
- $r$ is the reference corpus length

$$
p_n = \frac{\sum_{S \in C} \sum_{\text{gram}_n \in S} \text{count}_\text{clip}(\text{gram}_n)}
{\sum_{S \in C} \sum_{\text{gram}_n \in S} \text{count}(\text{gram}_n)}
$$

Where,
- $\text{count}_\text{clip} = \min(\text{count}, \text{max_ref_count})$ is the maximum number of $n$-gram occuring on each reference

$Count_\text{clip} = \min(Count;MaxRefCount)$. In other words,one truncates each word’s count, if necessary, to not exceed the largest count observed in any single reference for that word.

Where $Count$ is the number of times the n-gram occurs in candidate sentence; and $Count_\text{clip}$ is the maximum number of n-grams occurrences in any reference sentence

### [ROUGE](https://www.aclweb.org/anthology/W04-1013/)

- Recall-Oriented Understudy for Gisting Evaluation
- A method for Automatic Evaluation of **Summarization**
- Recall-based
- Measures:
  - ROUGE-N: Measures n-gram overlap
  - ROUGE-L: Measures longest matching sequence of words using longest common subsequence
  - ROUGE-S: Measures skip-gram coocurrence
  - ROUGE-W: Weighted longest common subsequence

$$
\text{ROUGE}_N =
\frac{\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{count}_\text{match}(\text{gram}_n)}
{\sum_{S \in R} \sum_{\text{gram}_n \in S} \text{count}(\text{gram}_n)}
$$

Where,
- the numerator is the number of matched $n$-grams between the generated sentence and gold sentence
- the denominator is the total number of $n$-grams for each sentence
- $R$ is the set of referenced summaries (generated and gold)
- $S$ is the sentences within $R$
- $\text{count}_\text{match}$ is the maximum number of $n$-grams cooccuring in a candidate summary and reference summaries

## Leaderboards & Benchmarks

### [GLUE](gluebenchmark.com)
- General Language Understanding Evaluation benchmark
- A collection of tasks for multitask evaluation for **NLU**
  - CoLA
  - SST
  - MRPC: Paraphrasing
  - STS
  - QQP: Sentence Similarity
  - NLI: 
    - [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
    - QNLI
    - [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)
  - RTE
  - DM
- Introduced new benchmark [SuperGLUE](super.gluebenchmark.com)
- ChineseGLUE benchmark [CLUE](https://github.com/ChineseGLUE/ChineseGLUE)
- Bio-medical Language Understanding [BLUE](https://github.com/ncbi-nlp/BLUE_Benchmark)

### [SQuAD 2.0](rajpurkar.github.io)
- Stanford Question Answering Dataset
- Reading comprehension dataset where answer to every question is a span of text from the corresponding passage.
- 2.0 has 100,000 questions from SQuAD 1.1 with 50,000 unanswerable questions
- Problem: For each observation in the training set, we have a context, question and text
- Goal: Answer questions when possible, determine there is no answer when if none.
- Evaluation Metric: Exact Match score & F_1 Score
- [Reference](https://rajpurkar.github.io/mlx/qa-and-squad/)

### SNLI
[SNLI](https://nlp.stanford.edu/projects/snli/)

## Glossary

### Softmax Temperature
The temperature $\tau$ is used to alter the distribution weights of the Softmax layer.
- Larger $\tau$ leads to a more uniform $P_t$ and more diverse output
- Smaller $\tau$ leads to a more skewed $P_t$ and less diverse output
$$
P_t(w) = \frac{\exp(s_w/\tau)}{\sum_{w' \in V} \exp S_{w'}/\tau}
$$


<!-- ### [METEOR](https://www.aclweb.org/anthology/W05-0909/)

- Metric for Evaluation of Translation with Explicit ORdering
- The metric is based on the harmonic mean of unigram precision and recall
- Includes stemming and synonymy matching

### [NIST](https://dl.acm.org/doi/10.5555/1289189.1289273)

- NIST also calculates how informative a particular n-gram is

### [WER](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.89.424)

- Word Error Rate
- A common metric of the performance of a speech recognition or machine translation system
- WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level -->

#### Word Embeddings
- word/phrases are represented as vectors of real numbers in an n-dimensional space, where n is the number of words in the corpus(vocabulary)
- same goes to character embedding and sentence embedding
- character embedding works better for bigger models/languages with with morphology

#### Corpus (p. Corpora)
- a collection of text documents

#### Fine-tuning (Transfer Learning)
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

### Gradient Clipping

$$
g \gets \frac{\eta g}{||g||}
$$

![](https://i.imgur.com/fZcDDgr.png)
