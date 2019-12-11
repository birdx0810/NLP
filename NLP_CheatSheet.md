# Natural Language Processing
###### tags: `NLP`
> "You shall know a word by the company it keeps" -- Firth (1957)

## Concepts
![](https://i.imgur.com/iGMDwwC.png)
- The Imitation Game (Alan Turing, 1950)
- The Chinese Room Argument (John Searle, 1980)

## Conferences
[AAAI]()
[ICLR]()
[PAKDD]()
[ACL](https://www.aclweb.org/anthology/)
[IJCAI]()
[KDD]()
[COLING]()
[CIKM]()
[NIPS]()
[ACM]()
[TREC]()

## NLP Levels

- Speech (Phonetics & Phonological Analysis)
- Text (Optical Character Recognition & Tokenization)
- Morphology (Def: the study of words, how they are formed, and their relationship to other words in the same language)
- Syntax & Semantics
- Discourse Processing

## NLP Goal

- Text Mining
  - POS Tagging (Dependency Parsing)
  - NER Tagging
  - Text Classification
  - Semantic Analysis
  - Similarity Prediction
  - Word Ranking
  - Natural Language Inference
  - Abstract Meaning Representation

- Tokenization
  - BPE
  - WordPiece

- Word Representations (Embeddings)
  - word2vec
  - GloVe
  - Seq2Seq
  - FastText
  - ELMo
  - Attention
    - BERT
    - RoBERTa
    - BERT-WWM
    - NEZHA
    - ALBERT
    - T5
  - ERNIE
    - THUxHUAWEI
    - Baidu

- Next Word/Sentence Prediction (Language Models)
  - n-gram model
  - RNN
    - LSTM
    - GRU
  - GAN

- Topic Classification
  - Grover (Fake News Detection)
  - Stance Detection
  - Spam Detection

- Machine Translation
  - Neural Machine Translation
  - Transformers

- Summarization
  - Extractive: Generates the summarization through extracting important information of the original document
  - Abstractive: Generates summarization after understanding the semantic meaning of document

- Chatbots
  - Question Answering
    - Open Domain: Deals with questions about nearly anything
    - Closed Domain: Deals with questions under a specific domain
  - Conversational Agents
    - Rule-based
    - Retrieval-based
    - Generative

## Modules (API)

nltk
gensim
spacy
stanfordnlp
google-cloud-language

### Chinese

[ckiptagger](https://github.com/ckiplab/ckiptagger)
[thulac](https://github.com/thunlp/THULAC-Python)
[snownlp](https://github.com/isnowfy/snownlp)
[jieba](https://github.com/fxsjy/jieba)
[opencc](https://github.com/BYVoid/OpenCC)

## Datasets

### Natural Language Inference
[SNLI](https://nlp.stanford.edu/projects/snli/)

### Social Networks
- Twitter
  - Official API: [Documentation](https://developer.twitter.com/en/docs)
  - 3rd Party Python Wrappers:
    - [Tweepy](https://tweepy.readthedocs.io/en/latest/)
    - [Python-Twitter](https://github.com/bear/python-twitter)
- Reddit
  - Official API: [Documentation](https://www.reddit.com/dev/api)
  - 3rd Party Python Wrappers:
    - [PRAW](https://praw.readthedocs.io/en/latest/)

### Chatbots
[Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator): Unstructured Multi-turn Dialogue System
[bAbI](https://research.fb.com/downloads/babi/): A set of prerequisite toy tasks

### Summarization
[Multi-News](https://github.com/Alex-Fabbri/Multi-News): Large-Scale Multi-Document Summarization Dataset
[CNN-DailyMail](https://github.com/abisee/cnn-dailymail): Dataset for Pointer-Generator by Abigail See
[GigaWord](https://drive.google.com/open?id=1eNUzf015MhbjOZBpRQOfEqjdPwNz9iiS): Dataset for Pointer-Generator by Abigail See

## Evaluation & Benchmarks

### BLEU (Scoring)

- Bilingual Evaluation Understudy
- A method for Automatic Evaluation of **Machine Translation**
- Range from 0.0 (perfect mismatch) to 1.0 (perfect match)
- Weighted cumulative n-grams
  - BLEU-1 (1,0,0,0)
  - BLEU-2 (0.5, 0.5, 0, 0)
  - BLEU-3 (0.33, 0.33, 0.33, 0)
  - BLEU-4 (0.25, 0.25, 0.25, 0.25)

### ROUGE

- Recall-Oriented Understudy for Gisting Evaluation
- Used for Machine Translation
- Measures:
  - ROUGE-N: Measures n-gram overlap
  - ROUGE-L: Measures longest matching sequence of words using longest common subsequence
  - ROUGE-S: Measures skip-gram coocurrence
- [Reference](https://rxnlp.com/how-rouge-works-for-evaluation-of-summarization-tasks)

### [SQuAD 2.0](rajpurkar.github.io)

- Stanford Question Answering Dataset
- Reading comprehension dataset where answer to every question is a span of text from the corresponding passage.
- 2.0 has 100,000 questions from SQuAD 1.1 with 50,000 unanswerable questions
- Problem: For each observation in the training set, we have a context, question and text
- Goal: Answer questions when possible, determine there is no answer when if none.
- Evaluation Metric: Exact Match score & F_1 Score
- [Reference](https://rajpurkar.github.io/mlx/qa-and-squad/)

### [GLUE](gluebenchmark.com)

- General Language Understanding Evaluation benchmark
- A collection of tasks for multitask evaluation for **natural language understanding**
- Introduced new benchmark [SuperGLUE](super.gluebenchmark.com)


視覺化
病人分層（好/不好）