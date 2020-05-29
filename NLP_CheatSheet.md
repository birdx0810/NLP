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

## NLP Goal
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
- Abstract Meaning Representation
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
    - ERNIE
      - THUxHUAWEI
      - Baidu
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
  - Seq2seq
  - Neural Machine Translation
  - Transformers
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
    - Generative

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
8. Predict/Decode

[Text Classification Flow Chart](https://developers.google.com/machine-learning/guides/text-classification/images/TextClassificationFlowchart.png) by Google Developers
[Language Processing Pipeline](https://spacy.io/usage/processing-pipelines)

## Modules (API)
- nltk
- gensim
- spacy
- stanfordnlp
- allennlp
- google-cloud-language
- nlp-architect
- flair

### Chinese
- [ckiptagger](https://github.com/ckiplab/ckiptagger)
- [thulac](https://github.com/thunlp/THULAC-Python)
- [snownlp](https://github.com/isnowfy/snownlp)
- [jieba](https://github.com/fxsjy/jieba)
- [opencc](https://github.com/BYVoid/OpenCC) (Translator)

## Datasets

### bABI
- [bAbI](https://research.fb.com/downloads/babi/)(ICLR 2015): A set of prerequisite toy tasks for NLP

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

### Conversational Models
#### English
- A Survey of Available Corpora for Building Data-Driven Dialogue Systems]([Serban et al., 2015](https://breakend.github.io/DialogDatasets/))
- Multi-relation Question Answering Dataset (COLING 2018) [Download](http://coai.cs.tsinghua.edu.cn/hml/media/files/PathQuestion.zip)
- Commonsense Conversation (IJCAI 2018) [Download](http://coai.cs.tsinghua.edu.cn/file/commonsense_conversation_dataset.tar.gz)
- [DailyDialog](http://yanran.li/dailydialog)(2017): Human-written and less noisy that reflects daily communication of humans
- [Twitter](https://www.kaggle.com/thoughtvector/customer-support-on-twitter)(Kaggle 2017): A large corpus of modern English (mostly) conversations between consumers and customer support agents on Twitter
- [OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles-v2016.php)(2016): A new collection of translated movie subtitles from http://www.opensubtitles.org/
- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)(SIGDIAL 2015): Unstructured Multi-turn Dialogue System
- [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)(2011): Contains a large collection of fictional conversations from raw movie scripts

#### Chinese
- CrossWOZ (TACL 2020): A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset [Download](https://coai.cs.tsinghua.edu.cn/file/CrossWOZ_data.zip)
- [KdConv](https://github.com/thu-coai/KdConv)(ACM 2020)
- Grayscale Dataset for Dialogue (2020) [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/grayscale_data_release.zip)
- Retrieval Generation Chat (EMNLP 2019) [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Retrieval_Generation_Chat.zip)
- Restoration-200K Datasets (EMNLP 2019) [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Restoration-200K.zip)
- Chinese Dialogue Sentence Function Dataset (ACL 2019) [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/dialog-acts.tar.gz)
- Weibo Conversation Dataset (AAAI 2019) [Download](https://ai.tencent.com/ailab/nlp/dialogue/datasets/weibo_utf8.zip)
- Dialogue Question Generation Dataset (ACL 2018) [Download](http://coai.cs.tsinghua.edu.cn/file/QGdata.zip)
- [Douban Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection)(ACL 2017)

### Summarization
- [Multi-News](https://github.com/Alex-Fabbri/Multi-News): Large-Scale Multi-Document Summarization Dataset
- [CNN-DailyMail](https://github.com/abisee/cnn-dailymail): Dataset for Pointer-Generator by Abigail See
- [GigaWord](https://drive.google.com/open?id=1eNUzf015MhbjOZBpRQOfEqjdPwNz9iiS): Dataset for Pointer-Generator by Abigail See

## Evaluation Metrics

Accuracy
Precision
Recall
F1-score

### Perplexity
- Perplexity evaluates the probability distribution (language model) over entire sentences or text (lower is better)

$$
\begin{align}
\text{PPL}(W) &= P(w_1 w_2 ... w_N)^{-\frac{1}{N}}
&= \sqrt[N]{\frac{1}{\Pr(w_1 w_2 ... w_N)}}
\end{align}
$$

### [BLEU](https://www.aclweb.org/anthology/P02-1040/)

- Bilingual Evaluation Understudy
- A method for Automatic Evaluation of **Machine Translation**
- Range from 0.0 (perfect mismatch) to 1.0 (perfect match)
- Weighted cumulative n-grams
  - BLEU-1 (1,0,0,0)
  - BLEU-2 (0.5, 0.5, 0, 0)
  - BLEU-3 (0.33, 0.33, 0.33, 0)
  - BLEU-4 (0.25, 0.25, 0.25, 0.25)

### [ROUGE](https://www.aclweb.org/anthology/W04-1013/)

- Recall-Oriented Understudy for Gisting Evaluation
- Used for Machine Translation
- Measures:
  - ROUGE-N: Measures n-gram overlap
  - ROUGE-L: Measures longest matching sequence of words using longest common subsequence
  - ROUGE-S: Measures skip-gram coocurrence
- [Reference](https://rxnlp.com/how-rouge-works-for-evaluation-of-summarization-tasks)

### [METEOR](https://www.aclweb.org/anthology/W05-0909/)

## Leaderboards & Benchmarks

### [GLUE](gluebenchmark.com)
- General Language Understanding Evaluation benchmark
- A collection of tasks for multitask evaluation for **natural language understanding**
  - CoLA
  - SST
  - MRPC: Paraphrasing
  - STS
  - QQP: Sentence Similarity
  - NLI: 
    - ([MultiNLI](https://www.nyu.edu/projects/bowman/multinli/))
    - QNLI
    - WNLI
  - RTE
  - DM
- Introduced new benchmark [SuperGLUE](super.gluebenchmark.com)
- ChineseGLUE benchmark [CLUE](https://github.com/ChineseGLUE/ChineseGLUE)

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