# -*- coding: utf-8 -*-
'''
This is a walkthrough that introduces different types of word embeddings:
- Count Vectorizer
- TF-IDF
- word2vec
- GloVe
- FastText
The code below shows the methods for obtaining the word vectors and vocabulary dictionary.
Please take note that the inputs are raw text from the corpus variable below. Thus, it has not been preprocessed/cleaned.
Author: birdx0810
'''

sentences = [
    'Outline is an open-source Virtual Private Network (VPN).',
    'Outline is a created by Jigsaw with the goal of allowing journalists to have safe access to information, communication and report the news.', 
    'It is well known for its ease of use and could be deployed and hosted by the average journalist.',
    'Jigsaw is an incubator within Alphabet that uses technology to address geopolitical issues and combat extremism.',
    'One of their other known projects is Digital Attack Map.'
]

#########################
# Count Vectorizer (One-hot encoding)
# Method: count the occurrence of each word in each document
#########################

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
count_vectors = cv.fit_transform(sentences)
vocabulary = cv.vocabulary_
vectors = count_vectors.toarray()

#########################
# TF-IDF Transformation/Vectorization
# Method: Term Frequency-Inverse Document Frequency, a statistical measure for calculating the relevance of a term within document.
# Note: 
# - This is a follow up approach on Count Vectorization, thus the input is from derived from above and not the raw corpus.
#########################

# Transformer Method
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_weight = transformer.fit(count_vectors)
tfidf_score = transformer.transform(count_vectors)
vectors = tfidf_score.toarray()

# Vectorizer Method
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(use_idf=True)
tfidf_score = vectorizer.fit_transform(sentences)
vectors = tfidf_score.toarray()

#########################
# word2vec
# Method: Common Bag-of-Words and Skip-Gram
# Note: 
# - sg = skipgram(1) or cbow(0, default)
# - hs = hierarchical softmax(1) or negative sampling(0, default)
# - w2v.train(more_sentences) to learn new sentences (online learning)
#########################

from gensim.models import word2vec

tokenized = [sentence.split() for sentence in sentences]
w2v = word2vec.Word2Vec(tokenized, min_count=1)
vocabulary = w2v.wv.vocab
vectors = w2v.wv.vectors

#########################
# GloVe
# Method: Capturing word embeddings through word-frequency and co-occurrence counts with a matrix.
#########################

from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(tokenized, window=10)
glove = Glove(no_components=512, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=5, no_threads=2)
vocabulary = corpus.dictionary
vectors = glove.word_vectors
print(vectors.shape)

#########################
# FastText
# Method: Takes into account the subword features and adds them to the final vector.
# Note: 
# - This is a follow-up approach on w2v. Thus, the `sg`, `hs` parameters are also applicable for the gensim version.
# - Another method used is the bag of tricks supervised method available using Facebook's original module.
#########################

# Using gensim
from gensim.models import fasttext

tokenized = [sentence.split() for sentence in sentences]
ft = fasttext.FastText(tokenized, min_count=1)
vocabulary = ft.wv.vocab
vectors = ft.wv.vectors

# Using fasttext
