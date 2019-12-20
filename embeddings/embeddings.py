# -*- coding: utf-8 -*-
'''
This is a walkthrough that introduces different types of word embeddings:
- Count Vectorizer
- TF-IDF
- word2vec
- GloVe
- FastText
The code below shows the methods for obtaining the word vectors and vocabulary dictionary.
Please take note that the inputs are raw text from the `sentences` variable below. Thus, it has not been preprocessed/cleaned.
Author: birdx0810
'''

sentences = [
    'Outline is an open-source Virtual Private Network (VPN).',
    'Outline is a created by Jigsaw with the goal of allowing journalists to have safe access to information, communication and report the news.', 
    'It is well known for its ease of use and could be deployed and hosted by the average journalist.',
    'Jigsaw is an incubator within Alphabet that uses technology to address geopolitical issues and combat extremism.',
    'One of their other known projects is Digital Attack Map.'
]

tc_sent = [
    '資管系的技術課程幾乎都離不開程式。', 
    '當然，有一個好的編譯器可以提高撰寫程式的效率。',
    'Visual Studio Code 是微軟所開發的免費開源程式碼編譯器；',
    '也是根據 Stack Overflow 最多開發者在使用的編譯器哦～',
    '那我們來看看這個編譯器為啥會這麼厲害吧～'
]

#########################
# Count Vectorizer (One-hot encoding)
# Method: count the occurrence of each word in each document
#########################
def count_embeddings(sentences):
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()
    count_vectors = cv.fit_transform(sentences)
    vocabulary = cv.vocabulary_
    vectors = count_vectors.toarray()
    return vocabulary, vectors

#########################
# TF-IDF Transformation/Vectorization
# Method: Term Frequency-Inverse Document Frequency, a statistical measure for calculating the relevance of a term within document.
# Note: 
# - This is a follow up approach on Count Vectorization, thus the input is from derived from above and not the raw corpus.
#########################

# Transformer Method
'''
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_weight = transformer.fit(count_vectors)
tfidf_score = transformer.transform(count_vectors)
vocabulary = transformer.vocabulary_
vectors = tfidf_score.toarray()
'''

# Vectorizer Method
def tfidf_embeddings(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(use_idf=True)
    tfidf_score = tfidf.fit_transform(sentences)
    vocabulary = tfidf.vocabulary_
    vectors = tfidf_score.toarray()
    return vocabulary, vectors

#########################
# word2vec
# Method: Common Bag-of-Words and Skip-Gram
# Note: 
# - sg = skipgram(1) or cbow(0, default)
# - hs = hierarchical softmax(1) or negative sampling(0, default)
# - w2v.train(more_sentences) to learn new sentences (online learning)
#########################

def w2v_embeddings(sentences):
    from gensim.models import word2vec

    tokenized = [sentence.split() for sentence in sentences]
    w2v = word2vec.Word2Vec(tokenized, min_count=1)
    vocabulary = w2v.wv.vocab
    vectors = w2v.wv.vectors
    return vocabulary, vectors

#########################
# GloVe
# Install: `pip install glove_python`
# Method: Capturing word embeddings through word-frequency and co-occurrence counts with a matrix.
#########################

def glove_embeddings(sentences):
    from glove import Corpus, Glove

    tokenized = [sentence.split() for sentence in sentences]
    corpus = Corpus()
    corpus.fit(tokenized, window=10)
    glove = Glove(no_components=512, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=5, no_threads=2)
    vocabulary = corpus.dictionary
    vectors = glove.word_vectors
    return vocabulary, vectors

#########################
# FastText
# Method: Takes into account the subword features and adds them to the final vector.
# Note: 
# - This is a follow-up approach on w2v. Thus, the `sg`, `hs` parameters are also applicable for the gensim version.
# - Another method used is the bag of tricks supervised method available using Facebook's original module.
#########################

def fasttext_embeddings(sentences):
    # Using gensim
    from gensim.models import fasttext

    tokenized = [sentence.split() for sentence in sentences]
    ft = fasttext.FastText(tokenized, min_count=1)
    ft.build_vocab(sentences=tokenized, update=True)
    ft.train(sentences=tokenized, total_examples=len(tokenized), epochs=10)
    vocabulary = ft.wv.vocab
    vectors = ft.wv.vectors
    return vocabulary, vectors
    

    # Using fasttext
    '''
    import fasttext

    model = fasttext.train_unsupervised(sentences, model='skipgram')
    vocabulary = model.get_words()
    vectors = model.get_output_matrix()
    print(type(vectors))
    return vectors
    '''

#########################
# ELMo
# Method: bi-directional LSTM and multiple vectors for single token for learning different contexts.
# Note: 
# - This is a follow-up approach on w2v. Thus, the `sg`, `hs` parameters are also applicable for the gensim version.
# - Another method used is the bag of tricks supervised method available using Facebook's original module.
#########################

def elmo(sentences):
    #TODO: ELMo wrapper
    pass


if __name__ == '__main__':
    voc, vec = fasttext_embeddings(sentences)
    with open('path.txt', 'w') as f:
        for word, obj in voc.items():
            # print(f'{word} {str(vec[obj.index].tolist()).strip("[").strip("]")}')
            f.write(f'{word} {str(vec[obj.index].tolist()).strip("[").strip("]")}\n')

