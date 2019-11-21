# -*- coding: utf-8 -*-
'''
This is a walkthrough that introduces different types of word tokenizers:

The code below shows various tokenization methods for characters and words (hopefully applicable for different languages, e.g. Chinese).
The tokenizers would be written as a function, with the input being a list of raw `sentences` as shown below.
Please note that the methods below do not necessary preprocess or clean the data below.
Author: birdx0810
'''

sentences = [
    'Outline is an open-source Virtual Private Network (VPN).',
    'Outline is a created by Jigsaw with the goal of allowing journalists to have safe access to information, communication and report the news.', 
    'It is well known for its ease of use and could be deployed and hosted by the average journalist.',
    'Jigsaw is an incubator within Alphabet that uses technology to address geopolitical issues and combat extremism.',
    'One of their other known projects is Digital Attack Map.'
]

def word_tokenizer(sentences):
    for s in sentences:
        tokenized = [s.split() for s in sentences]
    return tokenized

tokenized = word_tokenizer(sentences)
print(tokenized)