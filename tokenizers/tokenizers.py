# -*- coding: utf-8 -*-
'''
This is a walkthrough that introduces different types of word tokenizers:
- Character Tokenizer
- Word (White-space) Tokenizer
The code below shows various tokenization methods for characters and words (hopefully applicable for different languages, e.g. Chinese).
The tokenizers would be written as a function, with the input being a list of `sentences` as shown below.
Please note that the methods below does not preprocess or clean the data below.
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

def char_tokenizer(sentences):
    tokenized = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            tmp.append(word)
        tokenized.append(tmp)
    return tokenized

def word_tokenizer(sentences):
    for s in sentences:
        tokenized = [s.split() for s in sentences]
    return tokenized

#TODO: Byte-Pair Encoding
def bpe_tokenizer(sentences, count=5):
    # Create Character Level Vocab
    tokens = char_tokenizer(sentences)
    processed = []
    for sentence in tokens:
        sent = ' '.join(sentence) + (' </w>')
        processed.append(sent)
    
    for i in range(count):
        # Get pairing statistics

        # Merge Vocabulary

        pass


if __name__ == '__main__':
    tokenized = bpe_tokenizer(tc_sent)
    # print(tokenized)