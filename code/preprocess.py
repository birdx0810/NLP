# -*- coding: utf-8 -*-
'''
This is an example for preprocessing a raw text corpus, which is The Tragedy of Romeo and Juliet in The Complete Works of William Shakespeare from Project Gutenberg.
Of course, we could easily get it via nltk.corpus package as shown in the latter part of the code. But as we could see, there is a large difference in the number of words of our corpus.
Thus, it is indeed more helpful knowing how to preprocess/normalize raw text as we could have a full grasp of our dataset.

nltk requirements:
```
import nltk
nltk.download('shakespeare')
nltk.download('punkt')
```
'''

from nltk import word_tokenize

# Corpus path
path = f'data/shakespeare.txt'

with open(path, 'r') as f:
    corpus = []
    start = False
    for line in f:
        if start == True and line != '\n':
            if line.isupper() or line.strip('\n').isnumeric() or line.startswith(' ') or line.startswith('Scene') or line.startswith('Contents') or line.startswith('SCENE') or line.startswith('ACT'):
                continue
            else: corpus.append(line.strip('\n').strip())
        if line == 'THE TRAGEDY OF ROMEO AND JULIET\n':
            # Ignore text until title is found (fully matches string)
            start = True

# Tokenize sentences
text = []
for sentence in corpus:
    words = word_tokenize(sentence)
    for word in words:
        text.append(word)

print(f'Tokens: {len(text)}')

from nltk.corpus import shakespeare

text = shakespeare.words('r_and_j.xml')
# Since there are forewords included, we need to remove them
text = text[50:]

print(f'Tokens: {len(text)}')
