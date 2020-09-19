# Tokenizers

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
