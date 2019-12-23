

# Corpus path

path = f'data/shakespeare.txt'
with open(path, 'r') as f:
    corpus = f.readlines()
    corpus = [line.strip('\n') for line in corpus if not line.isupper() or line is not '']

print(corpus[0:4])