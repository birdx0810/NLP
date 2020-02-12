# What can we cram into a single vector

> Conneau et. al (2018)
>
> facebook
>
> 



A **probing task** is a classification problem that focuses on simple linguistic properties of a sentence.



## Introduction

1. Introduced 10 probing tasks
   - require single sentence embedding for generality and interpretability
   - it should be possible to construct large training sets in order to train parameter-rich multi-layer classifiers
   - nuisance variables such as lexical cues or sentence length should be controlled
   - tasks should address interesting set of linguistic properties (meaningful?)
2. Systemize probing task methodology
3. Explore a wide range of encoding architectures
4. Open source: [Link](https://github.com/facebookresearch/SentEval/tree/master/data/probing)



## Probing Tasks

### Surface information

- **SentLen** predict the *length* of sentences in terms of number of words
- **WC** (word content) recover information of the original words in the sentence

### Syntactic information

- **BShift** (bigram shift) tests if encoder is sensitive to legal word orders
- **TreeDepth** tests if an encoder infers hierarchical structure of sentences (longest path of root to leaf)
- **TopConst** (top constituent) clustering sentences into 20 constituent types/classes (19 + others)

### Semantic information

- **Tense** predict the tense of the main-claused verb (present, past etc.)
- **SubjNum** (subject number) finds the number of subjects of the main clause (avoid overlap)
- **ObjNum** (object number) finds the number of direct objects of the main clause (avoid overlap)
- **SOMO** (semantic odd man out) randomly replaces verbs and nouns $o$ with another verb or noun $r$, predicts whether a sentence has been modified
- **CoordInv** (coordination inversion) predict sentence is intact or modified after inverting the orders of half of the sentences



Test sentences were tested using Amazon Mechanical Turk