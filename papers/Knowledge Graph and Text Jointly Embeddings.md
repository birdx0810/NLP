# Knowledge Graph and Text Jointly Embeddings
###### tags: `GroupMeeting`

> Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen
> Microsoft & SYSU(CN)

## TL;DR

Embed Entities and Words into same continuous vector space
Goal: Score any candidate relational facts between entity and words

## Knowledge Graph Embeddings

- Relation facts are represented ion the form of a triplet
    - $(\text{head entity, relation, tail entity})$
- Entity is represented as a $k$-dimensional vector
- Relation is characterized by an operation $\mathfrak{R}^k$
- Candidate fact can be asserted by simple vector operations
- Learnt by minimizing global loss function of all entities and relations in KG
- Each entity embedding encodes both local and global connectivity patterns of the original graph
- Problems: Missing facts (entities/relations) = *Out-of-kb*

## Word Embeddings

- Learning from unlabeled text corpus (word2vec) by predicting the context of each word or predicting the current word given its context
- e.g. $vec(\text{king}) - vec(\text{queen}) \approx vec(\text{boy}) - vec(\text{girl})$
- Problem: Does not know the relation between entity pairs

## Introduction

Consists of:
- Knowledge model
- Text model
- Alignment model

Knowledge and text model score fact $(h,r,t)$ based on $||\textbf{h} + \textbf{r} - \textbf{t}||$
- $r$ is supervised in KE
- $r$ is a hidden variable

Alignment model guarantees embeddings of enities and words lie in the same space and impels the two models to enhance each other.