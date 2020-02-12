# ERNIE: Enhanced Language Representation with Informative Entities

> Zhang et. al (2019)
>
> Accepted by ACL 2019
>
> Affiliates: Tsinghua University, Beijing & Huawei Noah's Ark Lab
>
> Code: [Link](https://github.com/thunlp/ERNIE)



## TL;DR

ERNIE = BERT + KG

Loss Function: dEA + MLM + NSP



## Introduction

- Existing pre-trained language models rarely consider incorporating knowledge graphs
- Utilize both large-scale textual corpora and KGs to train ERNIE, which take full advantage of lexical, syntactic, and knowledge information simultaneously

![](https://i.imgur.com/9Rt7cO6.png)

Solid blue lines: existing knowledge facts

Dotted red lines: extracted from sentence in red

Dotted-dash green lines: extracted from sentence in green



### Main Challenges

1. Structured Knowledge Encoding: 
   - How to effectively extract & encode related information in KGs for language representation models
   - Recognize named entity mentions in text and align corresponding entity embeddings in KGs
2. Heterogeneous Information Fusion: 
   - How to design a pre-training objective to fuse lexical, syntactic, and knowledge information
   - Masked Language Model, Next Sentence Prediction and Denoising Entity Autoencoder
     - randomly masking named entity alignments and ask model to learn appropriate entities from KGs



Knowledge Driven NLP tasks:

1. Entity Typing
2. Relation Classification



## Related Work

Pre-training approaches include:

- Feature-based (Word2vec, GloVe)
- Fine-tuning (ULMFiR, GPT, BERT)

They ignore the incorporation of knowledge information