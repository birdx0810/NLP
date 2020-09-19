# Transformers
![](https://raw.githubusercontent.com/thunlp/PLMpapers/master/PLMfamily.jpg)
Code: [Huggingface Implementation](https://github.com/huggingface/transformers)

## Attention is all you need
> a.k.a Transformers
> Vaswani et al. (2017)
> Affiliates: Google
> Paper: [Link](https://arxiv.org/abs/1706.03762)
> Code: [Havard: The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

Loss Function: Cross Entropy

![](https://i.imgur.com/w5XmS2C.png)

Consists of an Encoder and Decoder
- Encoder (self attention)
    1. Self attention layer
    2. Feed-forward network

- Decoder
    1. Self attention layer
    2. Encoder-Decoder attention
    3. Feed-forward network

How does the model know the word sequence/order (without RNN)?
- Positional encoding
    - add a (one-hot) vector representing the sequence of each input word

Reference:
[Transformer - Heo Min-suk](https://www.youtube.com/watch?v=z1xs9jdZnuY)

#### Scaled Dot-Product Attention
$\text{Attention}(Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}}) \times V$

- $QK^T$ is the attention score where $Q$ (m $\times$ 1)
- $d_k$ is keys of dimension
- The output is the attention layer output (vector)

#### Multi-Head Attention
- Parallelization (8 attention layers)
$$
\begin{aligned}
Multihead(Q,K,V) &= \text{concat}(head_1, ...,head_h)W^O \\
\text{where } head_i &= \text{Attention}(QW^Q_i,KW^K_i,VW^V_i)
\end{aligned}
$$
References:

#### Encoder Layer (6 identical layers with different weights)
- Input: Word embeddings $\to$ Positional encoding $\to$ Attention Layer
- Concat Attention outputs * Weight Matrix -> FC Layer
- Output: Word embedding (same size as input)
* Residual Connection followed by normalization
    * for retaining the information in the positional encodings

#### Decoder Layer (6 identical layers)
**Masked MH Att. -> MH Att. -> FC Layer**
- Input word vectors one at a time
- Generates a word from input word
* Masked layer prevents future words to be part of the attention
* Second MH Att. layer $q$ are from the decoder and $k$, $v$ the are from encoder

#### Final Linear Layer & Softmax Layer
Linear: Generate logit
Softmax: Probability of word
Label smoothing (regularization for noisy labels)

## BERT (Biderectional Encoder Representation from Transformers)
###### Encoder
> Devlin et. al (2018) #NAACL
> Code: [GitHub](https://github.com/google-research/bert)
> Paper: [Link](https://arxiv.org/abs/1810.04805)

Reads entire sequence at once (non-directional)

$$
\text{Input: [CLS] } s_1 \text{ [SEP] } s_2 \text{ [SEP] }
$$

- Pre-train Objective (Optional)
    - Masked Language Model (MLM): Predicts masked tokens over vocabulary
    - Next Sentence Prediction (NSP): Binary Classification of whether both sentences are taken from the same context/reference
- Fine-tune Inference
  - GLUE: [CLS] for prediction
  - SQuAD: [CLS] Document [SEP] Question [SEP]
  - CNN-DM: BERT(Word Representation) + Downstream

#### Dataset
Wikipedia
BookCorpus

## SpanBERT
###### Encoder
> Joshi et al. (2019 July 24)
> Affiliation: facebook
> Paper: [Link](https://arxiv.org/abs/1907.10529)
> Code: [GitHub](https://github.com/facebookresearch/SpanBERT)

Span Masking (WWM)
Span Boundary Objective (SBO): Predict current words using front and back words
~~NSP~~

## RoBERTa
###### Encoder
> Liu et al. (2019 July 26)
> Paper: [Link](https://arxiv.org/abs/1907.11692)
> Code: [GitHub]

Dataset Size: 40GB
Batch Size: 2048, 4096
Dynamic Masking (BERT always mask same > overfit)
No NSP
Add sentences until fit sequence length 512

## ALBERT
###### Encoder
> Lan et al. (2019 Sept)
> Paper: [Link](https://arxiv.org/abs/1909.11942)
> Code: [GitHub](https://github.com/google-research/ALBERT)

Weight Sharing
Sentence Order Prediction (SOP)

### GPT: Improving Language Understanding by Generative Pre-training
###### Decoder
> Paper: [Link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
> Code: [GitHub](https://github.com/openai/finetune-transformer-lm)

### GPT-2: Language Models are Unsupervised Multitask Learners
###### Decoder
> Paper: [Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
> Code: [GitHub](https://github.com/openai/gpt-2)

### GPT-3: Language Models are Few-Shot Learners
###### Decoder
> Paper: [Link](https://arxiv.org/abs/2005.14165)
> Code: [GitHub](https://github.com/openai/gpt-3)

### TransformerXL: Attentive Language Models Beyond a Fixed-Length Context
###### AutoEncoder
> Paper: [Link](https://arxiv.org/abs/1901.02860)
> Code: [GitHub](https://github.com/kimiyoung/transformer-xl)

The vanilla transformer faces a couple of limitations:
- fixed maximum sequence length
- segments do not respect the sentence boundaries

To address this issue, two techniques were introduced:
- Segment-level Recurrence
![](https://2.bp.blogspot.com/--MRVzjIXx5I/XFCm-nmEDcI/AAAAAAAADuM/HoS7BQOmvrQyk833pMVHlEbdq_s_mXT2QCLcBGAs/s640/GIF2.gif)
- Relative Positional Encodings
![](https://4.bp.blogspot.com/-Do42uKiMvKg/XFCns7oXi5I/AAAAAAAADuc/ZS-p1XHZUNo3K9wv6nRG5AmdEK7mJsrugCLcBGAs/s640/xl-eval.gif)

### XLM: Cross-lingual Language Model Pretraining
###### AutoEncoder
> Paper: [Link](https://arxiv.org/abs/1901.07291)
> Code: [GitHub](https://github.com/facebookresearch/XLM/)

A transformer pre-trained using **one of the following** objectives, where each objective is selected w.r.t the downstream task:
- Masked Language Modeling
- Causal Language Modeling (next token prediction)
- Translation Language Modeling

### XLNet: Generalized Autoregressive Pretraining for Language Understanding
###### AutoRegressive
> Paper: [Link](https://arxiv.org/abs/1906.08237)
> Code: [GitHub](https://github.com/zihangdai/xlnet)

Denoising autoencoders aim to denoise/reconstruct the corrupted (masked) input data into the original data. They have the advantage of learning from forward-backward context. But the disadvantage lies in the `[MASK]` token being absent in finetuning and they are being assumed to be independent from other tokens.

Auto-regressive LMs are good for decoding tasks, but has loses forward-backward context.

### T5
###### AutoEncoders
> Paper: [Link](https://arxiv.org/abs/1910.10683)
> Code: [GitHub](https://github.com/google-research/text-to-text-transfer-transformer)
