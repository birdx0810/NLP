# A Neural Conversational Model

> Oriol Vinyals, Quoc Le (ICML Workshop 2015)
> Paper: [Link](https://arxiv.org/abs/1506.05869)

![](https://i.imgur.com/BVdU8P5.png)

## Methodology

Based on the seq2seq framework of ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf))

### Encoder
- Uses LSTMs due to the vanishing gradient problems in vanilla RNNs
- Input sentence could be the concatenation of the previous conversations
- The `<eos>` token could be viewed as the "thought vector" of the sentence storing the information of the context sentence

### Decoder
- Outputs the reply to the context
- The predicted output is used as the input to predict the next output
- Beam search would be a less greedy approach

Objective
: Maximise the **cross-entropy** of the correct sequence given its context

***Notes***:
- The objective function **does not capture the actual objective** achieved through human communication
- Lacks consistency and general world knowledge for an unsupervised model

## Evaluation

- vs. [CleverBot](https://www.cleverbot.com/human)
- Evaluated by manual inspection (MTurk) and Perplexity

***Notes***:
- There remains an open research problem of designing a good metric to measure the quality of a NCM

## Conclusion
- Generating simple and basic conversations are feasible and could produce rather proper answers
- Could extract "knowledge" from a noisy but open-domain dataset
- Requires substantial modifications to deliver realistic conversations