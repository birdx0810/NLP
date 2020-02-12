# Sequence to Sequence Learning with Neural Networks

> Ilya Sutskever, Oriol Vinyals and Quoc V. Le
>
> Google

DNNs are known to be extremely powerful and flexible for machine learning, yet their applications are limited to problems whose inputs and targets can be encoded with vectors of a fixed dimensionality.

Many problems are expressed with sequences such as speech recognition, machine translation, question answering etc.

Sequences pose a challenge for DNNs because they require that the dimensionality of the inputs and outputs to be known and fix. 

This paper propose an architecture of two LSTMs: (1) to read the input sequence to obtain a large fixed dimensional vector representation, and (2) to extract output sequence from the vector representation (a language model).

![](https://i.imgur.com/4jaOYHW.png) 

## Model

RNNs are a natural generalization of feedforward neural networks to sequences. 

- Given a sequence of inputs $(x_1,...,x_T)$, a standard RNN computes a sequence of outputs $(y_1,...,y_T)$. Where the length of the target sequence

$$
\begin{aligned}
h_t &= \sigma(W^{hx}x_t + W^{hh}h_{t-1}) \\
y_t &= W^{yh}h_t
\end{aligned}
$$



