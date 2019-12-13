# Neural Turing Machine

> Graves et. al, 2014  
> Affiliation: Google  
> Original Paper: [Link](https://arxiv.org/abs/1410.5401)  
> GitHub: [Link](https://github.com/MarkPKCollier/NeuralTuringMachine)

- Computers are mainly composed of memory, control flow and arithmetic/logical operations
- Modern machine learning has largely neglected the use of logical flow control and external memory when it is evident that memory is helpful for storage & retrieval
- A Neural Turing Machine (NTM) is a differentiable computer that can be trained by gradient descent **(Every component is differentiable)**

## Model Architecture

![](https://i.imgur.com/HQLydeS.png)

- Neural Network Controller
  - Interacts with the external world via input/output vectors
- External Memory Bank 
  - Controller also interacts with **memory matrix** using read/write heads (operations)
  - Read/write operations are "blurry" and the controllers would be "focusing" on a small portion of memory only (Attention weights?)
  - Let $\mathcal{M}_t$ be an $N \times M$ memory matrix at timestamp $t$ ; where $N$ is the number of memory locations and $M$ is the embedding size at each location    

- Read Heads
  - Let $w_t$ be a vector of weightings over $N$ locations emitted by a read head at time $t$. Since all weightings are normalised, the $N$ elements $w_t(i)$ obey the following constraints
  - $\sum_i w_t(i) = 1, \ 0 \leq w_t(i) \leq 1, \ \forall i$
  - The length $M$ read vector $r_t$ returned by the head is defined as a convex combination of the row-vectors $M_t(i)$ in memory
  - $r_t \gets \sum_iw_t(i)M_t(i)$

- Write Heads
  - decompose of two parts: 
  - *erase* vector $e_t$
    - pointwise multiplication (memory location is reset to 0 only if both weight and element are 1)
    - Memory vectors $M_{t-1}(i)$ Given weight $w_t$
    - $\tilde{M}_t(i) \gets M_{t-1}(i)[1-w_T(i)e_t]$
    - $1$ is a vector of all 1-s and 
  - *add* vector $a_t$
    - $M_t(i) \gets \tilde{M}_t(i)+w_t(i)a_t$