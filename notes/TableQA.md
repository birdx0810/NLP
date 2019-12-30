# TableQAs

## Dataset & Leaderboard Benchmarks
- [WikiSQL](https://hackmd.io/9XLn1JDcQjmjwgghthJBsQ?both#Dataset-WikiSQL)
- [Spider](https://yale-lily.github.io/spider)
- [CoSQL](https://yale-lily.github.io/cosql)
- [SParC](https://yale-lily.github.io/sparc)

## Seq2SQL: Generating Structured Queries from Natural Language Using Reinforcement Learning
> Zhong et al., 2017
> Affiliation: SalesForce
> GitHub: [Code]() [Dataset](https://github.com/salesforce/WikiSQL)
- Introduced:
	- Seq2SQL (natural_lang-to-SQL translator model)
	- WikiSQL (natural language questions, SQL queries and SQL tables)

## Dataset: WikiSQL
- The WikiSQL task is to ==generate a SQL query from a natural language question and table schema==.
- A large crowd-sourced dataset for developing natural language interfaces for relational databases.

![](https://i.imgur.com/xMWrrNq.png)

- Table Column tokens
	- `Pick #`, `CFL Team` etc.
- Question tokens
	- `How`, `many`, `CFL`, `teams` etc.
- SQL Tokens
	- `SELECT`, `WHERE`, `COUNT` etc.

---

## Model Architecture

![](https://i.imgur.com/0UQpfy5.png)

- Generates a SQL query token-by-token from input sequence
- Input sequence is the concatenation of:
	- column names - the selection column and condition column
	- the question - the conditions of query
	- limited vocab - SQL language e.g. `SELECT`, `COUNT`, etc.
- Model takes augmented input and generates a SQL query via a pointer network

**Input sequence $x$**

$$
x = [\text{<col>};x_1^c;x_2^c;x_N^c;\text{<sql>};x^s;\text{<question>};x^q]
$$

Where
- The input sequence is a concatenation of features with sentinel tokens between the sequences to demarcate the boundaries
- $x_j^c = [x_{j,1}^c;x_{j,2}^c;...;x_{j, T_j}^c]$ the sequence of words in the name of the $j$th column, with $T_j$ being the total number of words in the $j$th column
- $x^s$ is the set of unique words in the SQL vocabulary
- $x^q$ is the sequence of words in the question

### Augmented Pointer Network

![](https://i.imgur.com/yX7QIsa.png =480x)
At each step, the decoder produces a vector that modulates a content-based attention mechanism over input.

1. The network encodes $x$ into a two-layer biLSTM encoder
	- $\displaystyle h^{enc} = \text{two_layer_biLSTM_encoder}(x)$
	- $h^{enc}_t$ is the hidden state of the encoder corresponding to $t$th word in sequence
2. Apply pointer network to h^{enc}
	- $\displaystyle y_0^{enc} = \text{pointer_network}(h^{enc})$
3. Decoder network is a two-layer LSTM
	- $g_s = \text{two_layer_LSTM_decoder}(y_{s-1})$
4. Decoder produces a scalar attention score for each position $t$ of input sequence
	- $\displaystyle \alpha^{ptr}_{s,t} = W^{ptr}tanh(U^{ptr}g_s+V^{ptr}h_t)$
5. Choose input token with highest score as next token of SQL query
	- $y_s = \text{argmax}(\alpha^{ptr}_s)$

**Problem**: APN does not leverage SQL structure.

### Seq2SQL

- SQL Structure consists of three components
	1. Aggregation operator
		- Loss: Cross Entropy
	2. `SELECT` identify column(s)
		- Loss: Cross Entropy
	3. `WHERE` row filter
		- Loss: Policy Gradient

#### Aggregation operator (e.g. `COUNT`, `SUM`, `MAX`) [Optional]
$$
\alpha^{inp}_t = W^{inp}h^{enc}_t$ \\
\beta^{inp} = \text{softmax}(\alpha^{inp}) \\
\kappa^{inp} = \sum_t \beta^{inp} h^{enc}_t
$$
- Where $\alpha^{inp} = [\alpha^{inp}_1,\alpha^{inp}_2,...,\alpha^{inp}_n]$ is a vector of scores
- Let $\alpha^{agg}$ denote the scores over operations such as `COUNT`, `MAX`, `MIN` and no aggregation operator `NULL`
- $\alpha^{agg}$ is calculated by adding a MLP to $\kappa^{agg}$
$$
\alpha^{agg} = W^{agg} / tanh(V^{agg}\kappa^{agg}+b^{agg}) + c^{agg}
$$

#### `SELECT` identify column(s)

#### `WHERE` row filter
		

--- 

## Further Reads
- SQLNet
- TypeSQL
- IncSQL
- SQLova
- X-SQL
- Execution Guided Decoding

---

# TableQA

> Vakulenko and Savenkov, 2017 #InformationRetrieval #SEMANTiCs2017
> Paper: [Link](https://arxiv.org/abs/1705.06504)
> GitHub: [Link](https://github.com/svakulenk0/MemN2N-tableQA)

- Most of the data around us are structured (tabular). Tabular data when large is difficult to analyze and search through. SQL queries are a threshold for the normal user.
- Task: Given an input table (or set of tables) $T$ and a natural language question (user query) $Q$ , output the correct answer $A$.

![](https://i.imgur.com/S1RbFdb.png)

---

## Glossary
- [NL2SQL/Seq2Tree](https://arxiv.org/pdf/1601.01280.pdf)