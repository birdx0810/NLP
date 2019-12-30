# Abstract Meaning Representation

> Banarescu et. al, 2013 #ACL
>
> Paper: [Link](https://amr.isi.edu/a.pdf)
>
> Documentation: [Link](https://github.com/amrisi/amr-guidelines)

- Definition: 
  - Semantic representation language aimed at large-scale human annotation in order to build a giant semantics bank
  - Capture many aspects of meaning (features) within a simple data structure
  - Intends to abstract away from syntactic representations, in the sense that sentences which have similar meaning should be assigned the same AMR, even if they are not identically worded (What if sentences with similar/identical words have  different meanings?)
  - Biased towards English (as all programming languages), it not is not meant to function as an international supplementary language

## PENMAN (Bateman, 1990) Notation

![](https://i.imgur.com/IEodey9.png)

- **Edges** (:ARG) are relations

- **Nodes** (x/ item) are variables labeled with concepts. (`x` is an instance of an `item`)

- **Concepts** are technically edges that represent a variable (a type)

  - Could be mentioned multiple times ($\geq 1$ instance for each node)

- **Constants** are singleton nodes without variable (only value)

- **Basic Example**

  - "The dog is eating a bone"

  - ```
    (e / eat-01
    	:ARG0 (d / dog)
    	:ARG1 (b / bone)
    ```

  - Parse a sentence to a tree-like graph:![](https://i.imgur.com/sDhpNMq.png)
  - Non-alphabetic concepts starts with "!" 

- **Advanced Example**

  - "The dog ate the bone that he found"

    1. Find focus (eat)
    2. Add entities (dog, bone) $\to$ dog eat bone 
    3. Invert relation if needed $\to$ bone was found
    4. Add reentrancies $\to$ bone was found by dog

  - ```
    (e / eat-01
    	:ARG0 (d / dog)
    	:ARG1 (b / bone)
    		: ARG1-of (f / find-01
    					:ARG0 d))
    ```



### Reentrancy

- Repeat variables that are referenced multiple times

  - ```
    (w / want-01
    	:ARG0 (d / dog)
    	:ARG1 (e / eat-01
    		:ARG0 d
    		:ARG1 (b / bone)))
    ```

  - ![](https://i.imgur.com/8oqDqtn.png)



### Focus

- The main assertion (verb) or head of sentence (for inverse relations)

  - $\text{X ARG0-of Y = Y ARG0 X}$

  - "The dog **ran**" (Bold is focus)

  - ```
    (r / ran-01
    	:ARG0 (d / dog))
    ```

  - "The **dog** that ran" (Bold is focus)

  - ```
    (d /dog
    	:ARG0-of (r / ran))
    ```



### Constants

- `:polarity` - negation 

  - "The dog **did not** eat the bone"

  - ```
    (e / eat-01 :polarity - 
    	:ARG0 (d / dog)
    	:ARG1 (b / bone))
    ```

- `:quant` - numbers

  - "The dog ate **four** bones"

  - ```
    (e / eat-01
    	:ARG0 (d / dog)
    	:ARG1 (b / bone :quant 4))
    ```

- `:name` - names

  - "**Bono** the dog"

  - ```
    (d / dog
    	:name (n / name :op1 "Bono"))
    ```



### Entities

- Named entities



### Non-core Inventory

- `:mod` - used for non-core roles

  - e.g. "The yummy food"

  - ```
    (f / food
    	:mod (y / yummy))
    ```

- `:domain` $=$ `:mod-of`
  - e.g. "The food is yummy"

  - ```
    (y / yummy
    	:domain (f / food))
    ```

### Non-core Roles

- Relations with arguments but don't have specific meaning
- Coordination: e.g. Apples and bananas
- Names: e.g. Donald Trump



### Lexicon

- [PropBank](https://propbank.github.io/) ([Paper](http://www.lrec-conf.org/proceedings/lrec2002/pdf/283.pdf))
  - Generalize
    - Parts of speech and etmologically related words
    - Plurality and articles: mentions of same term go to the same variable (pronouns & nominal mentions), captures demonstratives (e.g. this house)
    - Tense: Doesn't generalize well cross-linguistically
  - Ignore
    -  Synonyms





### English AMR Alignment

> Szubert et. al, 2018
>
> Paper: [Link](https://www.aclweb.org/anthology/N18-1106.pdf)
>
> GitHub: [Link](https://github.com/ida-szubert/amr_ud)



## Neural AMR

> Seq2Seq Model for AMR Parsing and Generation
>
> Konstas et. al, 2017
>
> Paper: [Link](https://arxiv.org/abs/1704.08381)





## Glossary

- [Universal Dependencies](https://universaldependencies.org/)
  - Definition: a cross-lingual treebank annotation for multilingual parsing and learning



## Reference

Homepage: http://amr.isi.edu/
Guidelines: https://github.com/amrisi/amrguidelines/blob/master/amr.md
AMR Dictionary: http://www.isi.edu/~ulf/amr/lib/amr-dict.html