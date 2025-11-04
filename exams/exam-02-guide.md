# Exam 02 Study Guide

## ANLY-5800 Fall 2025 | Georgetown University

*Review material for exam covering lectures 4 - 6.*

---

## Overview

This study guide covers distributional semantics, word embeddings, and language modeling from lectures 4-6 of ANLY-5800. The material progresses from information retrieval and topic modeling through neural word representations to sequence modeling.

**Lecture Topics:**
- **Lecture 4:** Distributional Semantics and Information Retrieval (TF-IDF, PMI, Matrix Factorization, Topic Modeling)
- **Lecture 5:** Word Embeddings and Word2Vec (Skip-gram, CBOW, Training Strategies)
- **Lecture 6:** Language Modeling Part I (N-gram Models, Markov Assumption, Smoothing)

**Scope Notes:**
- Exam covers Lectures 4–6 only
- Direct LDA questions will not appear (see note in Lecture 4 section)
- Neural language modeling is high-level awareness only unless otherwise noted in class

---

## Lecture 4: Distributional Semantics and Information Retrieval

### Key Concepts

#### The Problem of Uniform Representation

**Zipf's Law:** Word frequency ∝ $1/\text{rank}^{\alpha}$ where $\alpha \approx 1$
- Few high-frequency words dominate
- Most words are rare
- Raw counts don't distinguish informative from uninformative words

**Information Theory Perspective:**
- High-frequency words → low information content
- Low-frequency words → high information content

#### TF-IDF Weighting

**Components:**
- **Term Frequency:** $\text{TF}(x)_{i,j} = x_j^{(i)}$ (raw count)
- **Inverse Document Frequency:** $\text{IDF}(x)_j = \left[\sum_{i=1}^{M} \mathbb{I}\{x_j^{(i)} > 0\}\right]^{-1}$

**TF-IDF Weight:** $W_{i,j} = \text{TF}(x)_{i,j} \cdot \text{IDF}(x)_j$

**Log-Scaled Variant:** $W_{i,j} = (1 + \log x_{j}^{(i)}) \cdot \log\left(\frac{M}{M_j}\right)$

where $M_j$ is the number of documents containing word $j$ at least once (i.e., document frequency).

**Document Retrieval:**
1. Weight query: $\mathbf{q}' = \mathbf{q} \odot \text{IDF}(\mathbf{x})$
2. Compute cosine similarity: $S_i(\mathbf{q}) = \frac{\mathbf{W}_i^T \mathbf{q}'}{\|\mathbf{W}_i\|_2 \|\mathbf{q}'\|_2}$
3. Select: $\hat{i} = \arg\max_i S_i(\mathbf{q}')$

#### Pointwise Mutual Information

**Definition:** $\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)}$
- Measures co-occurrence beyond chance
- Positive PMI: $\text{PPMI}(i,j) = \max(0, \text{PMI}(i,j))$

#### Dimensionality Reduction

**Motivation:** High-dimensional sparse vectors suffer from:
- Distance concentration (all distances become similar)
- Computational inefficiency
- Curse of dimensionality

**Distance Concentration Theorem:** In high dimensions with independent coordinates, relative contrast $\frac{\mathbb{E}[D_{\max}] - \mathbb{E}[D_{\min}]}{\mathbb{E}[D_{\min}]} \to 0$

#### Matrix Factorization Methods

**Truncated SVD:** $\mathbf{W} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ where $\mathbf{U} \in \mathbb{R}^{M \times K}$, $K \ll N$

**Latent Semantic Analysis (LSA):**
1. Apply truncated SVD to TF-IDF matrix
2. Documents: $\mathbf{U}$ (rows are documents in latent space)
3. Words: $\mathbf{V}$ (columns are latent concepts)
4. Query projection: $\mathbf{q}^{(\ell)} = \mathbf{q}^{(w)} \mathbf{V} \mathbf{\Sigma}^{-1}$

**Non-Negative Matrix Factorization (NMF):** $\mathbf{W} \approx \mathbf{U} \mathbf{V}$ with $\mathbf{U}, \mathbf{V} \geq 0$
- Maintains interpretability
- Topics as non-negative word combinations

#### Topic Modeling

**Framework:** Document ↔ Topic ↔ Word

**Latent Dirichlet Allocation (LDA):**

**Generative Process:**
1. Document-topic distribution: $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$
2. For each word:
   - Topic: $z_{d,i} \sim \text{Categorical}(\boldsymbol{\theta}_d)$
   - Word: $w_{d,i} \sim \text{Categorical}(\boldsymbol{\phi}_{z_{d,i}})$

**Inference:** Collapsed Gibbs sampling
$$p(z_{d,i} = k \mid \mathbf{Z}_{-(d,i)}, \mathbf{W}) \propto (n_{d,k}^{-(d,i)} + \alpha_k) \cdot \frac{n_{k,w_{d,i}}^{-(d,i)} + \beta_{w_{d,i}}}{\sum_{v} (n_{k,v}^{-(d,i)} + \beta_v)}$$

**Note**: we did not cover LDA this semester due to time constraints. Direct LDA questions will not be on Exam 02, however, gaining a general understanding of this approach and how it contrasts with other topic modeling approaches is encouraged. You can find details in the lecture notes.

### Study Questions

1. **Information Retrieval:** Explain why cosine similarity is preferred over Euclidean distance for document retrieval.
2. **Dimensionality:** Why does the distance concentration phenomenon make dimensionality reduction necessary for high-dimensional text data?
3. **Topic Modeling:** Compare the assumptions and trade-offs between LSA, NMF, and LDA for topic discovery.

---

## Lecture 5: Word Embeddings and Word2Vec

### Key Concepts

#### Feature Representation for Sequential Data

**Vocabulary Representation:** Each word $w \in \mathcal{V}$ is represented as a one-hot encoded vector:
$$X_w = \mathbb{I}(w) \in \{0,1\}^N$$

**Context Windows:** For center word $X_t$ and window size $k$:
$$\text{Context}(X_t) = \{X_{t-k}, \ldots, X_{t-1}, X_{t+1}, \ldots, X_{t+k}\}$$

**Skip-gram Dataset:** Pairs each center word with each context word:
$$\mathcal{D} = \{(X_w^{(1)}, X_c^{(1)}), \ldots, (X_w^{(M)}, X_c^{(M)})\}$$

#### The Word2Vec Model Architecture

**Embedding Matrices:**
1. **Center word embeddings:** $U \in \mathbb{R}^{K \times N}$
2. **Context word embeddings:** $V \in \mathbb{R}^{K \times N}$

**Model Predictions:** Score for context word $c$ given center word $w$:
$$\text{score}(w,c) = U_w^T V_c$$

**Probability:** $P(X_c | X_w; U, V) = \frac{e^{U_w^T V_c}}{\sum_{j=1}^N e^{U_w^T V_j}}$

#### Training Algorithm

**Objective:** Minimize negative log-likelihood:
$$\text{NLL}(U, V | X_w, X_c) = -\log P(X_c = c | X_w = w)$$

**Gradients:**
- Center word: $\nabla_{U_w} \text{NLL} = V \cdot (P_{X_c|X_w} - X_c)^T$
- Context words: $\nabla_V \text{NLL} = U_w \cdot (P_{X_c|X_w} - X_c)^T$

#### Training Strategies

**Computational Challenge:** Softmax normalization requires $O(N)$ operations per training example

**Negative Sampling:**
- Transform to binary classification: $P(D = 1 | w, c) = \sigma(U_w^T V_c)$
- Sample $k$ negative contexts from noise distribution: $P_n(w) = \frac{[\text{count}(w)]^{3/4}}{\sum_{w'} [\text{count}(w')]^{3/4}}$
- Objective: $\mathcal{L}_{\text{NEG}}(w,c) = \log \sigma(U_w^T V_c) + \sum_{i=1}^k \mathbb{E}_{n_i \sim P_n} [\log \sigma(-U_w^T V_{n_i})]$
- Complexity: $O(K(1+k))$ vs. $O(KN)$

**Hierarchical Softmax:**
- Organize vocabulary in binary tree
- Path-based probability computation: $P(w | U_w) = \prod_{i=1}^{L(w)} P(\text{choice}_i | U_w, n_i)$
- Complexity: $O(K \log N)$

### Study Questions

1. **Architecture:** Explain the role of the two embedding matrices $U$ and $V$ in Word2Vec.
2. **Training:** Why is negative sampling necessary, and how does it reduce computational complexity?
3. **Properties:** What semantic properties emerge in learned word embeddings?

---

## Lecture 6: Language Modeling Part I

### Key Concepts

#### The Language Modeling Problem

**Definition:** A language model assigns probabilities to sequences of tokens:
$$P(X^{(1)}, X^{(2)}, \ldots, X^{(T)})$$

**Applications:**
- Unconditional generation (text completion, dialogue)
- Conditional generation (translation, speech recognition)
- Sequence labeling (POS tagging, NER)
- Representation learning (contextual embeddings)

#### The Fundamental Challenge

**Exponential Complexity:** Number of possible sequences of length $T$ is $N^T$
- For $N = 10^4$ and $T = 10$: need to store $10^{40}$ probabilities
- Direct tabulation is computationally intractable

#### N-Gram Language Models

**Chain Rule Decomposition:**
$$P(X^{(1)}, \ldots, X^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$$

**Markov Assumption:** Each token depends only on preceding $n-1$ tokens:
$$P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)}) \approx P(X^{(i)} | X^{(i-n+1)}, \ldots, X^{(i-1)})$$

**Common Cases:**
- **Unigram (n=1):** $P(X^{(i)})$ — tokens are independent
- **Bigram (n=2):** $P(X^{(i)} | X^{(i-1)})$ — first-order Markov
- **Trigram (n=3):** $P(X^{(i)} | X^{(i-2)}, X^{(i-1)})$ — second-order Markov

#### Parameter Estimation

**Maximum Likelihood Estimate:**
$$P(X^{(i)} = j | X^{(i-n+1)}, \ldots, X^{(i-1)}) = \frac{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)}, j)}{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)})}$$

#### The Sparsity Problem

**Challenge:** As $n$ increases, many n-grams appear rarely or never in training data
- Zero probabilities break the model
- Data sparsity worsens with larger $n$

**Smoothing Techniques:**

**Add-k Smoothing (Laplace):**
$$P_{\text{smooth}}(w_i | w_{i-n+1}^{i-1}) = \frac{\text{Count}(w_{i-n+1}^{i-1}, w_i) + k}{\text{Count}(w_{i-n+1}^{i-1}) + k|V|}$$

**Good-Turing Smoothing:** Redistributes probability mass from seen to unseen events

**Kneser-Ney Smoothing:** Uses absolute discounting with interpolation:
$$P_{KN}(w_i | w_{i-n+1}^{i-1}) = \frac{\max(\text{Count}(w_{i-n+1}^i) - d, 0)}{\text{Count}(w_{i-n+1}^{i-1})} + \lambda(w_{i-n+1}^{i-1}) P_{KN}(w_i | w_{i-n+2}^{i-1})$$

#### Evaluation

**Perplexity:** Measures how well model predicts test sequence
$$\text{PP}(W) = P(w_1, \ldots, w_N)^{-1/N} = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i | w_1, \ldots, w_{i-1})}}$$

- Lower perplexity = better model
- Geometric mean of inverse probabilities
- Interpretable as "average branching factor"

### Study Questions

1. **Modeling:** Why is the Markov assumption necessary for tractable language modeling?
2. **Sparsity:** Explain the data sparsity problem and how smoothing techniques address it.
3. **Evaluation:** What does perplexity measure, and why is it a good metric for language models?

---

## Key Formulas Reference

### Mathematical Foundations
- **Chain rule:** $P(X^{(1)}, \ldots, X^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$
- **Bayes' theorem:** $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$

### Information Retrieval
- **TF-IDF:** $W_{i,j} = (1 + \log x_{i,j}) \cdot \log\left(\frac{M}{M_j}\right)$, where $M_j$ is the number of documents containing term $j$
- **PMI:** $\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)}$
- **Cosine similarity:** $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$

### Word Embeddings
- **Word2Vec score:** $\text{score}(w,c) = U_w^T V_c$
- **Skip-gram probability:** $P(c|w) = \frac{e^{U_w^T V_c}}{\sum_j e^{U_w^T V_j}}$
- **Negative sampling:** $P(D=1|w,c) = \sigma(U_w^T V_c)$

### Language Modeling
- **Chain rule:** $P(X^{(1)}, \ldots, X^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$
- **N-gram MLE:** $P(w_i | w_{i-n+1}^{i-1}) = \frac{\text{Count}(w_{i-n+1}^i)}{\text{Count}(w_{i-n+1}^{i-1})}$
- **Perplexity:** $\text{PP}(W) = P(w_1, \ldots, w_N)^{-1/N}$

### Neural Language Modeling
- **Architectures:** CNNs, RNNs, attention (and their associated time/memory complexity)
- **Regularization Techniques:** Dropout; norm penalties (L1/L2); weight decay as MAP
- **Scope:** High-level awareness; primary evaluation focuses on n-gram LMs from Lecture 6

---

## Practice Problems

### Problem 1: TF-IDF and Document Retrieval (Lecture 4)
Given a corpus with 4 documents and vocabulary {"cat", "dog", "pet"}:

**Document frequencies:**
- "cat": appears in 2 documents
- "dog": appears in 3 documents
- "pet": appears in 1 document

a) Calculate IDF values for each word using $\log\left(\frac{M}{M_j}\right)$.
b) For query "cat pet", compute the TF-IDF weighted query vector.
c) Explain why "pet" receives higher weight than "dog" in retrieval.

### Problem 2: PMI and PPMI (Lecture 4)
Suppose in a corpus: $P(i)=0.10$, $P(j)=0.05$, $P(i,j)=0.012$.

a) Compute $\text{PMI}(i,j)$.
b) Compute $\text{PPMI}(i,j)$.
c) Interpret what this value implies about co-occurrence beyond chance.

### Problem 3: Word2Vec Training Strategies (Lecture 5)
For a Word2Vec model with vocabulary size $N = 50{,}000$, embedding dimension $K = 200$:

a) What is the per-example complexity for full softmax?
b) With negative sampling and $k=10$, what is the per-example complexity?
c) Briefly compare negative sampling vs. hierarchical softmax in terms of complexity and approximation.

### Problem 4: N-gram Language Models and Smoothing (Lecture 6)
Training corpus: "the cat sat on the mat the dog ran".

a) Compute bigram probabilities $P(\text{cat}|\text{the})$ and $P(\text{mat}|\text{the})$.
b) Using add-$k$ smoothing with $k=1$ and vocabulary size $|V|$, write the smoothed formula for $P(\text{cat}|\text{the})$ and plug in counts.
c) Why is smoothing necessary as $n$ increases in n-gram models?

### Problem 5: Perplexity Calculation (Lecture 6)
For a test sentence "the cat sat" with:
- $P(\text{the}) = 0.1$
- $P(\text{cat}|\text{the}) = 0.3$
- $P(\text{sat}|\text{cat}) = 0.2$

a) Compute the sentence probability under a bigram model.
b) Compute the perplexity.
c) Interpret what this perplexity implies about model predictive power.

---

## Exam Preparation Tips

### Concepts
- **Connect approaches:** See how distributional semantics bridges from classical IR to neural methods
- **Understand trade-offs:** Know when to use each method (generative vs. discriminative, exact vs. approximate)
- **Geometric intuition:** Visualize embeddings, topic spaces, and probability distributions
- **Sparsity issues:** Understand why smoothing is necessary and how different methods address sparsity
- **Computational complexity:** Know the complexity trade-offs between exact and approximate methods
- **Model assumptions:** Understand when independence assumptions break down and their implications

### Preparation
- **Probability foundations:** Master conditional probability, Bayes' theorem, and MLE
- **Matrix operations:** Be comfortable with matrix factorization and dimensionality reduction
- **Optimization:** Understand gradient descent, convexity, and approximation methods

### Practical Applications
- **Algorithm details:** Know the steps for key algorithms (Word2Vec training, Gibbs sampling, smoothing)
- **Parameter interpretation:** Understand what hyperparameters control in each model
- **Evaluation metrics:** Know how to compute and interpret perplexity, similarity measures, and classification accuracy
---