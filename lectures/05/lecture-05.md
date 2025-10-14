# Lecture 5: Word Embeddings and Word2Vec

## Feature Representation for Sequential Data

In natural language, words appear in sequences. Consider a text corpus where we observe words at various positions:

$$\ldots, X_{t-3}, X_{t-2}, X_{t-1}, X_t, X_{t+1}, X_{t+2}, X_{t+3}, \ldots$$

where $X_t$ represents the word at position $t$ in our corpus.

### Vocabulary Representation

**Definition 5.1** Let $\mathcal{V}$ be our vocabulary of size $N$. Each word $w \in \mathcal{V}$ is represented as a one-hot encoded vector:

$$X_w = \mathbb{I}(w) = \begin{bmatrix} 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix} \in \{0,1\}^N$$

where the $i$-th entry is 1 if $w$ is the $i$-th word in the vocabulary, and 0 otherwise.

This representation, while sparse and high-dimensional, provides an unambiguous encoding of discrete tokens into a format amenable to mathematical manipulation.

### Context Windows and the Skip-gram Dataset

The fundamental insight of distributional semantics is that words appearing in similar contexts tend to have similar meanings. To operationalize this, we define a context window.

**Definition 5.2** For a center word $X_t$ and context window size $k$, the context consists of the $2k$ surrounding words:
$$\text{Context}(X_t) = \{X_{t-k}, \ldots, X_{t-1}, X_{t+1}, \ldots, X_{t+k}\}$$

**Example:** For $k=2$ and center word $X_t$:
$$\underbrace{X_{t-2}, X_{t-1}}_{\text{left context}}, \; X_t, \; \underbrace{X_{t+1}, X_{t+2}}_{\text{right context}}$$

**Dataset Construction:** We construct training dataset $\mathcal{D}$ by pairing each center word with each word in its context:

$$\mathcal{D} = \{(X_w^{(1)}, X_c^{(1)}), \ldots, (X_w^{(M)}, X_c^{(M)})\}$$

where $X_w$ denotes a center word and $X_c$ denotes a context word. For each occurrence of a word in the corpus, we generate multiple training pairs—one for each word in its context window.

---

## The Word2Vec Model Architecture

Word2Vec employs two embedding matrices that map discrete vocabulary indices to continuous vector representations.

### Embedding Matrices

**Definition 5.3** The model maintains two matrices:

1. **Center word embeddings:** $U \in \mathbb{R}^{K \times N}$, where column $U_w$ is the $K$-dimensional embedding for center word $w$

2. **Context word embeddings:** $V \in \mathbb{R}^{K \times N}$, where column $V_c$ is the $K$-dimensional embedding for context word $c$

Here $K \ll N$ is the embedding dimension (typically $K \in [50, 300]$ while $N$ may be 50,000 or larger).

**Extraction Operation:** For word $w$ with one-hot encoding $X_w$:
$$U_w = U \cdot X_w$$

This extracts the $w$-th column of $U$, giving us the embedding vector for word $w$.

### Model Predictions

The Skip-gram model predicts context words given a center word. The score for context word $c$ given center word $w$ is computed via the inner product:

$$\text{score}(w,c) = U_w^T V_c$$

**Key Observations:**
- $U_w^T V_c$ is large when center word $w$ and context word $c$ frequently co-occur
- $U_w^T V_c$ is small (or negative) when they rarely co-occur
- The inner product provides a natural measure of compatibility between embeddings

To convert scores to probabilities, we apply the softmax function:

**Definition 5.4** The probability of observing context word $c$ given center word $w$ is:

$$P(X_c | X_w; U, V) = \frac{e^{U_w^T V_c}}{\sum_{j=1}^N e^{U_w^T V_j}} \in [0,1]^N$$

This ensures proper normalization: $\sum_{c=1}^N P(X_c | X_w; U, V) = 1$.

---

## The Word2Vec Training Algorithm

### Algorithm Overview

**Input:** Corpus yielding dataset $\mathcal{D} = \{(X_w^{(i)}, X_c^{(i)})\}_{i=1}^M$, embedding dimension $K$, learning rate $\eta$

**Output:** Trained embedding matrices $U^*, V^*$

**Step 1: Initialization**
$$U, V \sim \mathcal{N}(0, 0.01^2) \in \mathbb{R}^{K \times N}$$

Initialize both matrices with small random values drawn from a Gaussian distribution.

**Step 2: Training Loop**

For each training pair $(X_w, X_c) \in \mathcal{D}$:

**(a) Extract center word embedding:**
$$U_w = U \cdot X_w \in \mathbb{R}^K$$

**(b) Compute logits:**
$$z = U_w^T V \in \mathbb{R}^N$$

This computes the score between center word $w$ and all possible context words simultaneously.

**(c) Compute probability distribution:**
$$P(X_c | X_w; U, V) = \text{softmax}(U_w^T V) = \frac{e^{U_w^T V}}{\sum_{j=1}^N e^{U_w^T V_j}} \in [0,1]^N$$

**(d) Compute negative log-likelihood loss:**
$$\text{NLL}(U, V | X_w, X_c) = -X_c \cdot \log P_{X_c | X_w}$$

Since $X_c$ is one-hot encoded, this simplifies to:
$$\text{NLL} = -\log P(X_c = c | X_w = w)$$

where $c$ is the index of the true context word.

**(e) Compute gradients:**

The gradient with respect to the center word embedding:
$$\nabla_{U_w} \text{NLL} = V \cdot (P_{X_c|X_w} - X_c)^T \in \mathbb{R}^{K \times 1}$$

The gradient with respect to the context word embeddings:
$$\nabla_V \text{NLL} = U_w \cdot (P_{X_c|X_w} - X_c)^T \in \mathbb{R}^{K \times N}$$

These gradients follow from the chain rule applied to the softmax and cross-entropy loss.

**(f) Update parameters:**
$$U_w \leftarrow U_w - \eta \nabla_{U_w} \text{NLL}$$
$$V \leftarrow V - \eta \nabla_V \text{NLL}$$

**Step 3: Repeat**

Continue iterating through the dataset for multiple epochs until convergence.

### Computational Considerations

The softmax normalization requires summing over all $N$ vocabulary words, making each gradient computation $O(KN)$. For large vocabularies, this becomes computationally prohibitive.

**Practical Solutions:**
1. **Negative Sampling:** Replace softmax with binary classification, sampling only a few negative examples
2. **Hierarchical Softmax:** Use a binary tree structure to reduce complexity from $O(N)$ to $O(\log N)$

These approximations maintain the essential structure while dramatically improving training efficiency.

### Theoretical Justification

The Word2Vec objective can be understood as factorizing the pointwise mutual information (PMI) matrix. The model learns to represent words such that:

$$U_w^T V_c \approx \text{PMI}(w,c) = \log \frac{P(w,c)}{P(w)P(c)}$$

This provides connection between the embedding geometry and statistical properties of word co-occurrence.

---

## Training Strategies for Word2Vec

The computational bottleneck in the standard Word2Vec algorithm lies in the softmax normalization, which requires $O(N)$ operations per training example. For vocabularies with hundreds of thousands of words, this becomes intractable. We present two principal strategies that address this challenge while preserving the essential learning dynamics.

### Negative Sampling

Negative sampling transforms the multi-class classification problem into a series of binary classification problems, dramatically reducing computational complexity.

#### Mathematical Formulation

**Definition 5.5** Instead of modeling $P(X_c | X_w)$ over all vocabulary words, we model the probability that word $c$ appears in the context of word $w$:

$$P(D = 1 | w, c) = \sigma(U_w^T V_c) = \frac{1}{1 + e^{-U_w^T V_c}}$$

where $D = 1$ indicates that $(w,c)$ is a genuine word-context pair from the corpus, and $\sigma(\cdot)$ is the sigmoid function.

**Negative Sample Generation:** For each positive pair $(w,c)$ from the corpus, we sample $k$ negative context words $\{n_1, n_2, \ldots, n_k\}$ according to a noise distribution $P_n(w)$.

**Definition 5.6** The noise distribution is typically chosen as:
$$P_n(w) = \frac{[\text{count}(w)]^{3/4}}{\sum_{w' \in \mathcal{V}} [\text{count}(w')]^{3/4}}$$

The $3/4$ exponent smooths the unigram distribution, giving rare words higher sampling probability than their corpus frequency would suggest.

#### Negative Sampling Objective

**Definition 5.7** The negative sampling objective for a single training example $(w,c)$ is:

$$\mathcal{L}_{\text{NEG}}(w,c) = \log \sigma(U_w^T V_c) + \sum_{i=1}^k \mathbb{E}_{n_i \sim P_n} [\log \sigma(-U_w^T V_{n_i})]$$

This objective maximizes the probability of the positive pair while minimizing the probability of negative pairs.

#### Algorithm: Negative Sampling Training

**Input:** Training pair $(w,c)$, negative samples $k$, noise distribution $P_n$

**Step 1:** Sample negative contexts
$$\{n_1, n_2, \ldots, n_k\} \sim P_n$$

**Step 2:** Compute positive gradient
$$\nabla_{U_w} \mathcal{L}_{\text{pos}} = V_c \cdot \sigma(-U_w^T V_c)$$
$$\nabla_{V_c} \mathcal{L}_{\text{pos}} = U_w \cdot \sigma(-U_w^T V_c)$$

**Step 3:** Compute negative gradients
For each negative sample $n_i$:
$$\nabla_{U_w} \mathcal{L}_{\text{neg}}^{(i)} = -V_{n_i} \cdot \sigma(U_w^T V_{n_i})$$
$$\nabla_{V_{n_i}} \mathcal{L}_{\text{neg}}^{(i)} = -U_w \cdot \sigma(U_w^T V_{n_i})$$

**Step 4:** Update parameters
$$U_w \leftarrow U_w + \eta \left(\nabla_{U_w} \mathcal{L}_{\text{pos}} + \sum_{i=1}^k \nabla_{U_w} \mathcal{L}_{\text{neg}}^{(i)}\right)$$

**Complexity Analysis:** Negative sampling reduces the per-example complexity from $O(KN)$ to $O(K(1+k))$, where typically $k \in [5, 20] \ll N$.

### Hierarchical Softmax

Hierarchical softmax organizes the vocabulary in a binary tree structure, reducing the normalization complexity from $O(N)$ to $O(\log N)$.

#### Tree Construction

**Definition 5.8** Let $\mathcal{T}$ be a binary tree where:
- Each leaf corresponds to a word $w \in \mathcal{V}$
- Each internal node $n$ has an associated parameter vector $\theta_n \in \mathbb{R}^K$
- The path from root to leaf $w$ is denoted $\text{path}(w) = \{n_1, n_2, \ldots, n_{L(w)}\}$

where $L(w)$ is the depth of word $w$ in the tree.

**Huffman Tree Construction:** To minimize expected computation, we construct a Huffman tree where frequent words have shorter paths:
$$\mathbb{E}_{w \sim P_{\text{corpus}}}[L(w)] = \sum_{w \in \mathcal{V}} P_{\text{corpus}}(w) \cdot L(w)$$

#### Probability Computation

**Definition 5.9** The probability of word $w$ given center word embedding $U_w$ is:

$$P(w | U_w) = \prod_{i=1}^{L(w)} P(\text{choice}_i | U_w, n_i)$$

where $\text{choice}_i \in \{0,1\}$ indicates whether we take the left (0) or right (1) branch at node $n_i$.

**Binary Classification at Each Node:** At internal node $n_i$:
$$P(\text{choice}_i = 1 | U_w, n_i) = \sigma(U_w^T \theta_{n_i})$$
$$P(\text{choice}_i = 0 | U_w, n_i) = 1 - \sigma(U_w^T \theta_{n_i})$$

#### Hierarchical Softmax Training

**Step 1:** For training pair $(w,c)$, identify path $\text{path}(c) = \{n_1, \ldots, n_{L(c)}\}$

**Step 2:** Compute loss along path
$$\mathcal{L}_{\text{HS}}(w,c) = -\sum_{i=1}^{L(c)} \left[ d_i \log \sigma(U_w^T \theta_{n_i}) + (1-d_i) \log \sigma(-U_w^T \theta_{n_i}) \right]$$

where $d_i \in \{0,1\}$ is the direction taken at node $n_i$ on the path to $c$.

**Step 3:** Update parameters
For each node $n_i$ on the path:
$$\nabla_{U_w} \mathcal{L}_{\text{HS}} = \sum_{i=1}^{L(c)} \theta_{n_i} \cdot (d_i - \sigma(U_w^T \theta_{n_i}))$$
$$\nabla_{\theta_{n_i}} \mathcal{L}_{\text{HS}} = U_w \cdot (d_i - \sigma(U_w^T \theta_{n_i}))$$

**Complexity Analysis:** Hierarchical softmax achieves $O(K \log N)$ complexity per example, with the logarithmic factor arising from the tree depth.

### Comparative Analysis

| Strategy | Complexity | Memory | Approximation Quality |
|----------|------------|--------|----------------------|
| Full Softmax | $O(KN)$ | $O(KN)$ | Exact |
| Negative Sampling | $O(K(1+k))$ | $O(KN)$ | High (with sufficient $k$) |
| Hierarchical Softmax | $O(K \log N)$ | $O(KN + K \log N)$ | Exact |

**Theoretical Guarantees:** Both approximation methods converge to embeddings with similar semantic properties as the full softmax, but with dramatically improved computational efficiency.

**Practical Considerations:**
- Negative sampling is simpler to implement and often preferred in practice
- Hierarchical softmax provides exact computation but requires careful tree construction
- The choice between methods often depends on vocabulary size and available computational resources

---

## Properties of Learned Embeddings

Upon convergence, the learned embeddings exhibit remarkable semantic and syntactic structure:

1. **Semantic similarity:** $\text{similarity}(U_{\text{king}}, U_{\text{queen}}) > \text{similarity}(U_{\text{king}}, U_{\text{apple}})$

2. **Analogical reasoning:** $U_{\text{king}} - U_{\text{man}} + U_{\text{woman}} \approx U_{\text{queen}}$

3. **Dimensional interpretability:** Individual dimensions sometimes correspond to interpretable semantic features

These properties emerge from the distributional hypothesis and the geometric structure imposed by the inner product similarity measure.