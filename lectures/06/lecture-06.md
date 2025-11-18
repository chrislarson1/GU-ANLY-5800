# Lecture 6: Language Modeling Part I

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

## Introduction: The Central Task of NLP

A language model assigns probabilities to the occurance of a token at a given position in a sequence conditioned on the presence of other tokens at other positions in the sequence.

**Definition 6.1** A language model is a probability distribution over sequences of tokens from a vocabulary $\mathcal{V}$:
$$P(X^{(1)}, X^{(2)}, \ldots, X^{(T)})$$
where each $X^{(i)} \in \mathcal{V} = \{1, 2, \ldots, N\}$ represents a token at position $i$ in the sequence.

Direct parameterization of such a distribution is not tractable. With language models, we use the Markov assumption to approximate this distrbution. LMs are the foundation of pretty much everything in NLP, including but not limited to the following:

**Unconditional Generation:**
- Text generation and completion
- Dialogue systems
- Summarization

**Conditional Generation:**
- Machine translation
- Speech recognition

**Sequence Labeling:**
- Part-of-speech tagging
- Named entity recognition

**Representation Learning:**
- Contextual word embeddings
- Transfer learning

The power of language modeling lies in its generality. Operating in the domain through which humans communicate—the space of natural language—language models can be applied to any task expressible through text. As we shall see, modern approaches have largely unified these disparate tasks under a text in-text out framework.

---

## The Probabilistic Framework

### The Fundamental Challenge

Consider a sequence $X = (X^{(1)}, \ldots, X^{(T)})$ where each token $X^{(i)} \in \{0, \ldots, N-1\}$ indexes into our vocabulary of size $N$. Our objective is to model:
$$P(X) = P(X^{(1)}, \ldots, X^{(T)})$$

**Exponential Complexity in $T$:** A naive approach would specify $P(X)$ by storing probabilities for all possible sequences. However, the number of distinct sequences of length $T$ is:
$$|\mathcal{X}| = N^T$$

For typical applications where $N \approx 10^4$ to $10^6$ and $T$ can be arbitrarily large, direct tabulation becomes computationally intractable. Even for modest vocabulary size $N = 10^4$ and sequence length $T = 10$, we would need to store $10^{40}$ probabilities.

Therefore, $P(X)$ cannot be modeled directly for arbitrary real-world sequences $\{V, T\}$. We require approximations that exploit structure in natural language. Despiting having to approximate the distribution of natural language, real world systems built on such models have achieved wild success.

---

## N-Gram Language Models

### The Chain Rule Decomposition

The product rule from probability theory provides our starting point:

**Theorem 6.1 (Chain Rule for Sequences)** Any joint distribution over a sequence can be factored as:
$$P(X^{(1)}, \ldots, X^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$$

**Proof:** Apply the product rule recursively:
$$\begin{aligned}
P(X^{(1)}, \ldots, X^{(T)}) &= P(X^{(T)} | X^{(1)}, \ldots, X^{(T-1)}) P(X^{(1)}, \ldots, X^{(T-1)}) \\
&= P(X^{(T)} | X^{(1)}, \ldots, X^{(T-1)}) P(X^{(T-1)} | X^{(1)}, \ldots, X^{(T-2)}) \cdots P(X^{(1)})
\end{aligned}$$

This factorization is exact—no approximation has been made. However, each conditional probability $P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$ still depends on the entire history, which grows with $i$.

### The Markov Assumption

The key insight enabling tractable language modeling is that recent context matters more than distant context. This motivates the Markov assumption.

**Definition 6.2 (n-gram Markov Assumption)** An n-gram model assumes that each token depends only on the preceding $n-1$ tokens:
$$P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)}) \approx P(X^{(i)} | X^{(i-n+1)}, \ldots, X^{(i-1)})$$

This is also referred to as the $(n-1)$-th order Markov property.

**Common Cases:**
- **Unigram (n=1):** $P(X^{(i)})$ — tokens are independent
- **Bigram (n=2):** $P(X^{(i)} | X^{(i-1)})$ — first-order Markov
- **Trigram (n=3):** $P(X^{(i)} | X^{(i-2)}, X^{(i-1)})$ — second-order Markov

Under the n-gram assumption, our sequence model becomes:
$$P(X^{(1)}, \ldots, X^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | X^{(i-n+1)}, \ldots, X^{(i-1)})$$

### Parameter Estimation via Maximum Likelihood

**Definition 6.3** For an n-gram model, the maximum likelihood estimate of conditional probabilities is:
$$P(X^{(i)} = j | X^{(i-n+1)}, \ldots, X^{(i-1)}) = \frac{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)}, j)}{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)})}$$

This follows directly from maximizing the log-likelihood over the training corpus.

**Example 6.1** Consider the bigram model trained on corpus: "the cat sat on the mat"

The bigram "the cat" appears once, and "the" appears twice (before "cat" and "mat"). Therefore:
$$P(\text{"cat"} | \text{"the"}) = \frac{\text{Count}(\text{"the cat"})}{\text{Count}(\text{"the"})} = \frac{1}{2} = 0.5$$

### The Sparsity Problem

The fundamental challenge with n-gram models emerges from Zipf's law (recall Lecture 01): word frequencies follow a power-law distribution where most words are rare. As $n$ increases:

**Parameter Growth:** The number of possible n-grams is $N^n$, which grows exponentially with $n$.

**Data Sparsity:** Most possible n-grams never appear in any finite training corpus. For these unseen n-grams, the MLE assigns probability zero.

**Undetermined:** If any n-gram in a test sequence has zero probability, the entire sequence receives zero probability:
$$P(X) = \prod_{i=1}^{T} P(X^{(i)} | X^{(i-n+1)}, \ldots, X^{(i-1)}) = 0$$

This renders the model useless for evaluating novel sequences.

In short, n-gram models estimate through counting, and do not generalize to unseen data.

### Smoothing Techniques

Smoothing redistributes probability mass from seen n-grams to unseen n-grams, addressing the zero-probability problem.

**Add-k Smoothing (Laplace Smoothing):**
$$P_{\text{smooth}}(X^{(i)} = j | X^{(i-n+1)}, \ldots, X^{(i-1)}) = \frac{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)}, j) + k}{\text{Count}(X^{(i-n+1)}, \ldots, X^{(i-1)}) + kN}$$

where $k > 0$ is a smoothing parameter (typically $k=1$).

**Properties:**
1. All probabilities are strictly positive
2. Probabilities still sum to 1
3. Simple to implement
4. Often too crude for practical applications

**Advanced Smoothing:** More sophisticated techniques like Kneser-Ney smoothing leverage the intuition that words appearing in many contexts should receive higher probability. These methods significantly outperform add-k smoothing but involve more complex parameter estimation.

### The Bias-Variance Tradeoff

The choice of $n$ involves a fundamental tradeoff:

**Small $n$ (e.g., bigram):**
- **Bias:** High approximation error — insufficient context for accurate predictions
- **Variance:** Low estimation error — many examples of each bigram

**Large $n$ (e.g., 5-gram):**
- **Bias:** Lower approximation error — richer contextual information
- **Variance:** High estimation error — extreme sparsity, few examples

This tradeoff is inescapable in finite-data regimes. The optimal choice depends on corpus size and language characteristics.

---

## Evaluation: Perplexity

Language models are evaluated based on how well they predict held-out test data. The standard metric is perplexity, which has a natural information-theoretic interpretation.

### Log-Likelihood

For test sequence $X = (X^{(1)}, \ldots, X^{(T)})$, the log-likelihood is:
$$\text{LL}(X) = \sum_{i=1}^{T} \log P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})$$

Higher log-likelihood indicates better predictive performance. However, log-likelihood is not sequence-length invariant, making cross-corpus comparisons problematic.

### Perplexity

**Definition 6.4** The perplexity of a language model on test sequence $X$ of length $T$ is:
$$\text{PPL}(X) = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)})\right)$$

Equivalently:
$$\text{PPL}(X) = P(X^{(1)}, \ldots, X^{(T)})^{-1/T}$$

**Interpretation:** Perplexity measures the effective vocabulary size that would produce the same uncertainty under a uniform distribution. A model with perplexity 100 is as surprised by the test data as a uniform distribution over 100 equally-likely alternatives.

### Boundary Cases

**Perfect Model:** If the model assigns probability 1 to the correct token at each position:
$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log(1)\right) = e^0 = 1$$

**Uniform Model:** A model uniformly distributing probability over vocabulary of size $N$:
$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log\frac{1}{N}\right) = \exp(\log N) = N$$

**Worst Possible Model:** If the model assigns zero probability to any token:
$$\text{PPL} = \infty$$

**Key Properties:**
1. **Lower is better:** Lower perplexity indicates better predictions
2. **Length-invariant:** Perplexity normalizes by sequence length
3. **Geometric mean:** Perplexity is the geometric mean of inverse probabilities
4. **Dependent on vocabulary size and corpus characteristics**

---

## Autoregressive Text Generation

Language models enable text generation through sequential sampling from the learned distribution. This process, termed autoregressive generation, samples one token at a time conditioned on all previous tokens.

### The Generation Algorithm

**Algorithm 6.1 (Autoregressive Generation)**

Given: Language model $P(X^{(i)} | X^{(1)}, \ldots, X^{(i-1)}; \theta)$, initial context $X_c = (X^{(1)}, \ldots, X^{(k)})$

**Procedure:**
1. Set $X = X_c$ (initialize with context)
2. **Repeat:**
   - Compute $i = \text{length}(X)$
   - Sample $\hat{X}^{(i+1)} \sim P(X | X^{(1)}, \ldots, X^{(i)})$
   - Append $\hat{X}^{(i+1)}$ to $X$
3. **Until:** $\hat{X}^{(i+1)} = \langle \text{STOP} \rangle$ or maximum length reached

**Key Observations:**
- Each token is sampled from the conditional distribution given all previous tokens
- The generation is sequential and cannot be parallelized
- Quality depends critically on the learned distribution $P(\cdot)$
- Different sampling strategies (greedy, beam search, nucleus sampling) trade off between quality and diversity

### Autoregression as Discriminative Modeling

Any parameterization of $P(X^{(i)} | X^{(i-n+1)}, \ldots, X^{(i-1)}; \theta)$ constitutes a discriminative model. Given the causal relationship $X^{(i-k:i)} \rightarrow X^{(i)}$, we can use this discriminative model to generate text by feeding its own output back as input.

This framework unifies language modeling with sequence-to-sequence tasks: both involve learning conditional distributions over tokens.

---

## Sequence Labeling with Hidden Markov Models

We now shift from unconditional language models to conditional models for sequence labeling tasks such as part-of-speech tagging and named entity recognition.

### The Sequence Labeling Problem

**Problem Statement:** Given a sequence of tokens $X^{(1)}, \ldots, X^{(T)}$ from vocabulary $\mathcal{V}$ where $|\mathcal{V}| = N$, assign a label $Y^{(i)} \in \mathcal{Y}$ to each token, where $\mathcal{Y}$ is a finite set of labels with $|\mathcal{Y}| = K$.

**Example Applications:**
1. **Part-of-speech tagging:** Assign grammatical categories (noun, verb, adjective, etc.) to words
2. **Named entity recognition:** Identify entities (person, organization, location) in text

**Key Challenge:** Labels exhibit strong sequential dependencies. The label at position $i$ depends on both the observed token $X^{(i)}$ and neighboring labels.

### Graphical Model Representation

A Hidden Markov Model represents the joint distribution over observed tokens $X$ and hidden labels $Y$ using a graphical model:

```
Y^(1) → Y^(2) → ... → Y^(T)
  ↓       ↓             ↓
X^(1)   X^(2)         X^(T)
```

This structure encodes two types of dependencies:
1. **Transition model:** $Y^{(i)} \rightarrow Y^{(i+1)}$ (label dependencies)
2. **Emission model:** $Y^{(i)} \rightarrow X^{(i)}$ (observation dependencies)

### The Markov Assumptions

**Assumption 6.1 (First-order Markov)** Each label depends only on the immediately preceding label:
$$P(Y^{(i)} | Y^{(1)}, \ldots, Y^{(i-1)}) = P(Y^{(i)} | Y^{(i-1)})$$

**Assumption 6.2 (Conditional Independence)** Each observation depends only on its corresponding label:
$$P(X^{(i)} | Y^{(1)}, \ldots, Y^{(T)}, X^{(1)}, \ldots, X^{(i-1)}) = P(X^{(i)} | Y^{(i)})$$

These assumptions enable tractable inference while capturing essential sequential structure.

### The Joint Distribution

Under the Markov assumptions, the joint distribution factors as:
$$P(X^{(1)}, \ldots, X^{(T)}, Y^{(1)}, \ldots, Y^{(T)}) = \prod_{i=1}^{T} P(X^{(i)} | Y^{(i)}) \cdot P(Y^{(i)} | Y^{(i-1)})$$

where we define $Y^{(0)} = \langle \text{START} \rangle$ as a special initial state.

This factorization reveals the two key components requiring parameterization:
1. **Emission probabilities:** $P(X^{(i)} | Y^{(i)})$
2. **Transition probabilities:** $P(Y^{(i)} | Y^{(i-1)})$

---

## HMM Parameter Estimation

### Emission Probabilities

The emission probabilities model the relationship between hidden labels and observed tokens.

**Definition 6.5** For label $k \in \mathcal{Y}$ and token $j \in \mathcal{V}$, the emission probability is:
$$\phi_{k,j} = P(X^{(i)} = j | Y^{(i)} = k)$$

**Maximum Likelihood Estimate:**
$$\hat{\phi}_{k,j} = \frac{\text{Count}(\text{word } j, \text{ tag } k)}{\text{Count}(\text{tag } k)}$$

This is the relative frequency of token $j$ appearing with label $k$ in the training corpus.

**Matrix Form:** The emission parameters can be organized as matrix $\boldsymbol{\Phi} \in \mathbb{R}^{K \times N}$ where:
- Each row corresponds to a label
- Each column corresponds to a token
- Entry $\phi_{k,j}$ gives $P(X=j|Y=k)$
- Each row sums to 1: $\sum_{j=1}^{N} \phi_{k,j} = 1$

### Transition Probabilities

The transition probabilities capture label sequence patterns.

**Definition 6.6** For labels $k, k' \in \mathcal{Y}$, the transition probability is:
$$\lambda_{k,k'} = P(Y^{(i+1)} = k' | Y^{(i)} = k)$$

**Maximum Likelihood Estimate:**
$$\hat{\lambda}_{k,k'} = \frac{\text{Count}(Y^{(i)} = k, Y^{(i+1)} = k')}{\text{Count}(Y^{(i)} = k)}$$

This is the relative frequency of label $k$ being followed by label $k'$.

**Matrix Form:** The transition parameters form matrix $\boldsymbol{\Lambda} \in \mathbb{R}^{K \times K}$ where:
- Entry $\lambda_{k,k'}$ gives $P(Y^{(i+1)}=k'|Y^{(i)}=k)$
- Each row sums to 1: $\sum_{k'=1}^{K} \lambda_{k,k'} = 1$

---

## Inference in Hidden Markov Models

Given trained parameters $\boldsymbol{\Phi}$ and $\boldsymbol{\Lambda}$, and an observed sequence $X$, we seek the most likely label sequence:
$$\hat{Y} = \arg\max_{Y} P(Y^{(1)}, \ldots, Y^{(T)} | X^{(1)}, \ldots, X^{(T)})$$

### The Viterbi Algorithm

Direct enumeration over all $K^T$ possible label sequences is intractable. The Viterbi algorithm exploits the Markov structure to find the optimal labeling efficiently using dynamic programming.

**Objective:**
$$\begin{aligned}
\hat{Y}^{(1:T)} &= \arg\max_{Y} \log P(Y, X) \\
&= \arg\max_{Y} \sum_{i=1}^{T} \log P(Y^{(i)} | Y^{(i-1)}) + \log P(X^{(i)} | Y^{(i)}) \\
&= \arg\max_{Y} \sum_{i=1}^{T} \log \lambda_{Y^{(i-1)}, Y^{(i)}} + \log \phi_{Y^{(i)}, X^{(i)}}
\end{aligned}$$

**Key Insight:** The optimal label at position $i$ depends only on the optimal path to position $i-1$ and the local scores at position $i$. This enables recursive computation.

**Algorithm 6.2 (Viterbi Decoding)**

**Input:** Observed sequence $X^{(1:T)}$, parameters $\boldsymbol{\Phi}, \boldsymbol{\Lambda}$

**Forward Pass:**

For $i = 1$ to $T$:
$$V_{i,k} = \max_{k'} \left[V_{i-1,k'} + \log \lambda_{k',k}\right] + \log \phi_{k,X^{(i)}}$$

where $V_{i,k}$ stores the log-probability of the best path to position $i$ ending in label $k$.

**Backward Pass:**

Starting from $\hat{Y}^{(T)} = \arg\max_k V_{T,k}$, recursively reconstruct:
$$\hat{Y}^{(i)} = \arg\max_{k} \left[V_{i,k} + \log \lambda_{k,\hat{Y}^{(i+1)}}\right]$$

**Complexity:** $O(TK^2)$ — polynomial in sequence length and label set size.

---

## Recurrent Neural Networks: Handling Variable-Length Sequences

While n-gram models and HMMs provide a foundation, they have fixed context windows. **Recurrent Neural Networks (RNNs)** process sequences of arbitrary length by maintaining a hidden state.

### Vanilla RNNs

**Definition 6.8 (Vanilla RNN).** For input sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T$ and hidden dimension $d_h$:
$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$
$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

**Key properties:**
- Hidden state $\mathbf{h}_t$ summarizes sequence history
- Parameters shared across all timesteps
- Can process sequences of any length

### Backpropagation Through Time (BPTT)

To compute gradients, we unroll the recurrence:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}
$$

Each term requires the chain rule through all previous timesteps:
$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} \mathbf{W}_{hh}^T \text{diag}(\tanh'(\mathbf{z}_i))
$$

### The Vanishing Gradient Problem

**Theorem 6.4 (Gradient vanishing).** For $t - k$ large, if the spectral norm $\|\mathbf{W}_{hh}\| < 1$, then:
$$
\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\right\| \leq \|\mathbf{W}_{hh}\|^{t-k} \to 0 \text{ exponentially}
$$

**Proof sketch:** The gradient is a product of $t-k$ matrices. If each has norm less than 1, the product shrinks exponentially. Since $\tanh'(z) \leq 1$, the derivative matrices have bounded norm.

**Consequence:** Vanilla RNNs cannot learn long-range dependencies (beyond ~10-20 steps). Information from early in the sequence is lost.

**Exploding gradients:** Conversely, if $\|\mathbf{W}_{hh}\| > 1$, gradients explode. **Solution:** Gradient clipping.

---

## Long Short-Term Memory (LSTM)

LSTMs, introduced by Hochreiter & Schmidhuber (1997), solve vanishing gradients through gating mechanisms and a separate memory cell.

### Architecture

**Definition 6.9 (LSTM cell).** At each timestep $t$, an LSTM maintains:
- **Hidden state** $\mathbf{h}_t \in \mathbb{R}^{d_h}$: short-term memory, output to next layer
- **Cell state** $\mathbf{c}_t \in \mathbb{R}^{d_h}$: long-term memory, internal to LSTM

**Three gates control information flow:**

**Forget gate** (what to remove from memory):
$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \in [0,1]^{d_h}
$$

**Input gate** (what new information to store):
$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \in [0,1]^{d_h}
$$
$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \in [-1,1]^{d_h}
$$

**Output gate** (what to output):
$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \in [0,1]^{d_h}
$$

**Cell and hidden state updates:**
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

where $\odot$ denotes element-wise multiplication.

### Why LSTMs Work

**Key insight:** The cell state $\mathbf{c}_t$ has an **additive** update path:
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**Gradient flow:**
$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t
$$

Unlike vanilla RNNs (multiplicative updates), gradients flow through addition, avoiding repeated matrix multiplications. If $\mathbf{f}_t \approx 1$ (forget gate open), gradients flow unimpeded across many timesteps.

**Theorem 6.5 (LSTM gradient flow).** The gradient $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_k}$ is:
$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_k} = \prod_{i=k+1}^{t} \text{diag}(\mathbf{f}_i)
$$

If forget gates are near 1, this product does not vanish.

---

## Gated Recurrent Unit (GRU)

GRUs (Cho et al., 2014) simplify LSTMs by merging cell and hidden state.

**Definition 6.10 (GRU cell).** Two gates:

**Reset gate** (how much past to forget):
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$

**Update gate** (interpolation between old and new):
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
$$

**Candidate activation:**
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)
$$

**Hidden state update:**
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

**Comparison with LSTM:**

| Feature | LSTM | GRU |
|---------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **States** | Hidden + cell | Hidden only |
| **Parameters** | More (~4× input size) | Fewer (~3× input size) |
| **Performance** | Often slightly better | Often comparable |
| **Training** | Slower | Faster |

**Use case:** GRU when computational budget is limited; LSTM when maximum capacity needed.

---

## Convolutional Neural Networks for Text

While RNNs process sequences sequentially, CNNs can capture local patterns in parallel.

### 1D Convolution for Text

**Definition 6.11 (1D Convolution).** For input sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ (embeddings) and filter $\mathbf{W} \in \mathbb{R}^{k \times d}$ of width $k$:
$$
[\mathbf{Z}]_i = \sum_{j=0}^{k-1} \sum_{m=1}^{d} W_{j,m} \cdot X_{i+j,m}
$$

where $i \in \{0, \ldots, T-k\}$.

**Intuition:** The filter slides over the sequence, detecting local patterns (n-grams).

**Multiple filters:** Use $F$ filters $\{\mathbf{W}^{(1)}, \ldots, \mathbf{W}^{(F)}\}$ to detect different patterns:
- Each filter learns to recognize a different n-gram pattern
- Filter $f$ produces feature map $\mathbf{z}^{(f)} \in \mathbb{R}^{T-k+1}$

**Multiple kernel sizes:** Use filters of different widths (e.g., $k \in \{3, 4, 5\}$) to capture various n-gram lengths.

### Pooling Operations

After convolution, reduce dimensionality via pooling:

**Max pooling** (extract most salient feature):
$$
z_{\text{pooled}}^{(f)} = \max_{i=1,\ldots,T-k+1} z_i^{(f)}
$$

**Interpretation:** The maximum activation indicates whether the pattern appears anywhere in the sequence.

**Average pooling:**
$$
z_{\text{pooled}}^{(f)} = \frac{1}{T-k+1} \sum_{i=1}^{T-k+1} z_i^{(f)}
$$

### CNN for Text Classification

**Architecture (Kim, 2014):**

1. **Embedding layer:** Map tokens to vectors $\mathbf{X} \in \mathbb{R}^{T \times d}$
2. **Convolutional layer:** Apply multiple filter sizes
   - 100 filters of size 3 → $\mathbf{Z}^{(3)} \in \mathbb{R}^{(T-2) \times 100}$
   - 100 filters of size 4 → $\mathbf{Z}^{(4)} \in \mathbb{R}^{(T-3) \times 100}$
   - 100 filters of size 5 → $\mathbf{Z}^{(5)} \in \mathbb{R}^{(T-4) \times 100}$
3. **Pooling layer:** Max-pool each filter's output → 300-dimensional vector
4. **Fully connected:** Concatenate pooled features, apply softmax

**Example:**
- Input: "I love this movie" → embeddings $\mathbf{X} \in \mathbb{R}^{4 \times 300}$
- After convolution and pooling: 300-dimensional feature vector (100 per filter size)
- Output: Softmax over sentiment classes

**Properties:**
- **Parallel:** All positions processed simultaneously (GPU-friendly)
- **Local:** Each filter sees only $k$ consecutive words
- **Translation invariant:** Same filter applied across sequence
- **Fast:** No sequential dependency

**Limitations:**
- **Fixed receptive field:** Cannot capture dependencies beyond filter width
- **No long-range modeling:** Unlike RNNs/Transformers

---

## Limitations of Classical and Early Neural Approaches

While n-gram models, HMMs, and early neural models provided important foundations, they suffer from fundamental limitations:

1. **Fixed Context Window:** N-gram models cannot capture long-range dependencies beyond $n$ tokens

2. **Discrete Representations:** Classical approaches treat words as atomic symbols, failing to capture semantic similarities

3. **Sparsity:** The curse of dimensionality limits effective context size

4. **Independence Assumptions:** The Markov assumptions, while enabling tractability, discard potentially valuable long-range information

5. **Sequential Processing (RNNs):** Even with LSTM/GRU, sequential computation limits parallelization

6. **Local Context (CNNs):** Convolutional models have limited receptive fields

These limitations motivated the development of attention-based models and Transformers, which we examine in later lectures. Modern approaches combine the best of these techniques: continuous representations, parallelizable computation, and unbounded context windows.

---

## Energy-Based Models and Statistical Physics

Energy-Based Models (EBMs) provide an alternative framework for modeling probability distributions. Rather than directly parameterizing probabilities, EBMs assign energies to configurations, with lower energy corresponding to higher probability. This framework, borrowed from statistical physics, provides a unified view of many neural architectures.

### The Energy-Based Framework

**Historical context.** In statistical mechanics, the state of a physical system is described by an energy function. Systems naturally evolve toward low-energy states. Boltzmann and Gibbs formalized how energy relates to probability in thermal equilibrium.

**Definition 6.12 (Energy-based model).** Assign energy $E(\mathbf{x}; \theta) \in \mathbb{R}$ to each configuration $\mathbf{x}$. The probability distribution is given by the **Gibbs (Boltzmann) distribution**:
$$
p(\mathbf{x}; \theta) = \frac{\exp(-E(\mathbf{x}; \theta)/T)}{Z(\theta, T)}
$$

where:
- $T > 0$ is the **temperature** parameter
- $Z(\theta, T) = \sum_{\mathbf{x}} \exp(-E(\mathbf{x}; \theta)/T)$ is the **partition function**

**Convention in machine learning:** Set $T = 1$ or absorb it into the energy:
$$
p(\mathbf{x}; \theta) = \frac{\exp(-E(\mathbf{x}; \theta))}{Z(\theta)}
$$

**Key principle:** Lower energy → higher probability. The system prefers low-energy configurations.

### Connection to Physics

**Analogy to thermodynamics:**

| Physics | Machine Learning |
|---------|------------------|
| Physical state (positions, momenta) | Data point $\mathbf{x}$ |
| Energy $E(\mathbf{x})$ | Energy function $E(\mathbf{x}; \theta)$ |
| Temperature $T$ | Inverse of $\beta$ (sharpness parameter) |
| Thermal equilibrium | Sampling from $p(\mathbf{x})$ |
| Partition function $Z$ | Normalization constant |
| Free energy $F = -T \log Z$ | Negative log-likelihood |

**The Gibbs distribution** arises from the principle of maximum entropy: among all distributions with a given expected energy, the Gibbs distribution has maximum entropy (is least committed beyond the energy constraint).

**Theorem 6.6 (Maximum entropy principle).** The distribution that maximizes entropy $H(p) = -\sum_{\mathbf{x}} p(\mathbf{x}) \log p(\mathbf{x})$ subject to the constraint $\mathbb{E}_p[E(\mathbf{x})] = \bar{E}$ is the Gibbs distribution.

**Proof sketch:** Use Lagrange multipliers. The Lagrangian is:
$$
\mathcal{L} = -\sum_{\mathbf{x}} p(\mathbf{x}) \log p(\mathbf{x}) - \lambda\left(\sum_{\mathbf{x}} p(\mathbf{x}) E(\mathbf{x}) - \bar{E}\right) - \mu\left(\sum_{\mathbf{x}} p(\mathbf{x}) - 1\right)
$$

Taking $\frac{\partial \mathcal{L}}{\partial p(\mathbf{x})} = 0$ yields $p(\mathbf{x}) \propto \exp(-\lambda E(\mathbf{x}))$, which is the Gibbs form with $\lambda = 1/T$.

### The Partition Function Challenge

**Computational bottleneck:** Computing $Z(\theta) = \sum_{\mathbf{x}} \exp(-E(\mathbf{x}; \theta))$ requires summing over all possible configurations.

**Example:** For binary vectors $\mathbf{x} \in \{0,1\}^{100}$, there are $2^{100} \approx 10^{30}$ terms—intractable.

**Implications:**
1. **Exact inference intractable:** Cannot compute $p(\mathbf{x})$ exactly
2. **Training difficult:** Gradient of log-likelihood involves expectations over $p(\mathbf{x})$
3. **Sampling required:** Must use MCMC or variational methods

**Gradient of log-likelihood:**
$$
\nabla_\theta \log p(\mathbf{x}; \theta) = -\nabla_\theta E(\mathbf{x}; \theta) + \mathbb{E}_{\mathbf{x}' \sim p(\cdot; \theta)}[\nabla_\theta E(\mathbf{x}'; \theta)]
$$

The second term (positive phase) requires sampling from the model—a chicken-and-egg problem.

### Approximation Methods

Several approaches circumvent the partition function:

**1. Contrastive Divergence:**
- Replace exact model samples with samples from a few steps of Gibbs sampling initialized at data
- Efficient but biased gradient estimator

**2. Noise Contrastive Estimation (NCE):**
- Reformulate as binary classification between data and noise
- Used in Word2Vec with negative sampling

**3. Score Matching:**
- Match gradients of log-probability rather than probabilities themselves
- Bypasses partition function entirely

---

## Hopfield Networks: Associative Memory as Energy Minimization

Hopfield networks (1982) are one of the earliest and most elegant examples of energy-based neural networks. They implement **content-addressable memory**: given a partial or corrupted pattern, retrieve the complete stored pattern.

### Classical Hopfield Networks

**Setup:** $N$ binary neurons $\mathbf{s} = (s_1, \ldots, s_N)$ where $s_i \in \{-1, +1\}$.

**Definition 6.13 (Hopfield energy function).** For symmetric weight matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$ with $W_{ii} = 0$:
$$
E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T \mathbf{W} \mathbf{s} - \mathbf{b}^T \mathbf{s} = -\frac{1}{2}\sum_{i,j} W_{ij} s_i s_j - \sum_i b_i s_i
$$

**Physical interpretation:**
- $W_{ij}$: interaction energy between neurons $i$ and $j$ (like spin-spin coupling in Ising model)
- $\mathbf{b}_i$: external field acting on neuron $i$
- Negative signs: system minimizes energy (stable states have low energy)

**Gibbs distribution:**
$$
p(\mathbf{s}) = \frac{\exp(-E(\mathbf{s})/T)}{Z} = \frac{\exp\left(\frac{1}{2T}\mathbf{s}^T \mathbf{W} \mathbf{s} + \frac{1}{T}\mathbf{b}^T \mathbf{s}\right)}{Z}
$$

At low temperature ($T \to 0$), the distribution concentrates on energy minima.

### Dynamics: Gradient Descent on Energy

**Asynchronous update rule:** At each step, randomly select neuron $i$ and update:
$$
s_i \leftarrow \text{sign}\left(\sum_j W_{ij} s_j + b_i\right) = \text{sign}(h_i)
$$

where $h_i = \sum_j W_{ij} s_j + b_i$ is the **local field** at neuron $i$.

**Theorem 6.7 (Energy decrease).** Each asynchronous update decreases or maintains energy.

**Proof:** Consider flipping $s_i$ from $-1$ to $+1$ (or vice versa). The energy change is:
$$
\Delta E = E(\mathbf{s}^{\text{new}}) - E(\mathbf{s}^{\text{old}}) = -\Delta s_i \left(\sum_j W_{ij} s_j + b_i\right) = -\Delta s_i \cdot h_i
$$

where $\Delta s_i = s_i^{\text{new}} - s_i^{\text{old}}$.

The update rule sets $s_i^{\text{new}} = \text{sign}(h_i)$. If $h_i > 0$, we set $s_i = +1$:
- If $s_i^{\text{old}} = -1$, then $\Delta s_i = 2$ and $\Delta E = -2h_i < 0$ (energy decreases)
- If $s_i^{\text{old}} = +1$, then $\Delta s_i = 0$ and $\Delta E = 0$ (no change)

Similarly for $h_i < 0$. Thus $\Delta E \leq 0$ always. □

**Corollary 6.1 (Convergence).** Since energy is bounded below and decreases at each step, the network converges to a fixed point (local energy minimum) in finite time.

### Hebbian Learning: Storing Patterns

**Goal:** Store $M$ patterns $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)}\}$ where $\mathbf{x}^{(m)} \in \{-1, +1\}^N$.

**Hebbian rule:** "Neurons that fire together, wire together."
$$
W_{ij} = \frac{1}{N}\sum_{m=1}^{M} x_i^{(m)} x_j^{(m)}
$$

Equivalently, in matrix form:
$$
\mathbf{W} = \frac{1}{N}\sum_{m=1}^{M} \mathbf{x}^{(m)} (\mathbf{x}^{(m)})^T = \frac{1}{N}\mathbf{X}\mathbf{X}^T
$$

where $\mathbf{X} = [\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)}]$.

**Why this works:** If $\mathbf{s} = \mathbf{x}^{(k)}$ (one of the stored patterns), the local field is:
$$
h_i = \sum_j W_{ij} x_j^{(k)} = \frac{1}{N}\sum_{m=1}^{M} \sum_j x_i^{(m)} x_j^{(m)} x_j^{(k)}
$$

For orthogonal patterns ($\mathbf{x}^{(m)} \cdot \mathbf{x}^{(k)} = N \delta_{mk}$):
$$
h_i = \frac{1}{N}\sum_j x_i^{(k)} x_j^{(k)} x_j^{(k)} = x_i^{(k)} \sum_j (x_j^{(k)})^2 = N x_i^{(k)}
$$

Thus $\text{sign}(h_i) = x_i^{(k)}$, so stored patterns are stable fixed points.

### Storage Capacity

**Theorem 6.8 (Hopfield capacity).** A Hopfield network with $N$ neurons can reliably store approximately $M \approx 0.15N$ random patterns.

**Intuition:** As $M$ increases, stored patterns interfere (cross-talk). When $M > 0.15N$, spurious attractors emerge and retrieval becomes unreliable.

**Proof sketch (informal):** The local field has signal term (desired pattern) and noise term (interference from other patterns). Signal scales as $N$, noise scales as $\sqrt{MN}$ (random walk). Reliable retrieval requires signal $\gg$ noise:
$$
N \gg \sqrt{MN} \implies M \ll N
$$

More precise analysis (using statistical mechanics) gives $M \approx 0.138N$.

### Limitations of Classical Hopfield Networks

1. **Low capacity:** Only $0.15N$ patterns
2. **Spurious attractors:** Network can converge to states that aren't stored patterns
3. **Symmetric weights:** Restricts to energy-based dynamics (no feedforward processing)
4. **Binary states:** Limited expressiveness

---

## Modern Hopfield Networks: Exponential Capacity

Ramsauer et al. (2021) showed that using continuous states and softmax nonlinearity dramatically increases capacity.

### Continuous States and Softmax

**Setup:** Continuous state $\mathbf{\xi} \in \mathbb{R}^d$ and stored patterns $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_M] \in \mathbb{R}^{d \times M}$.

**Definition 6.14 (Modern Hopfield energy).**
$$
E(\mathbf{\xi}) = -\text{lse}(\beta \mathbf{X}^T \mathbf{\xi}) + \frac{1}{2}\|\mathbf{\xi}\|^2 + \frac{\beta}{2} \sum_{i=1}^{M} \|\mathbf{x}_i\|^2
$$

where $\text{lse}(\mathbf{z}) = \frac{1}{\beta}\log \sum_i \exp(\beta z_i)$ is the log-sum-exp (smooth maximum).

**Intuition:**
- First term: negative log-sum-exp of similarities to stored patterns (low energy when $\mathbf{\xi}$ is similar to some $\mathbf{x}_i$)
- Second term: regularization (penalizes large $\|\mathbf{\xi}\|$)
- Third term: constant (doesn't affect dynamics)

**Update rule (gradient descent on energy):**
$$
\frac{\partial E}{\partial \mathbf{\xi}} = -\mathbf{X} \text{softmax}(\beta \mathbf{X}^T \mathbf{\xi}) + \mathbf{\xi}
$$

Setting to zero gives the fixed point:
$$
\mathbf{\xi}^* = \mathbf{X} \text{softmax}(\beta \mathbf{X}^T \mathbf{\xi}^*)
$$

**This is exactly the self-attention equation!** With queries $\mathbf{Q} = \mathbf{\xi}$, keys $\mathbf{K} = \mathbf{X}$, values $\mathbf{V} = \mathbf{X}$:
$$
\text{Attention}(\mathbf{\xi}, \mathbf{X}, \mathbf{X}) = \mathbf{X} \text{softmax}(\mathbf{X}^T \mathbf{\xi} / \sqrt{d})
$$

(The $\beta = 1/\sqrt{d}$ scaling from Lecture 08 corresponds to temperature in the Gibbs distribution.)

### Exponential Storage Capacity

**Theorem 6.9 (Exponential capacity).** A modern Hopfield network with $d$-dimensional states can store $M = \exp(d)$ patterns with reliable retrieval.

**Proof sketch:** The softmax concentrates exponentially on the pattern most similar to the query. For $M = \exp(d)$ random patterns, the probability of accidental high similarity between distinct patterns is negligible.

**Comparison:**
- **Classical Hopfield:** $M \approx 0.15N$ (linear in dimension)
- **Modern Hopfield:** $M \approx \exp(d)$ (exponential in dimension)

This exponential improvement explains why Transformers can store vast amounts of information in their parameters.

### Connection to Modern Neural Networks

**1. Attention Mechanisms:**
Modern Hopfield networks ARE attention mechanisms. The update rule is identical to self-attention in Transformers (Lecture 08).

**2. Energy-Based Training:**
Understanding attention as energy minimization provides intuition for why it works and how to design variants.

**3. Memory-Augmented Networks:**
Neural Turing Machines and Memory Networks (Lecture 07) extend these ideas with differentiable read/write operations.

---

## Summary: From Classical to Neural Approaches

We've covered a comprehensive spectrum of language modeling approaches:

**Classical Statistical Models:**
- **N-gram models:** Simple, interpretable, limited by sparsity
- **HMMs:** Structured prediction, efficient via dynamic programming (Viterbi)
- **Perplexity:** Standard evaluation metric

**Recurrent Neural Networks:**
- **Vanilla RNNs:** Sequential processing, vanishing gradients
- **LSTMs:** 3 gates + cell state, additive gradient flow, reliable long-term memory
- **GRUs:** 2 gates, simpler, often comparable performance

**Convolutional Neural Networks:**
- **1D convolutions:** Parallel processing of local patterns
- **Multiple filter sizes:** Capture different n-gram lengths
- **Pooling:** Extract most salient features
- **Use case:** Fast text classification

**Energy-Based Models:**
- **Gibbs distribution:** Connection to statistical physics
- **Partition function challenge:** Intractable normalization
- **Contrastive divergence, NCE:** Approximation methods

**Hopfield Networks:**
- **Classical:** Binary states, $0.15N$ capacity, Hebbian learning
- **Modern:** Continuous states, exponential capacity, equivalent to attention
- **Connection to Transformers:** Self-attention IS a Hopfield network

**Key Insights:**
1. Evolution from discrete to continuous representations
2. From fixed context to arbitrary-length sequences
3. From sequential to parallel computation (CNNs, later Transformers)
4. Energy-based perspective unifies many architectures
5. Hopfield networks anticipated modern attention mechanisms

These approaches laid the groundwork for the attention-based models and Transformers we'll explore in Lectures 7-8.

---

# Appendix

## A.1 MLE Derivation for N-Gram Models

We derive the maximum likelihood estimate for n-gram conditional probabilities from first principles.

**Setup:** Given a training corpus consisting of sequences $\mathcal{D} = \{X^{(1)}, \ldots, X^{(M)}\}$ where each $X^{(m)} = (x_1^{(m)}, \ldots, x_{T_m}^{(m)})$, we seek to estimate:
$$\theta_{c,w} = P(X^{(i)} = w | X^{(i-n+1:i-1)} = c)$$

where $c$ represents a context (sequence of $n-1$ preceding tokens) and $w$ is the next token.

**Likelihood Function:** Under the n-gram assumption, the likelihood of the corpus is:
$$L(\theta) = \prod_{m=1}^{M} P(X^{(m)}; \theta) = \prod_{m=1}^{M} \prod_{i=1}^{T_m} P(x_i^{(m)} | x_{i-n+1:i-1}^{(m)}; \theta)$$

Taking the logarithm:
$$\ell(\theta) = \sum_{m=1}^{M} \sum_{i=1}^{T_m} \log P(x_i^{(m)} | x_{i-n+1:i-1}^{(m)}; \theta)$$

**Reparameterization by Context:** Let $N(c, w)$ denote the count of times context $c$ is followed by word $w$ in the corpus, and $N(c) = \sum_w N(c, w)$ the total count of context $c$. Then:
$$\ell(\theta) = \sum_{c} \sum_{w} N(c, w) \log \theta_{c,w}$$

**Constraint:** For each context $c$, the probabilities must sum to 1:
$$\sum_{w} \theta_{c,w} = 1$$

**Lagrangian:** We form the Lagrangian with multipliers $\lambda_c$ for each context:
$$\mathcal{L}(\theta, \lambda) = \sum_{c} \sum_{w} N(c, w) \log \theta_{c,w} - \sum_{c} \lambda_c \left(\sum_{w} \theta_{c,w} - 1\right)$$

**First-Order Conditions:** Taking the derivative with respect to $\theta_{c,w}$ and setting to zero:
$$\frac{\partial \mathcal{L}}{\partial \theta_{c,w}} = \frac{N(c, w)}{\theta_{c,w}} - \lambda_c = 0$$

Therefore:
$$\theta_{c,w} = \frac{N(c, w)}{\lambda_c}$$

**Solving for Lagrange Multiplier:** Using the constraint $\sum_w \theta_{c,w} = 1$:
$$\sum_w \frac{N(c, w)}{\lambda_c} = 1$$
$$\frac{1}{\lambda_c} \sum_w N(c, w) = 1$$
$$\lambda_c = \sum_w N(c, w) = N(c)$$

**Final MLE:** Substituting back:
$$\hat{\theta}_{c,w} = \frac{N(c, w)}{N(c)} = \frac{\text{Count}(c, w)}{\text{Count}(c)}$$

This is the relative frequency estimator: the proportion of times context $c$ is followed by word $w$.

---

## A.2 MLE Derivation for HMM Emission Probabilities

We derive the maximum likelihood estimates for emission probabilities $\phi_{k,j} = P(X = j | Y = k)$ in an HMM.

**Setup:** Given fully-observed training data $\mathcal{D} = \{(X^{(m)}, Y^{(m)})\}_{m=1}^{M}$ where both tokens and labels are known, we estimate emission parameters.

**Likelihood Function:** The contribution of emission probabilities to the log-likelihood is:
$$\ell_{\text{emission}}(\phi) = \sum_{m=1}^{M} \sum_{i=1}^{T_m} \log P(x_i^{(m)} | y_i^{(m)}; \phi)$$

**Reparameterization by Counts:** Let $N(k, j)$ denote the count of times label $k$ emits token $j$, and $N(k) = \sum_j N(k, j)$ the total count of label $k$:
$$\ell_{\text{emission}}(\phi) = \sum_{k=1}^{K} \sum_{j=1}^{N} N(k, j) \log \phi_{k,j}$$

**Constraint:** For each label $k$, emission probabilities sum to 1:
$$\sum_{j=1}^{N} \phi_{k,j} = 1$$

**Lagrangian:** With multipliers $\mu_k$ for each label:
$$\mathcal{L}(\phi, \mu) = \sum_{k=1}^{K} \sum_{j=1}^{N} N(k, j) \log \phi_{k,j} - \sum_{k=1}^{K} \mu_k \left(\sum_{j=1}^{N} \phi_{k,j} - 1\right)$$

**First-Order Conditions:** Taking derivative with respect to $\phi_{k,j}$:
$$\frac{\partial \mathcal{L}}{\partial \phi_{k,j}} = \frac{N(k, j)}{\phi_{k,j}} - \mu_k = 0$$

Therefore:
$$\phi_{k,j} = \frac{N(k, j)}{\mu_k}$$

**Solving for Lagrange Multiplier:** Using the constraint:
$$\sum_{j=1}^{N} \frac{N(k, j)}{\mu_k} = 1$$
$$\mu_k = \sum_{j=1}^{N} N(k, j) = N(k)$$

**Final MLE:**
$$\hat{\phi}_{k,j} = \frac{N(k, j)}{N(k)} = \frac{\text{Count}(\text{label } k, \text{ token } j)}{\text{Count}(\text{label } k)}$$

**Interpretation:** The MLE for emission probabilities is the relative frequency of token $j$ among all tokens labeled as $k$.

---

## A.3 MLE Derivation for HMM Transition Probabilities

We derive the maximum likelihood estimates for transition probabilities $\lambda_{k,k'} = P(Y^{(i+1)} = k' | Y^{(i)} = k)$.

**Setup:** Given fully-observed training data with label sequences, we estimate transition parameters.

**Likelihood Function:** The contribution of transition probabilities to the log-likelihood is:
$$\ell_{\text{transition}}(\lambda) = \sum_{m=1}^{M} \sum_{i=1}^{T_m-1} \log P(y_{i+1}^{(m)} | y_i^{(m)}; \lambda)$$

**Reparameterization by Counts:** Let $N(k, k')$ denote the count of transitions from label $k$ to label $k'$, and $N(k) = \sum_{k'} N(k, k')$ the total count of label $k$ appearing (except at sequence end):
$$\ell_{\text{transition}}(\lambda) = \sum_{k=1}^{K} \sum_{k'=1}^{K} N(k, k') \log \lambda_{k,k'}$$

**Constraint:** For each label $k$, transition probabilities sum to 1:
$$\sum_{k'=1}^{K} \lambda_{k,k'} = 1$$

**Lagrangian:** With multipliers $\nu_k$ for each source label:
$$\mathcal{L}(\lambda, \nu) = \sum_{k=1}^{K} \sum_{k'=1}^{K} N(k, k') \log \lambda_{k,k'} - \sum_{k=1}^{K} \nu_k \left(\sum_{k'=1}^{K} \lambda_{k,k'} - 1\right)$$

**First-Order Conditions:** Taking derivative with respect to $\lambda_{k,k'}$:
$$\frac{\partial \mathcal{L}}{\partial \lambda_{k,k'}} = \frac{N(k, k')}{\lambda_{k,k'}} - \nu_k = 0$$

Therefore:
$$\lambda_{k,k'} = \frac{N(k, k')}{\nu_k}$$

**Solving for Lagrange Multiplier:** Using the constraint:
$$\sum_{k'=1}^{K} \frac{N(k, k')}{\nu_k} = 1$$
$$\nu_k = \sum_{k'=1}^{K} N(k, k') = N(k)$$

**Final MLE:**
$$\hat{\lambda}_{k,k'} = \frac{N(k, k')}{N(k)} = \frac{\text{Count}(Y^{(i)} = k, Y^{(i+1)} = k')}{\text{Count}(Y^{(i)} = k)}$$

**Interpretation:** The MLE for transition probabilities is the relative frequency of label $k$ being followed by label $k'$.

---

## A.4 Connection to Multinomial Distribution

All three MLE results share a common structure: they are maximum likelihood estimates for multinomial distributions.

**General Pattern:** For a categorical random variable $Z$ with outcomes $\{1, \ldots, K\}$ and probabilities $\{\theta_1, \ldots, \theta_K\}$, given $n$ observations with counts $\{n_1, \ldots, n_K\}$ where $\sum_k n_k = n$:

**Multinomial Likelihood:**
$$L(\theta) = \frac{n!}{n_1! \cdots n_K!} \prod_{k=1}^{K} \theta_k^{n_k}$$

**Log-Likelihood:**
$$\ell(\theta) = \text{const} + \sum_{k=1}^{K} n_k \log \theta_k$$

**MLE:** Subject to $\sum_k \theta_k = 1$:
$$\hat{\theta}_k = \frac{n_k}{n}$$

**Applications:**
1. **N-gram models:** For each context $c$, the next-word distribution is multinomial
2. **HMM emissions:** For each label $k$, the token distribution is multinomial
3. **HMM transitions:** For each label $k$, the next-label distribution is multinomial

This unifying perspective reveals that all three parameter estimation problems reduce to estimating multinomial distributions via relative frequency counts.

---

## A.5 Viterbi Algorithm Derivation

We derive the Viterbi algorithm for finding the most likely label sequence in an HMM, showing how dynamic programming exploits the Markov structure to reduce exponential complexity to polynomial time.

### Problem Setup

**Given:**
- Observed sequence $X^{(1:T)} = (X^{(1)}, \ldots, X^{(T)})$
- HMM parameters: emission probabilities $\boldsymbol{\Phi}$ and transition probabilities $\boldsymbol{\Lambda}$

**Goal:** Find the most likely label sequence:
$$\hat{Y}^{(1:T)} = \arg\max_{Y^{(1:T)}} P(Y^{(1:T)} | X^{(1:T)})$$

### From Posterior to Joint Distribution

By Bayes' rule:
$$P(Y^{(1:T)} | X^{(1:T)}) = \frac{P(X^{(1:T)}, Y^{(1:T)})}{P(X^{(1:T)})}$$

Since the denominator $P(X^{(1:T)})$ does not depend on $Y$, maximizing the posterior is equivalent to maximizing the joint:
$$\hat{Y}^{(1:T)} = \arg\max_{Y^{(1:T)}} P(X^{(1:T)}, Y^{(1:T)})$$

### Log-Space Formulation

Taking logarithms (a monotonic transformation that preserves the argmax):
$$\hat{Y}^{(1:T)} = \arg\max_{Y^{(1:T)}} \log P(X^{(1:T)}, Y^{(1:T)})$$

Under the HMM factorization:
$$\log P(X^{(1:T)}, Y^{(1:T)}) = \sum_{i=1}^{T} \log P(X^{(i)} | Y^{(i)}) + \sum_{i=1}^{T} \log P(Y^{(i)} | Y^{(i-1)})$$

Using our parameter notation:
$$\log P(X^{(1:T)}, Y^{(1:T)}) = \sum_{i=1}^{T} \left[\log \phi_{Y^{(i)}, X^{(i)}} + \log \lambda_{Y^{(i-1)}, Y^{(i)}}\right]$$

where $Y^{(0)} = \langle \text{START} \rangle$ is the initial state.

### The Naive Approach: Exponential Complexity

A naive enumeration would evaluate all possible label sequences:
- There are $K$ choices for each of $T$ positions
- Total sequences: $K^T$
- For each sequence, compute the sum of $T$ terms: $O(T)$
- **Total complexity: $O(TK^T)$** — exponential in sequence length

This is intractable for realistic problems (e.g., $K=45$ POS tags, $T=20$ words gives $45^{20} \approx 10^{33}$ sequences).

### Key Insight: Optimal Substructure

**Lemma A.1 (Optimal Substructure)** Let $\hat{Y}^{(1:T)}$ be an optimal labeling. Then for any position $i < T$, the prefix $\hat{Y}^{(1:i)}$ is an optimal labeling for the subproblem ending at position $i$ with label $\hat{Y}^{(i)}$.

**Proof sketch:** Suppose $\hat{Y}^{(1:i)}$ were not optimal for the subproblem. Then there exists another sequence $Y'^{(1:i)}$ ending in $\hat{Y}^{(i)}$ with higher log-probability. Replacing the prefix in $\hat{Y}^{(1:T)}$ with $Y'^{(1:i)}$ would yield a better complete sequence, contradicting optimality of $\hat{Y}^{(1:T)}$.

This optimal substructure property is the hallmark of problems amenable to dynamic programming.

### Dynamic Programming Recursion

**Definition:** Let $V_{i,k}$ denote the maximum log-probability of any label sequence ending at position $i$ with label $k$:
$$V_{i,k} = \max_{Y^{(1:i-1)}} \left[\sum_{j=1}^{i} \log \phi_{Y^{(j)}, X^{(j)}} + \sum_{j=1}^{i} \log \lambda_{Y^{(j-1)}, Y^{(j)}}\right]$$

subject to $Y^{(i)} = k$.

**Theorem A.1 (Viterbi Recursion)** The values $V_{i,k}$ satisfy the recurrence relation:
$$V_{i,k} = \log \phi_{k, X^{(i)}} + \max_{k' \in \{1,\ldots,K\}} \left[V_{i-1,k'} + \log \lambda_{k',k}\right]$$

with base case $V_{1,k} = \log \phi_{k,X^{(1)}} + \log \lambda_{\langle \text{START} \rangle, k}$.

**Proof:**
By definition, $V_{i,k}$ maximizes over all sequences $Y^{(1:i)}$ ending in state $k$. We can decompose this into:
1. The emission score at position $i$: $\log \phi_{k,X^{(i)}}$
2. The transition score from position $i-1$ to $i$: $\log \lambda_{Y^{(i-1)},k}$
3. The maximum score reaching position $i-1$ in state $Y^{(i-1)}$: $V_{i-1,Y^{(i-1)}}$

The total score for a sequence ending in state $k$ at position $i$ is:
$$\log \phi_{k,X^{(i)}} + \log \lambda_{Y^{(i-1)},k} + V_{i-1,Y^{(i-1)}}$$

To maximize over all such sequences, we take the maximum over all possible previous states $k'$:
$$V_{i,k} = \log \phi_{k,X^{(i)}} + \max_{k'} \left[V_{i-1,k'} + \log \lambda_{k',k}\right]$$

This completes the proof.

### Backpointers for Path Reconstruction

To recover the optimal sequence, we maintain backpointers:
$$B_{i,k} = \arg\max_{k'} \left[V_{i-1,k'} + \log \lambda_{k',k}\right]$$

$B_{i,k}$ stores which previous state $k'$ led to the optimal path ending in state $k$ at position $i$.

### The Complete Algorithm

**Forward Pass (Computing $V$ and $B$):**

**Initialization:**
For $k = 1, \ldots, K$:
$$V_{1,k} = \log \phi_{k,X^{(1)}} + \log \lambda_{\langle \text{START} \rangle, k}$$
$$B_{1,k} = \langle \text{START} \rangle$$

**Recursion:**
For $i = 2, \ldots, T$ and $k = 1, \ldots, K$:
$$V_{i,k} = \log \phi_{k,X^{(i)}} + \max_{k' \in \{1,\ldots,K\}} \left[V_{i-1,k'} + \log \lambda_{k',k}\right]$$
$$B_{i,k} = \arg\max_{k' \in \{1,\ldots,K\}} \left[V_{i-1,k'} + \log \lambda_{k',k}\right]$$

**Termination:**
$$\hat{Y}^{(T)} = \arg\max_{k \in \{1,\ldots,K\}} V_{T,k}$$

**Backward Pass (Path Reconstruction):**

For $i = T-1, T-2, \ldots, 1$:
$$\hat{Y}^{(i)} = B_{i+1, \hat{Y}^{(i+1)}}$$

### Complexity Analysis

**Time Complexity:**
- Forward pass: For each of $T$ positions, compute $K$ states, each requiring a max over $K$ previous states
- Operations: $O(T \cdot K \cdot K) = O(TK^2)$
- Backward pass: $O(T)$
- **Total: $O(TK^2)$** — polynomial instead of exponential

**Space Complexity:**
- Store $V$ and $B$ matrices: $O(TK)$

**Comparison:**
- Naive enumeration: $O(TK^T)$ — exponential
- Viterbi: $O(TK^2)$ — polynomial
- For $K=45$, $T=20$: Naive requires $\sim 10^{33}$ operations; Viterbi requires $\sim 40,000$ operations

### Why Dynamic Programming Works

The Viterbi algorithm succeeds because HMMs satisfy two key properties:

1. **Optimal Substructure:** The optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems:** Many label sequences share common prefixes

Without the Markov assumption, we would need to consider all possible histories, destroying the overlapping subproblem structure. The first-order Markov property ensures that the state at position $i$ summarizes all relevant information from the past, enabling efficient dynamic programming.

### Extension: Max-Marginals via Forward-Backward

**Remark A.1** The Viterbi algorithm finds the single most likely *sequence*:
$$\hat{Y}^{(1:T)} = \arg\max_{Y^{(1:T)}} P(Y^{(1:T)} | X^{(1:T)})$$

An alternative is to find the most likely label at *each position* independently:
$$\hat{Y}^{(i)} = \arg\max_{k} P(Y^{(i)} = k | X^{(1:T)})$$

This requires computing marginal probabilities via the forward-backward algorithm, which uses a similar dynamic programming approach but sums (rather than maximizes) over paths. The forward-backward algorithm has the same $O(TK^2)$ complexity but computes probabilities rather than finding the single best path.

**Trade-off:** Viterbi guarantees a valid label sequence (respecting transition probabilities), while per-position decoding via max-marginals may produce sequences with low joint probability.