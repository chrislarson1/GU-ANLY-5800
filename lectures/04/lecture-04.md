# Lecture 04: Distributional Semantics and Information Retrieval

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

In previous lectures, we represented documents using the bag-of-words model, treating each word as an independent feature. We now examine methods for assigning meaningful weights to these features and for discovering latent semantic structure in text corpora. These techniques form the foundation of information retrieval systems and provide our first glimpse into distributional semantics—the principle that words occurring in similar contexts tend to have similar meanings.

## The Problem of Uniform Representation

**Observation 1.1** The bag-of-words representation treats all words equally, assigning importance solely based on raw frequency counts. This is problematic for several reasons:

1. **High-frequency words carry low information:** Common words like "the," "a," "is" appear frequently across all documents but reveal little about document content
2. **We lack a reliable distance metric:** In the high-dimensional sparse vector space $\mathbb{R}^{|V|}$, standard distance measures become less meaningful
3. **Corpus non-uniformity:** Word distributions follow Zipf's law—a highly skewed distribution where a few words dominate

### Zipf's Law and Word Distributions

**Definition 1.1 (Zipf's Law)** In a corpus, the frequency of a word is inversely proportional to its rank in the frequency table:

$$P(x_j) = \frac{1/j^{\alpha}}{\sum_{i=1}^{|V|} 1/i^{\alpha}}, \quad \alpha > 1$$

where $j$ denotes the rank of word $x_j$ when words are ordered by decreasing frequency.

**Observation 1.2** Taking logarithms of both sides:

$$\log P(x_j) \propto -\alpha \log j$$

This linear relationship in log-log space characterizes heavy-tailed distributions. In practice, $\alpha \approx 1$ for natural language, though exact values vary by corpus.

**Implication:** A small number of high-frequency words dominate the corpus, while most words appear rarely. Raw frequency counts therefore do not distinguish informative from uninformative words.

### Information Theory Perspective

Recall from Lecture 01 that the self-information of event $x$ is $I(x) = -\log P(x)$. This quantity is inversely proportional to probability:

- **High-frequency words** (large $P(x)$) carry **low information** (small $I(x)$)
- **Low-frequency words** (small $P(x)$) carry **high information** (large $I(x)$)

**Motivation:** Many early information retrieval systems were designed to be sensitive to low-frequency words (which are distinctive) and insensitive to high-frequency words (which are generic). This insight motivates term weighting schemes that reweight word counts according to their informativeness.

---

## TF-IDF: Term Frequency - Inverse Document Frequency

The TF-IDF algorithm provides a principled approach to reweighting word counts by considering both local frequency (within a document) and global frequency (across the corpus).

### Definition and Components

**Definition 2.1 (Term Frequency)** For word $j$ in document $i$, the term frequency is simply the raw count:

$$\text{TF}(x)_{i,j} = x_j^{(i)}$$

where $x_j^{(i)}$ denotes the count of word $j$ in document $i$.

**Definition 2.2 (Inverse Document Frequency)** The inverse document frequency of word $j$ across corpus with $M$ documents is:

$$\text{IDF}(x)_j = \left[\sum_{i=1}^{M} \mathbb{I}\{x_j^{(i)} > 0\}\right]^{-1}$$

This counts the number of documents containing word $j$ and takes the reciprocal.

**Definition 2.3 (TF-IDF Weighting)** The TF-IDF weight for word $j$ in document $i$ is:

$$W_{i,j} = \text{TF}(x)_{i,j} \cdot \text{IDF}(x)_j$$

**Interpretation:** Words that appear frequently in a specific document but rarely across the corpus receive high weights. Words that appear in many documents receive low weights regardless of local frequency.

### Logarithmic Variant

**Definition 2.4 (Log-Scaled TF-IDF)** A widely used variant applies logarithmic scaling:

$$W_{i,j} = \left(1 + \log x_j^{(i)}\right) \cdot \log\left(\frac{M}{\sum_{i=1}^{M} \mathbb{I}\{x_j^{(i)} > 0\}}\right)$$

**Rationale:** The logarithmic transformation:
1. Diminishes the impact of extremely high term frequencies
2. Ensures zero weight for absent terms (when $x_j^{(i)} = 0$)
3. Provides a more balanced scale across terms of vastly different frequencies

**Lemma 2.1** The IDF term is maximal when a word appears in only one document and minimal when it appears in all documents.

**Proof:**
- When word $j$ appears in one document: $\text{IDF}(x)_j = M^{-1}$, so $\log \text{IDF}(x)_j = \log M$
- When word $j$ appears in all $M$ documents: $\text{IDF}(x)_j = 1$, so $\log \text{IDF}(x)_j = 0$ □

---

## Document Search Using TF-IDF

**Problem Statement:** Given:
- A TF-IDF weighted corpus matrix $\mathbf{W} \in \mathbb{R}^{M \times N}$ where $M$ is the number of documents and $N$ is vocabulary size
- A query $\mathbf{q} \in \mathbb{Z}^N$ represented as a vector of word frequency counts

Find the most relevant document in the corpus.

### Algorithm: TF-IDF Document Retrieval

**Step 1: Query Representation**

Transform the query into the same weighted space:

$$\mathbf{q}' = \mathbf{q} \odot \text{IDF}(\mathbf{x})$$

where $\odot$ denotes element-wise multiplication and $\text{IDF}(\mathbf{x})$ is the vector of IDF values for all vocabulary words.

**Step 2: Similarity Computation**

For each document $i$, compute similarity between query and document using cosine similarity:

$$S_i(\mathbf{q}) = \text{cos}(\mathbf{W}_i, \mathbf{q}') = \frac{\mathbf{W}_i^T \mathbf{q}'}{\|\mathbf{W}_i\|_2 \|\mathbf{q}'\|_2}$$

where $\mathbf{W}_i$ denotes row $i$ of matrix $\mathbf{W}$.

Expanding:

$$S_i(\mathbf{q}) = \frac{\sum_{j=1}^{N} W_{i,j} q_j' }{\sqrt{\sum_{j=1}^{N} W_{i,j}^2} \sqrt{\sum_{j=1}^{N} (q_j')^2}}$$

**Step 3: Document Selection**

$$\hat{i} = \arg\max_{i \in \{1,\ldots,M\}} S_i(\mathbf{q}')$$

**Remark 3.1** Cosine similarity measures the angle between vectors, making it invariant to document length. This normalization is crucial because longer documents naturally contain more words but aren't necessarily more relevant.

**Remark 3.2** The cosine similarity ranges from 0 (orthogonal vectors, no overlap) to 1 (parallel vectors, perfect alignment). Values closer to 1 indicate higher relevance.

---

## Pointwise Mutual Information

While TF-IDF weights words based on document frequency, pointwise mutual information (PMI) captures co-occurrence patterns between words.

### Definition and Motivation

**Definition 3.1 (Pointwise Mutual Information)** For words $i$ and $j$, the PMI is:

$$\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)}$$

**Interpretation:** PMI measures the extent to which words $i$ and $j$ co-occur more than would be expected by chance under independence.

**Derivation from Information Theory:**

$$\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)} = \log P(i,j) - \log P(i) - \log P(j)$$

$$= \log \frac{P(j|i)P(i)}{P(i)P(j)} = \log \frac{P(j|i)}{P(j)} = \log P(i|j) - \log P(i)$$

**Empirical Estimation:** Given a corpus, we estimate probabilities using counts:

$$\text{PMI}(i,j) = \log\left(\frac{\text{count}(i,j)}{\text{count}(i, \text{all } j)}\right) - \log\left(\frac{\text{count}(\text{all } i, j)}{\text{count}(\text{all } i, \text{all } j)}\right)$$

### Positive PMI (PPMI)

**Definition 3.2 (Positive PMI)** To address the issue that PMI can be negative (indicating words co-occur less than expected), we define:

$$\text{PPMI}(i,j) = \begin{cases}
\text{PMI}(i,j) & \text{if } P(i,j) > P(i)P(j) \\
0 & \text{otherwise}
\end{cases}$$

Equivalently: $\text{PPMI}(i,j) = \max(0, \text{PMI}(i,j))$

**Rationale:** Negative PMI values can arise from sampling noise, especially for rare words. PPMI focuses on positive associations while treating independence and negative associations uniformly.

**Remark 3.3** PMI is related to Zipf's law in that both address the non-uniformity of $P(x)$. High-frequency words tend to have lower PMI values with other words simply because they co-occur with everything.

---

## The Dimensionality Reduction Problem

**Observation 4.1** TF-IDF and PMI weightings do not change the sparsity or dimensionality of our representation. We still operate in $\mathbb{R}^{N}$ where $N = |V|$ can be $10^4$ to $10^6$.

**Goal:** Map weighted word vectors $\mathbf{W} \in \mathbb{R}^{M \times N}$ to a dense representation in a lower-dimensional space:

$$\mathbf{W} \in \mathbb{R}^{M \times N} \longrightarrow \mathbf{U} \in \mathbb{R}^{M \times K}, \quad K \ll N$$

**Motivation:** A dense, low-dimensional representation:
1. Reduces computational and storage requirements
2. Mitigates the curse of dimensionality
3. Can capture latent semantic relationships between words and documents
4. Provides more meaningful distance metrics

### Why distance metrics degrade in sparse high dimensions

**Theorem (Distance concentration; Beyer et al., 1999; Aggarwal et al., 2001).**

Let $\{\mathbf{x}^{(i)}\}_{i=1}^M$ be i.i.d. samples in $\mathbb{R}^N$ from a distribution with independent coordinates satisfying mild regularity conditions (e.g., finite variance and non-degenerate support). For any $\ell_q$ norm with $q \ge 1$, the relative contrast between nearest and farthest neighbor distances
$$
\mathrm{RC}_N = \frac{\mathbb{E}[D_{\max}] - \mathbb{E}[D_{\min}]}{\mathbb{E}[D_{\min}]}
$$
to a random query tends to 0 as $N \to \infty$. Consequently, nearest-neighbor rankings become unstable: almost all points are at nearly the same distance from the query.

**Interpretation:** In high dimensions the variance of pairwise distances collapses relative to their mean, so standard metrics cannot reliably discriminate "near" from "far." Dimensionality reduction mitigates this by concentrating information into fewer, more informative axes.

**Connection to Curse of Dimensionality:** This theorem provides the formal justification for why dimensionality reduction techniques (PCA, truncated SVD) introduced in Lecture 01 are essential. While the Eckart-Young-Mirsky theorem (Lecture 01, Theorem 4.2) guarantees that truncated SVD provides the *optimal* low-rank approximation in terms of reconstruction error, the distance concentration theorem here explains *why* we need dimensionality reduction at all—the $L_p$ norms defined in Lecture 01 (Definition 1.4) become unreliable as dimension grows, making low-dimensional projections mathematical necessities for meaningful distance-based operations, not merely computational conveniences.

**Proposition 4.1 (Sparse Bernoulli model: cosine, Euclidean, Jaccard).**

Let **x**, **y** ∈ {0,1}<sup>N</sup> have i.i.d. coordinates with
P[x<sub>j</sub>=1] = P[y<sub>j</sub>=1] = p<sub>N</sub>, where p<sub>N</sub> = c/N for fixed c > 0 (a stylized sparse bag-of-words model).
Let
- K<sub>x</sub> = ||**x**||<sub>0</sub> (number of nonzeros in **x**)
- K<sub>y</sub> = ||**y**||<sub>0</sub>
- T = ⟨**x**, **y**⟩ (overlap)

Then:
- **E**[T] = N·p<sub>N</sub><sup>2</sup> = c²/N → 0, and T → 0 in probability.
- Cosine similarity:
  cos(**x**, **y**) = T / sqrt(K<sub>x</sub> K<sub>y</sub>) → 0 in probability.
- Jaccard similarity:
  J = T / (K<sub>x</sub> + K<sub>y</sub> - T) → 0, since numerator vanishes while denominator stays Θ(1).
- If we ℓ₂-normalize to
  **x̂** = **x** / ||**x**||<sub>2</sub>, **ŷ** = **y** / ||**y**||<sub>2</sub>, then
  ||**x̂** - **ŷ**||<sub>2</sub><sup>2</sup> = 2(1 - cos(**x**, **y**)) → 2,
  so pairwise distances concentrate near √2 with vanishing variance.

**Proof sketch.**
With p<sub>N</sub> = c/N, we have K<sub>x</sub>, K<sub>y</sub> → Poisson(c) in distribution, and T ~ Binomial(N, p<sub>N</sub><sup>2</sup>) with mean c²/N; thus T → 0 in probability while K<sub>x</sub>, K<sub>y</sub> remain Θ<sub>p</sub>(1). Each bullet then follows by Slutsky’s theorem and the identity relating squared Euclidean distance of normalized vectors to cosine similarity.

**Consequence:** In sparse high-dimensional text vectors, most document pairs are almost orthogonal (cosine $\approx 0$), Jaccard overlap is almost always zero, and normalized Euclidean distances are nearly identical. Distances therefore provide little discriminatory power without dimensionality reduction or appropriate reweighting.

---

## Matrix Factorization Methods

Matrix factorization techniques decompose the term-document matrix into lower-dimensional factors that capture latent structure.

### Truncated Singular Value Decomposition

**Theorem 4.1 (Truncated SVD)** Given matrix $\mathbf{W} \in \mathbb{R}^{M \times N}$, the rank-$K$ truncated SVD is:

$$\mathbf{W} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{M \times K}$ with orthonormal columns
- $\mathbf{\Sigma} \in \mathbb{R}^{K \times K}$ diagonal with $\sigma_1 \geq \cdots \geq \sigma_K > 0$
- $\mathbf{V} \in \mathbb{R}^{N \times K}$ with orthonormal columns

**Interpretation:**
- Each column of $\mathbf{V}$ represents a "latent concept" as a combination of words
- Each row of $\mathbf{U}$ represents a document as a combination of these latent concepts
- The singular values in $\mathbf{\Sigma}$ weight the importance of each concept

**Selection of $K$:** Common approaches include:
1. **Elbow method:** Plot explained variance vs. $K$ from the full SVD and select $K$ at the "elbow"
2. **Fixed dimensionality:** Choose $K \ll N$ (e.g., $K = 100$ or $K = 300$ are common)
3. **Variance threshold:** Select smallest $K$ capturing desired fraction of total variance

**Computational Note:** Efficient algorithms exist for computing truncated SVD without computing the full decomposition, making this tractable even for large matrices.

### Latent Semantic Analysis (LSA)

**Definition 4.2 (LSA)** Latent Semantic Analysis applies truncated SVD to a TF-IDF weighted term-document matrix to obtain a low-dimensional semantic space.

**Algorithm: LSA for Document Retrieval**

Given weighted corpus matrix $\mathbf{W} \in \mathbb{R}^{M \times N}$:

**Step 1:** Compute truncated SVD:
$$\mathbf{W} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

**Step 2:** Represent documents in the latent space:
$$\mathbf{U} = \mathbf{W} \mathbf{V} \mathbf{\Sigma}^{-1}$$

where $\mathbf{U}$ is the compressed corpus with documents as rows.

**Remark 4.1** The columns of $\mathbf{V}$ form an orthonormal basis for the latent space. The transformation $\mathbf{V}\mathbf{\Sigma}^{-1}$ projects from the original $N$-dimensional word space to the $K$-dimensional latent space.

**Step 3:** Project query $\mathbf{q}^{(w)} \in \mathbb{R}^N$ (weighted) to latent space:
$$\mathbf{q}^{(\ell)} = \mathbf{q}^{(w)} \mathbf{V} \mathbf{\Sigma}^{-1} \in \mathbb{R}^K$$

**Step 4:** Compute cosine similarity in latent space:
$$S_i(\mathbf{q}^{(\ell)}) = \frac{\mathbf{u}_i^T \mathbf{q}^{(\ell)}}{\|\mathbf{u}_i\|_2 \|\mathbf{q}^{(\ell)}\|_2}$$

where $\mathbf{u}_i$ is row $i$ of $\mathbf{U}$.

**Step 5:** Select most relevant document:
$$\hat{i} = \arg\max_{i \in \{1,\ldots,M\}} S_i(\mathbf{q}^{(\ell)})$$

**Key Advantage:** LSA can capture semantic relationships beyond exact word matches. Documents can be retrieved even if they don't share query terms, provided they share latent semantic concepts.

### Non-Negative Matrix Factorization (NMF)

**Definition 4.3** Non-Negative Matrix Factorization seeks a factorization:

$$\mathbf{W} \approx \mathbf{U} \mathbf{V}$$

where $\mathbf{U} \in \mathbb{R}^{M \times K}$, $\mathbf{V} \in \mathbb{R}^{K \times N}$, and $\mathbf{U}, \mathbf{V} \geq 0$ (all entries non-negative).

**Motivation:** Unlike SVD, which can produce negative coefficients, NMF maintains interpretability when dealing with inherently non-negative data (like word counts).

**Observation 4.2** The columns of $\mathbf{V}$ represent basis vectors in the original $N$-dimensional word space. Each basis vector is a linear combination of words with non-negative weights, making them interpretable as "topics."

**Example:** A column of $\mathbf{V}$ might be:
$$\mathbf{v}_1^T = (0.5)\text{swim} + (0.7)\text{sun} + (1.5)\text{tan} + (0.2)\text{park} + \ldots$$

This could equivalently be represented as:
$$(0.5)\text{swim} + (1.4)\text{sun} + (0.0)\text{tan} + (0.6)\text{park} + \ldots$$

The non-negative constraint ensures these combinations remain interpretable as weighted bags of words.

**Optimization:** NMF is typically solved via alternating optimization. The objective function is:

$$\hat{\mathbf{U}}, \hat{\mathbf{V}} = \arg\min_{\mathbf{U},\mathbf{V} \geq 0} L(\mathbf{W}; \mathbf{U}, \mathbf{V}) + \alpha\|\mathbf{U}\|_F^2 + \alpha\|\mathbf{V}\|_F^2 + \rho(\text{sparsity terms})$$

where $L$ is a reconstruction loss (typically Frobenius: $L_F = \|\mathbf{W} - \mathbf{U}\mathbf{V}\|_F^2$) or KL-divergence:

$$L_{KL} = \sum_{i=1}^{M} \sum_{j=1}^{N} W_{ij}\left[\log\frac{W_{ij}}{(\mathbf{U}\mathbf{V})_{ij}}\right] - W_{ij} + (\mathbf{U}\mathbf{V})_{ij}$$

**Comparison with LSA:**

| Property | LSA (Truncated SVD) | NMF |
|----------|---------------------|-----|
| Orthogonality | Factors orthogonal | No orthogonality constraint |
| Sign | Can be negative | Non-negative only |
| Interpretability | Latent dimensions abstract | Topics more interpretable |
| Uniqueness | Unique (up to sign/rotation) | Not unique |
| Computation | Closed form (iterative algorithms fast) | Iterative optimization required |

---

## Topic Modeling Framework

**Definition 5.1** Topic modeling is a family of methods that discover an "intermediate" representation separating documents from words. The intermediary is referred to as "topics."

**Conceptual Framework:**

$$\text{Document} \longleftrightarrow \text{Topic} \longleftrightarrow \text{Word}$$

Each document is represented as a distribution over topics, and each topic is represented as a distribution over words.

**Observation 5.1** We have already encountered two topic modeling approaches:

1. **LSA:** $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
   - Topics are columns of $\mathbf{V}$ (right singular vectors)
   - Each topic is a linear combination of all words in vocabulary

2. **NMF:** $\mathbf{X} = \mathbf{U} \mathbf{V}$
   - Topics are rows of $\mathbf{V}$
   - Each topic is a non-negative weighted combination of words

**Common Structure:** In both cases:
- $\mathbf{U}$ represents documents in topic space
- $\mathbf{V}$ represents topics in word space
- The rank $K$ determines the number of topics

**Distinction from Classification:** Unlike supervised classification where categories are predefined, topic modeling discovers latent thematic structure in an unsupervised manner. Topics emerge from word co-occurrence patterns rather than being specified *a priori*.

---

## Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA), introduced by Blei, Ng, and Jordan (2003), represents a principled generative probabilistic model for topic modeling. Unlike LSA and NMF, which are primarily algebraic methods, LDA provides a full probabilistic framework with well-defined generative semantics.

### Generative Model

**Definition 5.2 (LDA Generative Process)** For a corpus of $M$ documents with vocabulary size $N$ and $K$ topics, LDA assumes the following generative process for document $d$:

1. **Choose document-topic distribution:** $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$
   - $\boldsymbol{\theta}_d$ is a probability distribution over $K$ topics
   - $\theta_{d,k}$ represents the proportion of document $d$ devoted to topic $k$

2. **For each word $w_{d,i}$ in document $d$:**
   - Choose topic assignment: $z_{d,i} \sim \text{Categorical}(\boldsymbol{\theta}_d)$
   - Choose word from topic: $w_{d,i} \sim \text{Categorical}(\boldsymbol{\phi}_{z_{d,i}})$

where $\boldsymbol{\phi}_k$ is the word distribution for topic $k$, with $\boldsymbol{\phi}_k \sim \text{Dir}(\boldsymbol{\beta})$.

**Key Insight:** Documents are mixtures of topics, and topics are mixtures of words. Each document has its own topic proportions $\boldsymbol{\theta}_d$, but all documents share the same set of topics $\{\boldsymbol{\phi}_k\}_{k=1}^K$.

### The Dirichlet Distribution

**Definition 5.3 (Dirichlet Distribution)** The Dirichlet distribution with parameter $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$ has density:

$$p(\boldsymbol{\theta}; \boldsymbol{\alpha}) = \frac{\Gamma\left(\sum_{k=1}^{K} \alpha_k\right)}{\prod_{k=1}^{K} \Gamma(\alpha_k)} \prod_{k=1}^{K} \theta_k^{\alpha_k - 1}$$

where $\theta_k \geq 0$ and $\sum_{k=1}^{K} \theta_k = 1$.

**Key Property:** The Dirichlet distribution is conjugate to the categorical distribution, enabling tractable posterior updates during inference.

**Interpretation for LDA:** The hyperparameter $\boldsymbol{\alpha}$ controls topic sparsity:
- Small $\alpha_k$ (e.g., $\alpha_k < 1$): Documents tend to focus on few topics
- Large $\alpha_k$ (e.g., $\alpha_k > 1$): Documents spread probability across many topics

### Inference

**Goal:** Compute the posterior distribution of latent variables given observed words:

$$p(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi} \mid \mathbf{W}; \boldsymbol{\alpha}, \boldsymbol{\beta})$$

where $\mathbf{Z} = \{z_{d,i}\}$ are topic assignments, $\boldsymbol{\Theta} = \{\boldsymbol{\theta}_d\}$ are document-topic distributions, and $\boldsymbol{\Phi} = \{\boldsymbol{\phi}_k\}$ are topic-word distributions.

**Challenge:** This posterior is intractable due to the coupling between variables. Exact inference requires summing over $K^{\sum_d n_d}$ possible topic assignments, which is computationally infeasible.

**Theorem 5.2 (Collapsed Gibbs Sampling for LDA)** By integrating out $\boldsymbol{\Theta}$ and $\boldsymbol{\Phi}$ (exploiting Dirichlet-Multinomial conjugacy), we obtain the conditional distribution for topic assignment $z_{d,i}$:

#### Gibbs Sampling

**Theorem 5.1 (Collapsed Gibbs Sampling)** By exploiting Dirichlet-Multinomial conjugacy, we can integrate out $\boldsymbol{\Theta}$ and $\boldsymbol{\Phi}$ to obtain:

$$p(z_{d,i} = k \mid \mathbf{Z}_{-(d,i)}, \mathbf{W}) \propto \underbrace{\left(n_{d,k}^{-(d,i)} + \alpha_k\right)}_{\text{document preference}} \cdot \underbrace{\frac{n_{k,w_{d,i}}^{-(d,i)} + \beta_{w_{d,i}}}{\sum_{v=1}^{N} (n_{k,v}^{-(d,i)} + \beta_v)}}_{\text{topic preference}}$$

where $n_{d,k}^{-(d,i)}$ is the count of words in document $d$ assigned to topic $k$ (excluding current position), and $n_{k,v}^{-(d,i)}$ is the count of word $v$ assigned to topic $k$ across the corpus (excluding current position).

**Algorithm:** Iteratively sample new topic assignments for each word token according to the conditional distribution above. After convergence, estimate:

$$\hat{\theta}_{d,k} = \frac{n_{d,k} + \alpha_k}{\sum_{k'} (n_{d,k'} + \alpha_{k'})}, \quad \hat{\phi}_{k,v} = \frac{n_{k,v} + \beta_v}{\sum_{v'} (n_{k,v'} + \beta_{v'})}$$

#### Variational Inference

**Approach:** Approximate the intractable posterior with a simpler distribution $q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi})$ that factorizes completely (mean-field approximation).

**Objective:** Maximize the Evidence Lower Bound (ELBO):

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{W}, \mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi})] - \mathbb{E}_q[\log q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi})] \leq \log p(\mathbf{W})$$

The ELBO is optimized via coordinate ascent, iteratively updating variational parameters $\boldsymbol{\gamma}_d$ (for document-topic distributions) and $\boldsymbol{\lambda}_k$ (for topic-word distributions) until convergence.

**Trade-off:** Variational inference is typically faster than Gibbs sampling but makes stronger independence assumptions.

### Comparison with LSA and NMF

| Property | LSA | NMF | LDA |
|----------|-----|-----|-----|
| **Framework** | Linear algebra (SVD) | Optimization | Probabilistic generative model |
| **Parameters** | Real-valued | Non-negative | Probability distributions |
| **Inference** | Closed form | Alternating optimization | Approximate (Gibbs/Variational) |
| **Interpretability** | Abstract factors | Non-negative combinations | Probabilistic semantics |
| **Uncertainty** | None | None | Full posterior distribution |
| **Theoretical foundation** | Eckart-Young theorem | Optimization theory | Bayesian statistics |

**Summary:** LDA provides the most complete probabilistic framework with well-defined generative semantics and uncertainty quantification, at the cost of requiring approximate inference. LSA offers computational simplicity with closed-form solutions. NMF balances interpretability with non-negative constraints.

---

## References

**Dimensionality Reduction and Distance Metrics:**
- K. Beyer, J. Goldstein, R. Ramakrishnan, U. Shaft (1999). "When Is Nearest Neighbor Meaningful?" In: *ICDT*.
- C. C. Aggarwal, A. Hinneburg, D. A. Keim (2001). "On the Surprising Behavior of Distance Metrics in High Dimensional Space." In: *ICDT*.

**Topic Modeling:**
- D. M. Blei, A. Y. Ng, M. I. Jordan (2003). "Latent Dirichlet Allocation." *Journal of Machine Learning Research* 3:993–1022.
- T. L. Griffiths, M. Steyvers (2004). "Finding Scientific Topics." *PNAS* 101:5228–5235.
- D. M. Blei, J. D. Lafferty (2006). "Dynamic Topic Models." In: *ICML*.
- D. M. Blei, J. D. McAuliffe (2008). "Supervised Topic Models." In: *NIPS*.
