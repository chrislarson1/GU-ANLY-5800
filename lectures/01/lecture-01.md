# Lecture 01: Mathematical Foundations

*Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

## Vector Spaces

We begin with the fundamental mathematical structures underlying machine learning algorithms. The standard representation of data in machine learning employs matrices where observations correspond to rows and features to columns.

**Definition 1.1** Let $\mathbf{X} \in \mathbb{R}^{M \times N}$ denote our data matrix:

$$
\mathbf{X} = \begin{bmatrix}
x_{1,1} & \dots & x_{1,N} \\
\vdots & \ddots & \vdots \\
x_{M,1} & \dots & x_{M,N}
\end{bmatrix}
$$

where $x_{i,j}$ represents the $j$-th feature value of the $i$-th observation, with $M$ observations and $N$ features.

The convention aligns with the mathematical structure of linear transformations and optimization procedures that form the algorithmic backbone of machine learning. It is worth noting that some other treatements represent data as the transpose of $\mathbf{X}$; all of the results shown here are invariant to this choice of convention.

### Inner Products and Geometric Structure

**Definition 1.2** For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{N}$, the inner product is:

$$\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^{N} a_i b_i$$

The inner product induces a geometric structure on our feature space. It provides both a notion of angle between vectors via $\cos \theta = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\Vert\mathbf{a}\Vert \cdot \Vert\mathbf{b}\Vert}$ and serves as the fundamental operation in linear models.

**Definition 1.3** The outer product $\mathbf{a} \otimes \mathbf{b} = \mathbf{a} \mathbf{b}^T$ yields a rank-1 matrix that captures the interaction structure between vector components.

### Norms and Metric Structure

A norm on a vector space provides a notion of magnitude. For our purposes, we require the standard axioms:

**Definition 1.4** A function $\Vert \cdot \Vert: \mathbb{R}^{N} \to \mathbb{R}_+$ is a norm if:
1. $\Vert\mathbf{x}\Vert = 0 \iff \mathbf{x} = \mathbf{0}$ (positive definiteness)
2. $\Vert\alpha \mathbf{x}\Vert = |\alpha| \cdot \Vert\mathbf{x}\Vert$ for all $\alpha \in \mathbb{R}$ (homogeneity)
3. $\Vert\mathbf{x} + \mathbf{y}\Vert \leq \Vert\mathbf{x}\Vert + \Vert\mathbf{y}\Vert$ (triangle inequality)

The $L_p$ norms form a parametric family:

$$\Vert\mathbf{x}\Vert_p = \left(\sum_{i=1}^{N} |x_i|^p\right)^{1/p}$$

Of particular importance are $L_1$ (Manhattan) and $L_2$ (Euclidean) norms, which induce different geometric properties and optimization behavior.

### Linear Transformations

**Definition 1.5** A mapping $T: \mathbb{R}^{N} \to \mathbb{R}^M$ is linear if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{N}$ and scalars $\alpha, \beta \in \mathbb{R}$,

$$T(\alpha \, \mathbf{x} + \beta \, \mathbf{y}) = \alpha \, T(\mathbf{x}) + \beta \, T(\mathbf{y}).$$

In finite-dimensional spaces, every linear transformation admits a matrix representation: there exists $\mathbf{A} \in \mathbb{R}^{M \times N}$ such that $T(\mathbf{x}) = \mathbf{A}\,\mathbf{x}$.

Given $\mathbf{A} = [a_{i,j}]_{i=1..M,\, j=1..N}$ and $\mathbf{x} = (x_1,\ldots,x_N)^T$, the action of $\mathbf{A}$ on $\mathbf{x}$ produces $\mathbf{y} = \mathbf{A}\mathbf{x} \in \mathbb{R}^M$ with components

$$y_i = \sum_{j=1}^{N} a_{i,j} \, x_j, \quad i = 1,\ldots,M.$$

Equivalently, letting $\mathbf{a}^{(j)}$ denote the $j$-th column of $\mathbf{A}$,

$$\mathbf{A}\mathbf{x} = \sum_{j=1}^{N} x_j \, \mathbf{a}^{(j)}.$$

### Matrix Decompositions

#### Singular Value Decomposition

The Singular Value Decomposition represents one of the most fundamental results in linear algebra, with origins tracing back to Eugenio Beltrami's work in the 1870s, though the modern formulation emerged through contributions from multiple mathematicians including Camille Jordan, Léon Autonne, and Carl Eckart.

**Theorem 4.1 (Singular Value Decomposition)** For any matrix $\mathbf{X} \in \mathbb{R}^{M \times N}$, there exist orthogonal matrices $\mathbf{U} \in \mathbb{R}^{M \times M}$ and $\mathbf{V} \in \mathbb{R}^{N \times N}$ such that:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

where $\mathbf{\Sigma} \in \mathbb{R}^{M \times N}$ is a rectangular diagonal matrix with non-negative entries $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$, where $r = \min(M,N)$.

The columns of $\mathbf{U}$ are called left singular vectors, the columns of $\mathbf{V}$ are right singular vectors, and the diagonal entries $\sigma_i$ are singular values.

**Geometric Interpretation:** The SVD provides a canonical decomposition of any linear transformation. The transformation $\mathbf{X}$ can be understood as a sequence of three operations:
1. $\mathbf{V}^T$: rotation in the domain space
2. $\mathbf{\Sigma}$: scaling along principal axes
3. $\mathbf{U}$: rotation in the codomain space

**Construction via Eigendecomposition:** The SVD can be constructed through eigendecompositions of the Gram matrices:
- $\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$
- $\mathbf{X}\mathbf{X}^T = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{U}^T$

The singular values are $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are the eigenvalues of $\mathbf{X}^T\mathbf{X}$.

**Rank and Approximation:** The rank of $\mathbf{X}$ equals the number of non-zero singular values. For rank $k$ approximation:

$$\mathbf{X}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Theorem 4.2 Eckart-Young-Mirsky**

Among all rank $k$ matrices, $\mathbf{X}_k$ minimizes both $\Vert\mathbf{X} - \mathbf{B}\Vert_F$ and $\Vert\mathbf{X} - \mathbf{B}\Vert_2$ for any matrix $\mathbf{B}$ of rank at most $k$.

#### Eigendecomposition and Spectral Theory

**Definition 4.3** For square matrix $\mathbf{X} \in \mathbb{R}^{N \times N}$, scalar $\lambda$ is an eigenvalue with corresponding eigenvector $\mathbf{v} \neq \mathbf{0}$ if:
$$\mathbf{X}\mathbf{v} = \lambda \mathbf{v}$$

Eigenvalues are roots of the characteristic polynomial $\det(\mathbf{X} - \lambda \mathbf{I}) = 0$.

**Theorem 4.3 Spectral Decomposition**

A symmetric matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$ can be diagonalized as:
$$\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$

where $\mathbf{Q}$ contains orthonormal eigenvectors and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_N)$ contains eigenvalues.

**Key Properties for Symmetric Matrices:**
1. All eigenvalues are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. The matrix is positive definite if and only if all eigenvalues are positive

**Geometric Interpretation:** Eigendecomposition reveals the principal axes of symmetric linear transformations. Each eigenvector defines a direction along which the transformation acts as pure scaling by the corresponding eigenvalue.

#### Principal Component Analysis

PCA emerges naturally from the eigendecomposition of covariance matrices, providing optimal low-dimensional representations under the $L_2$ norm.

**Problem Setup:** Given centered data matrix $\mathbf{X} \in \mathbb{R}^{M \times N}$ (rows are observations), the sample covariance matrix is:

$$\mathbf{C} = \frac{1}{M-1}\mathbf{X}^T\mathbf{X}$$

**Theorem 4.4 PCA Eigendecomposition**

The principal components are eigenvectors of the covariance matrix $\mathbf{C}$, ordered by decreasing eigenvalues.

**Derivation:** We seek direction $\mathbf{w}$ that maximizes projected variance:

$$\max_{\mathbf{w}} \text{Var}(\mathbf{X}\mathbf{w}) = \max_{\mathbf{w}} \mathbf{w}^T\mathbf{C}\mathbf{w} \quad \text{subject to } \Vert\mathbf{w}\Vert_2 = 1$$

Using Lagrange multipliers:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

Taking the derivative and setting to zero:

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

This yields the eigenvalue equation: $\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$

The maximum variance is achieved by the eigenvector corresponding to the largest eigenvalue.

#### The SVD-PCA Connection

**Theorem 4.5** For centered data matrix $\mathbf{X}$, PCA can be computed directly via SVD without forming the covariance matrix.

**Proof:** Let $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ be the SVD. Then:

$$\mathbf{C} = \frac{1}{M-1}\mathbf{X}^T\mathbf{X} = \frac{1}{M-1}\mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$$

The columns of $\mathbf{V}$ are eigenvectors of $\mathbf{C}$ with eigenvalues:

$$\lambda_i = \frac{\sigma_i^2}{M-1}$$

**Computational Advantages:**
1. **Numerical stability:** Avoids forming $\mathbf{X}^T\mathbf{X}$, which is often ill-conditioned
2. **Efficiency:** SVD algorithms are often faster than eigendecomposition for rectangular matrices
3. **Direct interpretation:** Singular vectors directly provide principal components

**Relationship Summary:**
- **Eigendecomposition:** Applies to square, symmetric matrices; reveals intrinsic geometric structure
- **SVD:** Applies to any rectangular matrix; provides most general factorization
- **PCA:** Statistical technique using eigendecomposition of covariance matrices, efficiently computed via SVD

In machine learning applications:
- SVD enables dimensionality reduction and matrix completion
- PCA provides optimal linear projections for data compression
- Both preserve maximum variance under orthogonality constraints
- The choice between direct eigendecomposition and SVD-based computation often depends on numerical considerations and data characteristics

---

## Probability Theory

Machine learning algorithms operate under uncertainty. Probability theory provides the mathematical framework for reasoning about random phenomena and forms the theoretical basis for statistical learning.

### Random Variables and Distributions

**Definition 2.1** A random variable $X$ is a measurable function from a probability space to a measurable space. For discrete random variables, we work with probability mass functions (PMFs) $P: \mathcal{X} \to [0,1]$ satisfying:
1. $P(x) \geq 0$ for all $x \in \mathcal{X}$
2. $\sum_{x \in \mathcal{X}} P(x) = 1$

**Definition 2.2** The Bernoulli distribution with parameter $\theta \in [0,1]$ has PMF:

$$P(Y = y) = \theta^y(1-\theta)^{1-y}, \quad y \in \{0,1\}$$

For continuous random variables, we employ probability density functions satisfying analogous conditions with integration replacing summation.

**Definition 2.3** The multivariate Gaussian distribution with parameters $\boldsymbol{\mu} \in \mathbb{R}^{N}$ and $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ (positive definite) has density:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{N/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

### Fundamental Rules

**Theorem 2.1 Product Rule**

For random variables $X$ and $Y$:

$$
P(X,Y) = P(Y|X)P(X) = P(X|Y)P(Y)
$$

**Corollary 2.1 Bayes' Theorem**

$$
P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}
$$

This provides the mathematical foundation for Bayesian inference and posterior probability computation.

**Definition 2.4** Random variables $X$ and $Y$ are independent if $P(X,Y) = P(X)P(Y)$. They are conditionally independent given $Z$ if $P(X,Y|Z) = P(X|Z)P(Y|Z)$.

---

### Information Theory

Information theory provides a mathematical framework for quantifying uncertainty and measuring the "surprise" contained in probabilistic events. We begin by defining information in terms of the properties any measure of it should obey. From there, we trace Shannon's line of reasoning that led him to the notion of information *entropy*.

#### Self-Information: Quantifying Surprise

Consider an event $x$ with probability $P(x)$. Intuitively, we expect that:
1. Highly probable events should convey little information
2. Rare events should convey substantial information
3. Information should combine additively for independent events

These intuitions lead us to define self-information as:

**Definition 3.1** The self-information of event $x$ is:

$$I(x) = -\log P(x)$$

Let us examine the boundary behavior of this function:

**Case 1:** $P(x) = 1$ (certain event)

$$I(x) = -\log(1) = 0$$
A certain event provides zero information—we learn nothing new.

**Case 2:** $P(x) \to 0$ (impossible event)

$$I(x) = -\log(P(x)) \to +\infty$$

An impossible event, if observed, would provide infinite information.

**Case 3:** $P(x) = 0.5$ (maximum uncertainty for binary event)

$$I(x) = -\log(0.5) = \log(2) \approx 1.44 \text{ nats}$$

The logarithmic structure ensures that independent events combine additively:

$$I(x,y) = -\log P(x,y) = -\log P(x) - \log P(y) = I(x) + I(y)$$

when $x$ and $y$ are independent.

#### Shannon Entropy: Expected Information Content

While self-information characterizes individual events, we often need to characterize entire probability distributions. This naturally leads to considering the expected information content.

**Definition 3.2** The Shannon entropy of distribution $P$ is:

$$H(P) = \mathbb{E}_{X \sim P}[I(X)] = -\sum_{x} P(x) \log P(x)$$

Entropy measures the average uncertainty in a distribution. Let us examine limiting cases:

**Case 1:** Deterministic distribution $P(x_0) = 1, P(x) = 0$ for $x \neq x_0$

$$H(P) = -(1 \cdot \log 1 + 0 \cdot \log 0) = 0$$
No uncertainty exists—we always know the outcome.

**Case 2:** Uniform distribution over $n$ outcomes: $P(x_i) = \frac{1}{n}$ for all $i$

$$H(P) = -\sum_{i=1}^{n} \frac{1}{n} \log \frac{1}{n} = \log n$$
Maximum uncertainty—all outcomes equally likely.

**Theorem 3.1** For a discrete distribution over $n$ outcomes, $H(P) \leq \log n$ with equality if and only if $P$ is uniform.

The uniform distribution maximizes entropy among all distributions with the same support.

#### Kullback-Leibler Divergence: Measuring Distributional Distance

Often we need to compare two probability distributions. The KL divergence quantifies how much one distribution differs from another.

**Definition 3.3** The Kullback-Leibler divergence from distribution $Q$ to distribution $P$ is:

$$D_{KL}(P \parallel Q) = \mathbb{E}_{X \sim P}\left[\log \frac{P(X)}{Q(X)}\right] = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

We can rewrite this as:

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log P(x) - \sum_{x} P(x) \log Q(x) = -H(P) + H(P,Q)$$

where $H(P,Q) = -\sum_{x} P(x) \log Q(x)$ is the cross-entropy.

**Key Properties:**

**Non-negativity:** $D_{KL}(P \parallel Q) \geq 0$ with equality if and only if $P = Q$ almost everywhere.

**Proof sketch:** By Jensen's inequality applied to the concave logarithm function:

$$D_{KL}(P \parallel Q) = -\sum_{x} P(x) \log \frac{Q(x)}{P(x)} \geq -\log \sum_{x} P(x) \frac{Q(x)}{P(x)} = -\log \sum_{x} Q(x) = 0$$

**Asymmetry:** Generally $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$

Let us examine boundary behavior:

**Case 1:** $P = Q$ (identical distributions)

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log 1 = 0$$
No divergence between identical distributions.

**Case 2:** $P(x_0) > 0$ but $Q(x_0) = 0$ for some $x_0$

$$D_{KL}(P \parallel Q) = +\infty$$
Infinite divergence when $P$ assigns probability to events that $Q$ considers impossible.

#### Cross-Entropy: The Connection to Machine Learning

**Definition 3.4** The cross-entropy between distributions $P$ and $Q$ is:
$$H(P,Q) = H(P) + D_{KL}(P \parallel Q) = -\sum_{x} P(x) \log Q(x)$$

In machine learning contexts:
- $P$ represents the true distribution (often empirical distribution from data)
- $Q$ represents our model's predicted distribution
- Minimizing $H(P,Q)$ with respect to $Q$ is equivalent to minimizing $D_{KL}(P \parallel Q)$

Thus, cross-entropy as the natural loss function for probabilistic classification tasks. When we minimize cross-entropy loss, we are effectively making our model's predictions as close as possible to the observed data distribution as measured by KL divergence.

---

## Maximum Likelihood Estimation

**Definition 4.1** Given independent observations $\mathbf{x}_1, \ldots, \mathbf{x}_n$ from distribution $f(\cdot; \boldsymbol{\theta})$, the likelihood function is:

$$L(\boldsymbol{\theta}) = \prod_{i=1}^{n} f(\mathbf{x}_i; \boldsymbol{\theta})$$

**Definition 4.2** The maximum likelihood estimator (MLE) is:

$$\hat{\boldsymbol{\theta}}_{MLE} = \arg\max_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^{n} \log f(\mathbf{x}_i; \boldsymbol{\theta})$$

The logarithmic transformation converts products to sums, simplifying computation and improving numerical stability.

**Example 4.1** For Bernoulli trials with $n$ observations and $k$ successes, the MLE for parameter $\theta$ is:

$$\hat{\theta}_{MLE} = \frac{k}{n}$$

This follows from maximizing the log-likelihood:

$$\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

Setting $\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$ yields the result.

**Theorem 4.1 (Consistency of MLE)** Under regularity conditions, the MLE is consistent:

$$\hat{\boldsymbol{\theta}}_{MLE} \xrightarrow{P} \boldsymbol{\theta}_0$$

as $n \to \infty$, where $\boldsymbol{\theta}_0$ is the true parameter value.

**Theorem 4.2 (Asymptotic Normality of MLE)** Under regularity conditions:

$$\sqrt{n}(\hat{\boldsymbol{\theta}}_{MLE} - \boldsymbol{\theta}_0) \xrightarrow{D} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\boldsymbol{\theta}_0))$$

where $\mathcal{I}(\boldsymbol{\theta})$ is the Fisher information matrix.

**Importantly**

I've provided the theoretical foundation for parameter estimation in machine learning models above. There are a few important things to understand:
- Nearly all of machinery involved with *learning* is rooted in maximum likelihood estimation and derivative methods.
- Nearly all of the work made in deep learning over the past 20 years has been not with the underlying estimation methods, but with clever parameterizations (i.e. network architecture) of the data distribution and more powerful optimization methods.


# Appendix

## [A.1] Derivation of the maximum likelihood estimate of N-coin flips

Solving for the maximum likelihood estimate of $n$ independent coin flips amounts to deriving a the MLE estimate for the Bernoulli parameter, $\theta$.

Consider $n$ independent coin flips, where each flip has probability $\theta$ of resulting in heads. Let $X_i$ denote the outcome of the $i$-th flip, where $X_i = 1$ indicates heads and $X_i = 0$ indicates tails. We observe $k = \sum_{i=1}^{n} X_i$ heads out of $n$ total flips.

Each $X_i$ follows a Bernoulli distribution with parameter $\theta$:

$$P(X_i = x_i) = \theta^{x_i}(1-\theta)^{1-x_i}, \quad x_i \in \{0,1\}$$

We seek the maximum likelihood estimator $\hat{\theta}_{MLE}$.

### Likelihood Function

Since the coin flips are independent, the joint probability mass function is the product of individual probabilities:

$$L(\theta) = P(X_1 = x_1, X_2 = x_2, \ldots, X_n = x_n; \theta) = \prod_{i=1}^{n} P(X_i = x_i; \theta)$$

Substituting the Bernoulli PMF:

$$L(\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i}$$

Using properties of exponents:

$$L(\theta) = \prod_{i=1}^{n} \theta^{x_i} \prod_{i=1}^{n} (1-\theta)^{1-x_i}$$

$$L(\theta) = \theta^{\sum_{i=1}^{n} x_i}\,(1-\theta)^{\sum_{i=1}^{n} (1-x_i)}$$

Since

$$
\sum_{i=1}^{n} x_i = k
$$
and
$$
\sum_{i=1}^{n} (1-x_i) = n - k
$$:

$$L(\theta) = \theta^k (1-\theta)^{n-k}$$

### Log-Likelihood Function

Taking the natural logarithm:

$$\ell(\theta) = \log L(\theta) = \log[\theta^k (1-\theta)^{n-k}]$$

Using logarithm properties:

$$\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

### Find the Critical Point

To find the maximum, we differentiate with respect to $\theta$ and set equal to zero:

$$\frac{d\ell(\theta)}{d\theta} = \frac{d}{d\theta}[k \log \theta + (n-k) \log(1-\theta)]$$

$$\frac{d\ell(\theta)}{d\theta} = k \cdot \frac{1}{\theta} + (n-k) \cdot \frac{1}{1-\theta} \cdot (-1)$$

$$\frac{d\ell(\theta)}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$$

Setting the first derivative equal to zero:

$$\frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

### Solve for θ

Rearranging:

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

Cross-multiplying:

$$k(1-\theta) = \theta(n-k)$$

Expanding:

$$k - k\theta = \theta n - k\theta$$

$$k = \theta n$$

Therefore:

$$\hat{\theta}_{MLE} = \frac{k}{n}$$

### Verify Maximum

We verify this is indeed a maximum by checking the second derivative:

$$\frac{d^2\ell(\theta)}{d\theta^2} = \frac{d}{d\theta}\left[\frac{k}{\theta} - \frac{n-k}{1-\theta}\right]$$

$$\frac{d^2\ell(\theta)}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2}$$

Since $k \geq 0$, $n-k \geq 0$, and $\theta \in (0,1)$, we have:

$$\frac{d^2\ell(\theta)}{d\theta^2} < 0$$

The second derivative is negative for all $\theta \in (0,1)$, confirming that $\hat{\theta}_{MLE} = \frac{k}{n}$ is indeed a maximum.

### Interpretation

The MLE for the Bernoulli parameter is simply the observed proportion of successes. This result aligns with intuition: if we observe 7 heads out of 10 flips, our best estimate for the probability of heads is $\frac{7}{10} = 0.7$.

### Boundary Considerations

Note that we must consider the constraints $\theta \in [0,1]$:
- If $k = 0$, then $\hat{\theta}_{MLE} = 0$
- If $k = n$, then $\hat{\theta}_{MLE} = 1$
- If $0 < k < n$, then $\hat{\theta}_{MLE} = \frac{k}{n} \in (0,1)$

In all cases, the MLE is the sample proportion $\frac{k}{n}$.
