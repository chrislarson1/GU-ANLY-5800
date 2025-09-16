# Lecture 02: Decision Boundary Classification

*Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

Throughtout this course, we'll discover that a very wide range of problems can be framed as ones of *classification*.

  (a) *Is a news article related to sports or politics?*

  (b) *Is an email spam?*

  (c) *Should the next word of a particular sentence be *apple*, *shirt*, or *cat*?*

Each of these examples can be modeled as a classification problem, to which we can apply approaches from statistics. Lectures 02 and 03 introduce linear models for text classification, and focus on problem settings similar to examples (a) and (b) above. In this lecture (02) we will develop geometric intuition by focusing on binary classification and models that learn the parameters of the separating hyperplane explicitely. In Lecture 03 we shift focus to statistical models, and show that their learned parameters have the same, albeit less obvious, interpretation. Later on we will tackle the problem setting 

## The Bag-of-Words Model

### From Text to Vectors

**Definition 1.1** A vocabulary $V$ is a finite set of unique words extracted from a text corpus. For a corpus containing documents $\{d_1, d_2, \ldots, d_n\}$, we define:

$$V = \bigcup_{i=1}^{n} \{w : w \text{ is a word in } d_i\}$$

**Definition 1.2 (Bag-of-Words Representation)** Given a vocabulary $V = \{w_1, w_2, \ldots, w_{|V|}\}$ and a document $d$, the bag-of-words representation is the vector $\mathbf{x}_d \in \mathbb{R}^{|V|}$ where:

$$(\mathbf{x}_d)_i = \text{count}(w_i, d)$$

the number of times word $w_i$ appears in document $d$.

**Example 1.1** Consider the text: *"the cat sat on the mat"*

With vocabulary $V = \{\text{cat}, \text{mat}, \text{on}, \text{sat}, \text{the}\}$, the bag-of-words representation is:
$$\mathbf{x}_d = [1, 1, 1, 1, 2]$$
corresponding to the word counts for each vocabulary word in order.

### Properties of Text Representations

**Observation 1.1** The bag-of-words transformation produces a histogram over words, exhibiting several characteristics:

1. **High dimensionality**: $|V|$ typically ranges from $10^4$ to $10^6$ for realistic corpora
2. **Sparsity**: Most documents contain only a small fraction of the vocabulary
3. **Loss of sequential information**: Word order is discarded, retaining only frequency information
4. **Fixed feature space**: All documents are represented in the same $|V|$-dimensional space

**Lemma 1.1** For a corpus with $n$ documents and vocabulary size $|V|$, the bag-of-words representation yields a data matrix $\mathbf{X} \in \mathbb{R}^{n \times |V|}$ where row $i$ contains the representation of document $d_i$.

## Foundations of Linear Classification

Classification represents one of the fundamental problems in statistical learning theory. Given a feature space $\mathcal{X} = \mathbb{R}^{|V|}$ and a label space $\mathcal{Y}$, we seek to learn a mapping $f: \mathcal{X} \to \mathcal{Y}$ from labeled training examples.

**Definition 1.3** The text classification problem: Given a training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$ where $\mathbf{x}_i \in \mathbb{R}^{|V|}$ is the bag-of-words representation of document $d_i$ and $y_i \in \mathcal{Y}$ is its class label, learn a function that accurately predicts the class of new documents.

## High-Dimensional Geometry and Linear Separability

Should we expect word histograms to be linearly separable according to the classification labels for real world problems? Somewhat counterintuitively, the answer is that in many cases we can make this assumption.

### The *Curse* of Dimensionality

**Definition 2.1** A hyperplane $H$ in $\mathbb{R}^N$ is the set:
$$H = \{\mathbf{x} \in \mathbb{R}^N : \mathbf{w}^T\mathbf{x} + b = 0\}$$
where $\mathbf{w} \in \mathbb{R}^{N}_{-} \{\mathbf{0}\}$ is the normal vector and $b \in \mathbb{R}$ is the bias term.

The hyperplane partitions $\mathbb{R}^N$ into three disjoint regions:
- $H^+ = \{\mathbf{x} : \mathbf{w}^T\mathbf{x} + b > 0\}$
- $H^- = \{\mathbf{x} : \mathbf{w}^T\mathbf{x} + b < 0\}$
- $H = \{\mathbf{x} : \mathbf{w}^T\mathbf{x} + b = 0\}$

### Theorem 2.1 (Cover's Theorem on Linear Separability)

Let $\{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$ be points in general position in $\mathbb{R}^d$. The probability that these points can be linearly separated into any desired binary labeling is:

$$P(\text{linear separability}) = 2^{-m} \sum_{k=0}^{d} \binom{m}{k}$$

**Corollary 2.1** For fixed $m$ and $d \to \infty$, $P(\text{linear separability}) \to 1$.

**Proof Sketch:** The result follows from counting the number of linearly separable boolean functions over $m$ points in $d$-dimensional space. Each hyperplane can implement at most $\sum_{k=0}^{d} \binom{m}{k}$ distinct dichotomies, and there are $2^m$ total possible labelings.

### Theorem 2.2 (Distance Concentration in High Dimensions)

Let $\mathbf{X}_1, \ldots, \mathbf{X}_n$ be i.i.d. random vectors in $\mathbb{R}^d$ with $\mathbb{E}[\|\mathbf{X}_i\|_2^2] < \infty$. Then:

$$\lim_{d \to \infty} \frac{\max_{i,j} \|\mathbf{X}_i - \mathbf{X}_j\|_2}{\min_{i,j} \|\mathbf{X}_i - \mathbf{X}_j\|_2} = 1$$

**Corollary 2.2** In high dimensions, all pairwise distances become approximately equal, pushing data points toward the boundary of any convex region.

### Volume Concentration Phenomenon

**Lemma 2.1** Consider a hypercube $[0,1]^d$ and its inner hypercube $[1/4, 3/4]^d$. The fraction of volume occupied by the inner cube is:

$$\frac{\text{Vol}([\frac{1}{4}, \frac{3}{4}]^d)}{\text{Vol}([0,1]^d)} = \left(\frac{1}{2}\right)^d \to 0 \text{ as } d \to \infty$$

This demonstrates that in high dimensions, most volume concentrates near the boundary, facilitating linear separability.

### Augmented Representation

For computational convenience, we employ the standard augmentation technique:

**Definition 2.2** Given $\mathbf{x} \in \mathbb{R}^N$ and decision function $\mathbf{w}^T\mathbf{x} + b$, we define:
- $\tilde{\mathbf{x}} = [1, \mathbf{x}] \in \mathbb{R}^{N+1}$
- $\tilde{\mathbf{w}} = [b, \mathbf{w}] \in \mathbb{R}^{N+1}$

Then $\mathbf{w}^T\mathbf{x} + b = \tilde{\mathbf{w}}^T\tilde{\mathbf{x}}$, allowing us to work with homogeneous coordinates throughout.

## The Perceptron Algorithm

We now examine algorithmic approaches for learning hyperplane parameters from data.

### Problem Formulation

**Definition 3.1** Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$ be a training dataset where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$. The linear classification problem seeks to find $\mathbf{w} \in \mathbb{R}^d$ such that:

$$\text{sign}(\mathbf{w}^T\mathbf{x}_i) = y_i \quad \forall i \in \{1, \ldots, m\}$$

**Definition 3.2** A dataset $\mathcal{D}$ is linearly separable if there exists $\mathbf{w} \in \mathbb{R}^d$ satisfying the above condition.

### The Perceptron Learning Algorithm

**Algorithm 3.1 (Perceptron)**
The Perceptron algorithm can be written as follows:

$$
\begin{aligned}
&\textbf{Input:} \quad \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m,\ \eta > 0 \\
&\textbf{Initialize:} \quad \mathbf{w} = 0 \\
&\textbf{Repeat:} \\
&\quad \text{For } i = 1 \text{ to } m: \\
&\qquad \text{If } y_i (\mathbf{w}^T \mathbf{x}_i) \leq 0: \\
&\qquad\quad \mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i \\
&\quad \text{Until no mistakes made in a full pass through the data} \\
&\textbf{Output:} \quad \mathbf{w}
\end{aligned}
$$

**Theorem 3.1 (Perceptron Convergence)** If the training data is linearly separable with margin $\gamma > 0$, then the Perceptron algorithm converges in at most $\left(\frac{R}{\gamma}\right)^2$ iterations, where $R = \max_i \|\mathbf{x}_i\|_2$.

**Proof Sketch:** The proof establishes two key bounds:
1. **Progress bound:** Each mistake increases $\mathbf{w}^T\mathbf{w}^*$ by at least $\gamma$
2. **Mistake bound:** Each mistake increases $\|\mathbf{w}\|_2^2$ by at most $R^2$

Combining these bounds via the Cauchy-Schwarz inequality yields the result.

### Geometric Interpretation of Updates

**Lemma 3.1** The Perceptron update rule has the following geometric interpretation:

For a misclassified example $(\mathbf{x}_i, y_i)$ with $y_i(\mathbf{w}^T\mathbf{x}_i) \leq 0$:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + y_i\mathbf{x}_i$$

This update satisfies:
$$(\mathbf{w}^{(t+1)})^T\mathbf{x}_i = \mathbf{w}^{(t)T}\mathbf{x}_i + y_i\|\mathbf{x}_i\|_2^2$$

The term $y_i\|\mathbf{x}_i\|_2^2$ represents a correction in the direction needed to properly classify $\mathbf{x}_i$.

### Limitations of the Perceptron

**Theorem 3.2** The Perceptron algorithm fails to converge if the data is not linearly separable.

**Proof:** If no separating hyperplane exists, the algorithm will continue to make mistakes indefinitely, never reaching the termination condition.

This limitation motivates the development of algorithms that can handle non-separable data through optimization-theoretic approaches.

## Support Vector Machines: Maximum Margin Classification

The Support Vector Machine addresses the Perceptron's limitation by formulating classification as a convex optimization problem, enabling principled handling of both separable and non-separable data.

### Geometric Margin and Distance to Hyperplane

**Definition 4.1** The geometric margin of point $\mathbf{x}$ with respect to hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ is:

$$\text{margin}(\mathbf{x}) = \frac{|\mathbf{w}^T\mathbf{x} + b|}{\|\mathbf{w}\|_2}$$

**Derivation:** Let $\mathbf{x}_H$ be the orthogonal projection of $\mathbf{x}$ onto the hyperplane. Then:
- $\mathbf{x}_H = \mathbf{x} - \alpha \frac{\mathbf{w}}{\|\mathbf{w}\|_2}$ for some scalar $\alpha$
- Since $\mathbf{x}_H$ lies on the hyperplane: $\mathbf{w}^T\mathbf{x}_H + b = 0$
- Substituting: $\mathbf{w}^T(\mathbf{x} - \alpha \frac{\mathbf{w}}{\|\mathbf{w}\|_2}) + b = 0$
- Solving: $\alpha = \frac{\mathbf{w}^T\mathbf{x} + b}{\|\mathbf{w}\|_2}$
- Therefore: $\|\mathbf{x} - \mathbf{x}_H\|_2 = |\alpha| = \frac{|\mathbf{w}^T\mathbf{x} + b|}{\|\mathbf{w}\|_2}$

### Maximum Margin Principle

**Definition 4.2** For a linearly separable dataset, the maximum margin hyperplane is the solution to:

$$\max_{\mathbf{w},b} \left\{ \min_{i=1,\ldots,m} \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|_2} \right\}$$

This formulation seeks the hyperplane that maximizes the distance to the closest training examples.

### Solution Constraints

**Observation:** The above objective is invariant to positive scaling of $(\mathbf{w}, b)$, which means that there are infinite solutions to 4.2.

**Solution:** We impose the normalization constraint:
$$\min_{i=1,\ldots,m} y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$$

This transforms the problem to:
$$\max_{\mathbf{w},b} \frac{1}{\|\mathbf{w}\|_2} \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i \in \{1 \dots M\}$$

### Hard-Margin SVM Formulation

**Theorem 4.1** The hard-margin SVM optimization problem is:

$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i$$

This is a convex quadratic program with linear constraints, admitting efficient solution algorithms.

## Soft-Margin SVM: Handling Non-Separable Data

The hard-margin formulation requires linear separability, which is rarely satisfied in practice. We extend the framework to handle noisy, non-separable data through regularization.

### Slack Variable Formulation

**Definition 4.3** The soft-margin SVM introduces slack variables $\xi_i \geq 0$ to relax the margin constraints:

$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{m} \xi_i$$

subject to:
$$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i$$
$$\xi_i \geq 0 \quad \forall i$$

### Interpretation of Slack Variables

**Lemma 4.1** The optimal slack variables satisfy:
$$\xi_i = \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$

This gives three cases:
- $\xi_i = 0$: Point satisfies margin constraint with equality or better
- $0 < \xi_i < 1$: Point is correctly classified but within the margin
- $\xi_i \geq 1$: Point is misclassified

### Regularization Parameter

**Definition 4.4** The parameter $C > 0$ controls the regularization trade-off:
- **Large $C$**: Emphasizes fitting the training data (low bias, high variance)
- **Small $C$**: Emphasizes large margin (high bias, low variance)

**Theorem 4.2** The soft-margin SVM objective can be written as:
$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{m} \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$

The term $\max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$ is known as the **hinge loss**.
