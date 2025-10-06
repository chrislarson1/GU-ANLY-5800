# Lecture 03: Statistical Learning for Text Classification

*Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

In Lecture 02, we introduced the classification problem and examined geometric approaches based on decision boundaries. We now shift our focus to probabilistic frameworks for classification, developing two complementary perspectives: generative models that learn the joint distribution $P(\mathbf{x}, y)$, and discriminative models that learn the conditional distribution $P(y|\mathbf{x})$ directly.

## A Taxonomy of Statistical Machine Learning

Machine learning algorithms can be categorized by how they model the relationship between the random variables at play within a given problem setting.

**Definition 1.1** Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{M}$ where $\mathbf{x}_i \in \mathbb{R}^N$ and $y_i \in \mathcal{Y}$:

- A **generative model** learns the joint distribution $P(\mathbf{x}, y; \boldsymbol{\theta})$ and performs inference via Bayes' rule:
  $$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y|\mathbf{x}) = \arg\max_{y \in \mathcal{Y}} \frac{P(\mathbf{x}|y; \boldsymbol{\phi})P(y; \boldsymbol{\lambda})}{P(\mathbf{x})}$$

- A **discriminative model** learns the conditional distribution $P(y|\mathbf{x}; \boldsymbol{\theta})$ directly:
  $$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y|\mathbf{x}; \boldsymbol{\theta})$$

**Observation 1.1** The generative approach requires modeling the feature distribution $P(\mathbf{x}|y)$, which for high-dimensional text data involves estimating a probability distribution over a very large space. The discriminative approach bypasses this by directly modeling the decision boundary.

**Observation 1.2** Generative models provide a complete probabilistic description of the data-generating process, enabling tasks beyond classification such as sampling and density estimation. Discriminative models focus computational resources solely on the classification objective.

---

## Generative Classification: Naive Bayes

The Naive Bayes classifier represents one of the earliest and most successful applications of probabilistic reasoning to text classification. Despite its simplifying assumptions, it remains effective in practice and provides important insights into generative modeling.

### The Generative Process

**Definition 2.1** The Naive Bayes generative model assumes the following data-generating process:

1. **Label distribution:** $y \sim \text{Categorical}(\boldsymbol{\lambda})$ where $\boldsymbol{\lambda} = [\lambda_1, \ldots, \lambda_K]$ and $\sum_{k=1}^{K} \lambda_k = 1$

2. **Feature distribution:** Given class $y=k$, features are drawn from a multinomial distribution:
   $$P(\mathbf{x}|y=k; \boldsymbol{\phi}_k) = \text{Multinomial}(\mathbf{x}; \boldsymbol{\phi}_k)$$
   where $\mathbf{x} = [x_1, \ldots, x_N]$ represents word counts from vocabulary of size $N$.

**Definition 2.2** The multinomial distribution over feature vector $\mathbf{x}$ with parameters $\boldsymbol{\phi} = [\phi_1, \ldots, \phi_N]$ is:

$$P(\mathbf{x}|\boldsymbol{\phi}) = \frac{\left(\sum_{j=1}^{N} x_j\right)!}{\prod_{j=1}^{N} x_j!} \prod_{j=1}^{N} \phi_j^{x_j}$$

where $\sum_{j=1}^{N} \phi_j = 1$ and $\phi_j \geq 0$ for all $j$.

**Remark 2.1** The multinomial coefficient $\frac{(\sum_j x_j)!}{\prod_j x_j!}$ counts the number of ways to arrange the observed word counts. In practice, this term can be omitted during inference as it doesn't depend on the class label.

### The Naive Bayes Assumption

**Definition 2.3** The **naive Bayes assumption** states that given the class label, all features are conditionally independent:

$$P(\mathbf{x}|y; \boldsymbol{\phi}) = \prod_{j=1}^{N} P(x_j|y; \phi_j)$$

This dramatically simplifies the model from $P(\mathbf{x}_1, \ldots, \mathbf{x}_N|y)$ to a product of univariate distributions.

**Lemma 2.1** Under the naive Bayes assumption with multinomial features, the likelihood factorizes:

$$P(\mathbf{x}|y=k; \boldsymbol{\phi}_k) = \prod_{j=1}^{N} \phi_{k,j}^{x_j}$$

where $\phi_{k,j}$ represents the probability of word $j$ in class $k$.

**Observation 2.1** Despite the strong independence assumption being violated in natural language (where word co-occurrences carry semantic information), Naive Bayes often performs surprisingly well in practice for certain classes of problems.

### Parameter Estimation

We estimate model parameters via maximum likelihood estimation.

**Theorem 2.1 (MLE for Class Prior)** The maximum likelihood estimate for the class prior is:

$$\hat{\lambda}_k = \frac{M_k}{M}$$

where $M_k = \sum_{i=1}^{M} \mathbb{I}(y_i = k)$ is the count of training examples in class $k$.

**Proof:** The log-likelihood for $\boldsymbol{\lambda}$ with constraint $\sum_{k=1}^{K} \lambda_k = 1$ is:

$$\mathcal{L}(\boldsymbol{\lambda}) = \sum_{i=1}^{M} \log P(y_i; \boldsymbol{\lambda}) = \sum_{k=1}^{K} M_k \log \lambda_k$$

Using Lagrange multipliers:

$$\mathcal{L}(\boldsymbol{\lambda}, \mu) = \sum_{k=1}^{K} M_k \log \lambda_k - \mu\left(\sum_{k=1}^{K} \lambda_k - 1\right)$$

Taking derivatives:

$$\frac{\partial \mathcal{L}}{\partial \lambda_k} = \frac{M_k}{\lambda_k} - \mu = 0 \implies \lambda_k = \frac{M_k}{\mu}$$

Applying the constraint:

$$\sum_{k=1}^{K} \lambda_k = \sum_{k=1}^{K} \frac{M_k}{\mu} = \frac{M}{\mu} = 1 \implies \mu = M$$

Therefore: $\hat{\lambda}_k = \frac{M_k}{M}$

**Theorem 2.2 (MLE for Feature Parameters)** The maximum likelihood estimate for the multinomial parameters is:

$$\hat{\phi}_{k,j} = \frac{\sum_{i=1}^{M} \mathbb{I}(y_i = k) x_{i,j}}{\sum_{i=1}^{M} \mathbb{I}(y_i = k) \sum_{j'=1}^{N} x_{i,j'}} = \frac{\text{count}(y=k, x=j)}{\text{count}(y=k, \text{ all words})}$$

**Proof sketch:** The log-likelihood with respect to $\boldsymbol{\phi}_k$ subject to $\sum_j \phi_{k,j} = 1$ is:

$$\mathcal{L}(\boldsymbol{\phi}_k) = \sum_{i: y_i=k} \sum_{j=1}^{N} x_{i,j} \log \phi_{k,j}$$

Following the same Lagrange multiplier approach as above yields the stated result.

**Remark 2.2** This estimate represents the empirical frequency of each word within documents of class $k$.

### Laplace Smoothing

**Definition 2.4** The MLE can yield zero probabilities for words not observed in the training set for a given class. **Laplace smoothing** (or additive smoothing) addresses this by adding a pseudo-count $\alpha > 0$:

$$\hat{\phi}_{k,j} = \frac{\alpha + \sum_{i=1}^{M} \mathbb{I}(y_i = k) x_{i,j}}{\alpha N + \sum_{i=1}^{M} \mathbb{I}(y_i = k) \sum_{j'=1}^{N} x_{i,j'}}$$

where $\alpha = 1$ corresponds to standard Laplace smoothing.

**Lemma 2.2** Under Laplace smoothing, $\hat{\phi}_{k,j} > 0$ for all $j$, ensuring that $P(\mathbf{x}|y=k) > 0$ for all possible feature vectors.

### Inference

**Theorem 2.3 (Naive Bayes Classification)** Given model parameters $\boldsymbol{\lambda}$ and $\boldsymbol{\phi}$, the predicted class is:

$$\hat{y} = \arg\max_{k} \log P(y=k) + \sum_{j=1}^{N} x_j \log \phi_{k,j}$$

**Derivation:** By Bayes' rule:

$$P(y=k|\mathbf{x}) \propto P(\mathbf{x}|y=k)P(y=k)$$

Under the naive Bayes assumption:

$$P(y=k|\mathbf{x}) \propto P(y=k) \prod_{j=1}^{N} \phi_{k,j}^{x_j}$$

Taking logarithms:

$$\log P(y=k|\mathbf{x}) = \log P(y=k) + \sum_{j=1}^{N} x_j \log \phi_{k,j} + \text{const}$$

Since the constant term doesn't depend on $k$, we can ignore it for classification.

**Observation 2.2** This decision rule is linear in the feature space defined by word counts, connecting back to the geometric perspective from Lecture 02.

---

## Discriminative Classification: Softmax Regression

While Naive Bayes models the joint distribution $P(\mathbf{x}, y)$, discriminative approaches directly parameterize $P(y|\mathbf{x})$. We now develop softmax regression (also called multinomial logistic regression), which provides a discriminative counterpart to Naive Bayes.

### Model Definition

**Definition 3.1** Softmax regression models the conditional class probabilities as:

$$P(y=k|\mathbf{x}; \boldsymbol{\theta}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{k'=1}^{K} \exp(\mathbf{w}_{k'}^T \mathbf{x} + b_{k'})}$$

where $\boldsymbol{\theta} = \{\mathbf{w}_1, \ldots, \mathbf{w}_K, b_1, \ldots, b_K\}$ are the model parameters.

**Definition 3.2** The **softmax function** $\sigma: \mathbb{R}^K \to (0,1)^K$ is defined by:

$$\sigma(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{k'=1}^{K} e^{z_{k'}}}$$

This function maps arbitrary real-valued scores (logits) to a valid probability distribution.

**Lemma 3.1** The softmax function satisfies:
1. $\sigma(\mathbf{z})_k > 0$ for all $k$
2. $\sum_{k=1}^{K} \sigma(\mathbf{z})_k = 1$
3. $\sigma(\mathbf{z} + c\mathbf{1})= \sigma(\mathbf{z})$ for any constant $c$ (translation invariance)

**Remark 3.1** The translation invariance implies that the model is overparameterized. In practice, we often set $\mathbf{w}_K = \mathbf{0}$ and $b_K = 0$ to fix this indeterminacy.

### Connection to Linear Models

**Observation 3.1** Each class $k$ has an associated linear decision function:

$$z_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + b_k$$

The softmax function converts these scores into normalized probabilities. The decision boundary between classes $i$ and $j$ is the hyperplane where $z_i(\mathbf{x}) = z_j(\mathbf{x})$, or equivalently:

$$(\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (b_i - b_j) = 0$$

### Parameter Estimation via Maximum Likelihood

**Definition 3.3** The negative log-likelihood (NLL) for softmax regression is:

$$\text{NLL}(\boldsymbol{\theta}) = -\sum_{i=1}^{M} \log P(y_i|\mathbf{x}_i; \boldsymbol{\theta}) = -\sum_{i=1}^{M} \sum_{k=1}^{K} \mathbb{I}(y_i = k) \log P(y=k|\mathbf{x}_i; \boldsymbol{\theta})$$

**Remark 3.2** Recall from Lecture 01 that minimizing NLL is equivalent to minimizing cross-entropy $H(P_{\text{data}}, P_{\text{model}})$ and thus equivalent to minimizing KL divergence $D_{KL}(P_{\text{data}} \parallel P_{\text{model}})$.

**Theorem 3.1 (Gradient of Softmax Regression)** The gradient of the NLL with respect to $\mathbf{w}_k$ is:

$$\nabla_{\mathbf{w}_k} \text{NLL}(\boldsymbol{\theta}) = \sum_{i=1}^{M} \left(P(y=k|\mathbf{x}_i; \boldsymbol{\theta}) - \mathbb{I}(y_i = k)\right) \mathbf{x}_i$$

**Proof:** Let $p_{i,k} = P(y=k|\mathbf{x}_i; \boldsymbol{\theta})$ denote the predicted probability. The NLL is:

$$\text{NLL} = -\sum_{i=1}^{M} \sum_{k=1}^{K} \mathbb{I}(y_i = k) \log p_{i,k}$$

Taking the derivative:

$$\frac{\partial \text{NLL}}{\partial \mathbf{w}_k} = -\sum_{i=1}^{M} \sum_{k'=1}^{K} \mathbb{I}(y_i = k') \frac{1}{p_{i,k'}} \frac{\partial p_{i,k'}}{\partial \mathbf{w}_k}$$

The key is computing $\frac{\partial p_{i,k'}}{\partial \mathbf{w}_k}$. For the softmax function with $z_k = \mathbf{w}_k^T \mathbf{x}_i + b_k$:

$$\frac{\partial p_{i,k'}}{\partial z_k} = \begin{cases}
p_{i,k}(1 - p_{i,k}) & \text{if } k' = k \\
-p_{i,k} p_{i,k'} & \text{if } k' \neq k
\end{cases}$$

Since $\frac{\partial z_k}{\partial \mathbf{w}_k} = \mathbf{x}_i$, we have:

$$\frac{\partial \text{NLL}}{\partial \mathbf{w}_k} = -\sum_{i=1}^{M} \left[\mathbb{I}(y_i = k)(1 - p_{i,k})\mathbf{x}_i - \sum_{k' \neq k} \mathbb{I}(y_i = k') p_{i,k} \mathbf{x}_i\right]$$

Simplifying:

$$= -\sum_{i=1}^{M} \left[\mathbb{I}(y_i = k)\mathbf{x}_i - p_{i,k}\mathbf{x}_i\right] = \sum_{i=1}^{M} (p_{i,k} - \mathbb{I}(y_i = k))\mathbf{x}_i$$

**Corollary 3.1** The gradient with respect to the bias term is:

$$\nabla_{b_k} \text{NLL}(\boldsymbol{\theta}) = \sum_{i=1}^{M} (P(y=k|\mathbf{x}_i; \boldsymbol{\theta}) - \mathbb{I}(y_i = k))$$

**Interpretation:** The gradient has an intuitive form: it's the sum over training examples of the prediction error weighted by the input features. When the model predicts class $k$ with probability 1 but the true class is different, the gradient is large. When predictions match labels, the contribution is zero.

### Optimization via Gradient Descent

**Algorithm 3.1 (Gradient Descent for Softmax Regression)**

The standard approach to minimizing NLL uses gradient descent:

$$
\begin{aligned}
&\textbf{Input:} \quad \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^M,\ \eta > 0,\ T_{\max} \\
&\textbf{Initialize:} \quad \mathbf{w}_k = \mathbf{0}, b_k = 0 \text{ for all } k \\
&\textbf{For } t = 1 \text{ to } T_{\max}: \\
&\quad \text{Compute } \nabla_{\mathbf{w}_k} \text{NLL}(\boldsymbol{\theta}^{(t)}) \text{ and } \nabla_{b_k} \text{NLL}(\boldsymbol{\theta}^{(t)}) \text{ for all } k \\
&\quad \mathbf{w}_k^{(t+1)} \leftarrow \mathbf{w}_k^{(t)} - \eta \nabla_{\mathbf{w}_k} \text{NLL}(\boldsymbol{\theta}^{(t)}) \\
&\quad b_k^{(t+1)} \leftarrow b_k^{(t)} - \eta \nabla_{b_k} \text{NLL}(\boldsymbol{\theta}^{(t)}) \\
&\textbf{Output:} \quad \boldsymbol{\theta}^{(T_{\max})}
\end{aligned}
$$

**Remark 3.3** In practice, several variants are commonly used:
- **Stochastic gradient descent (SGD):** Compute gradient using a single random example per iteration
- **Mini-batch gradient descent:** Use a small subset of examples to compute gradient
- **Momentum methods:** Incorporate velocity terms to accelerate convergence

**Theorem 3.2 (Convexity of Softmax Regression)** The negative log-likelihood for softmax regression is convex in $\boldsymbol{\theta}$.

**Proof sketch:** The NLL can be written as:

$$\text{NLL}(\boldsymbol{\theta}) = \sum_{i=1}^{M} \left[\log \sum_{k=1}^{K} \exp(\mathbf{w}_k^T \mathbf{x}_i + b_k) - \sum_{k=1}^{K} \mathbb{I}(y_i=k)(\mathbf{w}_k^T \mathbf{x}_i + b_k)\right]$$

The log-sum-exp function $\log \sum_k \exp(z_k)$ is convex, and the second term is linear, hence their sum is convex.

**Corollary 3.2** Any local minimum of the softmax regression objective is a global minimum.

---

## Comparison: Generative vs. Discriminative

**Theorem 4.1** Under the naive Bayes generative assumptions (categorical labels, multinomial features with naive Bayes independence), the resulting classifier has the same functional form as softmax regression.

**Proof sketch:** The posterior under naive Bayes is:

$$\log P(y=k|\mathbf{x}) = \log P(y=k) + \sum_{j=1}^{N} x_j \log \phi_{k,j} + \text{const}$$

This is linear in $\mathbf{x}$, identical to the logit form in softmax regression with weights:

$$\mathbf{w}_k = [\log \phi_{k,1}, \ldots, \log \phi_{k,N}]^T, \quad b_k = \log \lambda_k$$

**Observation 4.1** Despite the similar functional forms, the two approaches differ fundamentally:
- **Naive Bayes:** Estimates parameters $\boldsymbol{\phi}$ and $\boldsymbol{\lambda}$ via closed-form MLEs based on feature and label frequencies
- **Softmax Regression:** Estimates parameters $\mathbf{w}_k, b_k$ via iterative optimization of $P(y|\mathbf{x})$ directly

**Observation 4.2** The discriminative approach makes weaker assumptions—it doesn't require modeling $P(\mathbf{x}|y)$ and thus doesn't commit to a specific feature distribution. However, generative models can leverage unlabeled data more naturally and provide better sample efficiency when their assumptions hold.

---

## Practical Considerations

**Numerical Stability:** Computing $\exp(z_k)$ for large $|z_k|$ can cause overflow or underflow. The log-sum-exp trick provides numerical stability:

$$\log \sum_{k=1}^{K} \exp(z_k) = c + \log \sum_{k=1}^{K} \exp(z_k - c)$$

where $c = \max_k z_k$.

**Regularization:** To prevent overfitting, we often add an $L_2$ penalty:

$$\text{NLL}_{\text{reg}}(\boldsymbol{\theta}) = \text{NLL}(\boldsymbol{\theta}) + \frac{\lambda}{2} \sum_{k=1}^{K} \|\mathbf{w}_k\|_2^2$$

This corresponds to placing a Gaussian prior on the weights in a Bayesian framework.

**Multi-class vs. Binary Classification:** For $K=2$ classes, softmax regression reduces to logistic regression. The decision boundary is a single hyperplane rather than $K$ separate boundaries.
