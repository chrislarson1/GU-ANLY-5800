# Exam 01 Study Guide

## ANLY-5800 Fall 2025 | Georgetown University

*Review material for exam covering lectures 02 and 03.*

---

## Overview

**Lecture Topics:**
- **Lecture 2:** Decision Boundary Classification (Bag-of-Words, Perceptron, SVM)
- **Lecture 3:** Statistical Learning for Text Classification (Naive Bayes, Softmax Regression)

---

## Lecture 2: Decision Boundary Classification

### Key Concepts

#### Bag-of-Words Representation

**Definition:** Document $d$ with vocabulary $V$ becomes vector $\mathbf{x}_d \in \mathbb{R}^{|V|}$ where $(\mathbf{x}_d)_i = \text{count}(w_i, d)$

**Properties:**
- High dimensionality ($|V| \sim 10^4$ to $10^6$)
- Sparsity (most documents use small fraction of vocabulary)
- Loss of sequential information
- Fixed feature space across all documents

#### High-Dimensional Geometry

**Cover's Theorem:** In high dimensions, probability of linear separability approaches 1
- $P(\text{linear separability}) = 2^{-m} \sum_{k=0}^{d} \binom{m}{k}$
- For fixed $m$ and $d \to \infty$: $P \to 1$

**Distance Concentration:** In high dimensions, all pairwise distances become approximately equal
- Facilitates linear separability
- Volume concentrates near boundaries

#### The Perceptron Algorithm

**Problem:** Find $\mathbf{w}$ such that $\text{sign}(\mathbf{w}^T\mathbf{x}_i) = y_i$ for all training examples

**Algorithm:**
**Perceptron Algorithm (Pseudocode)**

```
Initialize: w = 0

Repeat:
    For each (x_i, y_i):
        If y_i * (w^T x_i) <= 0:
            w <- w + η * y_i * x_i
Until no mistakes
```


**Convergence Theorem:** If data is linearly separable with margin $\gamma$, Perceptron converges in at most $\left(\frac{R}{\gamma}\right)^2$ iterations

**Limitation:** Fails to converge on non-separable data

#### Support Vector Machines

**Geometric Margin:** Distance from point to hyperplane = $\frac{|\mathbf{w}^T\mathbf{x} + b|}{\|\mathbf{w}\|_2}$

**Hard-Margin SVM:**
$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i$$

**Soft-Margin SVM:** Introduces slack variables $\xi_i \geq 0$
$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{m} \xi_i$$
subject to: $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$

**Hinge Loss:** $\max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$

### Study Questions

1. **Geometric:** Explain why high-dimensional spaces favor linear separability using Cover's theorem.

2. **Algorithmic:** What happens to the Perceptron algorithm when data is not linearly separable? How does SVM address this?

3. **Optimization:** Interpret the regularization parameter $C$ in soft-margin SVM in terms of bias-variance tradeoff.

---

## Lecture 3: Statistical Learning for Text Classification

### Key Concepts

#### Generative vs. Discriminative Models

**Generative Models:**
- Learn joint distribution $P(\mathbf{x}, y; \boldsymbol{\theta})$
- Inference via Bayes' rule: $\hat{y} = \arg\max_{y} P(y|\mathbf{x}) = \arg\max_{y} \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}$
- Example: Naive Bayes

**Discriminative Models:**
- Learn conditional distribution $P(y|\mathbf{x}; \boldsymbol{\theta})$ directly
- Direct optimization of classification objective
- Example: Softmax Regression

#### Naive Bayes Classifier

**Generative Process:**
1. Label: $y \sim \text{Categorical}(\boldsymbol{\lambda})$
2. Features: $P(\mathbf{x}|y=k; \boldsymbol{\phi}_k) = \text{Multinomial}(\mathbf{x}; \boldsymbol{\phi}_k)$

**Naive Bayes Assumption:** $P(\mathbf{x}|y; \boldsymbol{\phi}) = \prod_{j=1}^{N} P(x_j|y; \phi_j)$

**Parameter Estimation (MLE):**
- Class prior: $\hat{\lambda}_k = \frac{M_k}{M}$
- Feature parameters: $\hat{\phi}_{k,j} = \frac{\text{count}(y=k, x=j)}{\text{count}(y=k, \text{all words})}$

**Laplace Smoothing:** $\hat{\phi}_{k,j} = \frac{\alpha + \sum_{i} \mathbb{I}(y_i = k) x_{i,j}}{\alpha N + \sum_{i} \mathbb{I}(y_i = k) \sum_{j'} x_{i,j'}}$

**Classification Rule:** $\hat{y} = \arg\max_{k} \log P(y=k) + \sum_{j=1}^{N} x_j \log \phi_{k,j}$

#### Softmax Regression

**Model:** $P(y=k|\mathbf{x}; \boldsymbol{\theta}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{k'=1}^{K} \exp(\mathbf{w}_{k'}^T \mathbf{x} + b_{k'})}$

**Softmax Function:** $\sigma(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{k'=1}^{K} e^{z_{k'}}}$
- Properties: positive, sums to 1, translation invariant

**Loss Function:** Negative log-likelihood = Cross-entropy
$$\text{NLL}(\boldsymbol{\theta}) = -\sum_{i=1}^{M} \sum_{k=1}^{K} \mathbb{I}(y_i = k) \log P(y=k|\mathbf{x}_i; \boldsymbol{\theta})$$

**Gradient:** $\nabla_{\mathbf{w}_k} \text{NLL} = \sum_{i=1}^{M} (P(y=k|\mathbf{x}_i) - \mathbb{I}(y_i = k)) \mathbf{x}_i$

**Key Property:** Objective is convex → any local minimum is global minimum

#### Comparison

**Functional Equivalence:** Under naive Bayes assumptions, both approaches yield linear decision boundaries

**Differences:**
- **Naive Bayes:** Closed-form MLE, stronger assumptions about $P(\mathbf{x}|y)$
- **Softmax:** Iterative optimization, weaker assumptions, direct modeling of $P(y|\mathbf{x})$

### Study Questions

1. **Probabilistic:** Derive the naive Bayes classification rule starting from Bayes' theorem.

2. **Optimization:** Why is the softmax regression objective convex? What are the implications?

3. **Comparison:** When might you prefer naive Bayes over softmax regression, and vice versa?

---

## Key Formulas Reference

### Probability & Information Theory
- **Bayes' theorem:** $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$
- **Cross-entropy loss:** $H(P,Q) = -\sum_{x} P(x) \log Q(x)$

### Classification Algorithms
- **Perceptron update:** $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$ (if mistake)
- **SVM objective:** $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i$
- **Softmax:** $P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}}}$
- **Naive Bayes:** $\hat{y} = \arg\max_k \left[\log P(y=k) + \sum_j x_j \log \phi_{k,j}\right]$

---

## Practice Problems

### Problem 1: Bag-of-Words and Linear Separability (Lecture 2)
Consider a binary text classification problem with vocabulary size 1000:

a) What is the dimensionality of the feature space using bag-of-words representation?

b) Explain why high-dimensional text data is often linearly separable (reference Cover's theorem).

c) What are the key properties of bag-of-words representation that make it suitable for classification?

### Problem 2: Perceptron Algorithm (Lecture 2)
Given the following 2D training data for binary classification:
- $(x_1, y_1) = ([2, 1], +1)$
- $(x_2, y_2) = ([1, 2], +1)$
- $(x_3, y_3) = ([-1, -1], -1)$
- $(x_4, y_4) = ([-2, 1], -1)$

a) Is this data linearly separable? Justify your answer.
b) Apply one iteration of the Perceptron algorithm starting with $\mathbf{w} = [0, 0]$ and $\eta = 1$.
c) What is the convergence guarantee for the Perceptron algorithm?

### Problem 3: SVM Concepts (Lecture 2)
For Support Vector Machines:

a) What is the geometric margin of a point $\mathbf{x}$ with respect to hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$?

b) Write the optimization objective for soft-margin SVM and explain each term.

c) What is the hinge loss and how does it relate to the slack variables?

### Problem 4: Naive Bayes Classification (Lecture 3)
Consider a spam classification problem with vocabulary {"free", "money", "meeting", "lunch"}. Given the following training data:

**Spam emails:**
- Email 1: "free money"
- Email 2: "free free money"

**Ham emails:**
- Email 3: "lunch meeting"
- Email 4: "meeting"

a) Calculate the MLE estimates for the class priors $P(\text{spam})$ and $P(\text{ham})$.

b) Calculate the MLE estimates for word probabilities (e.g., $P(\text{free}|\text{spam})$).

c) Classify the test email "free lunch" using naive Bayes.

### Problem 5: Softmax Regression (Lecture 3)
For a 3-class softmax regression problem:

a) Write the softmax function for computing $P(y=k|\mathbf{x})$.

b) What is the gradient of the negative log-likelihood with respect to $\mathbf{w}_k$?

c) Why is the softmax regression objective convex, and what does this guarantee?

### Problem 6: Generative vs. Discriminative (Lecture 3)
Compare Naive Bayes and Softmax Regression:

a) What distributions does each method model?

b) Under what conditions do they produce the same decision boundaries?

c) What are the trade-offs between the two approaches in terms of assumptions, computational complexity, and data requirements?

---

## Exam Preparation Tips

### Conceptual Understanding
- **Understand the "why":** Don't just memorize formulas—understand the motivation behind each method
- **Connect the approaches:** See how geometric (Perceptron, SVM) and probabilistic (Naive Bayes, Softmax) methods relate
- **Geometric intuition:** Visualize decision boundaries, margins, and high-dimensional separability

### Mathematical Preparation
- **Derive key results:** Practice deriving MLE estimates, gradients, and optimization objectives
- **Probability rules:** Master Bayes' theorem, conditional probability, and likelihood functions
- **Optimization concepts:** Understand convexity, gradient descent, and convergence guarantees

### Practical Applications
- **Algorithm steps:** Be able to write out algorithms (Perceptron, gradient descent for softmax)
- **Parameter interpretation:** Understand what each parameter controls (learning rate, regularization, smoothing (we didn't cover this in class, please do self study))
- **Trade-offs:** Know when to use each method and their relative advantages/disadvantages

---