# Lecture 05: Neural Networks

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

### Introduction to Artificial Neural Networks

---

## Lesson Plan

- Neural Networks fundamentals
- Activation functions
- Training neural networks
- Regularization techniques
- Backpropagation
- Bias-variance tradeoff
- Learning rate scheduling and annealing

---

## The Machine Learning Framework

At a high level, machine learning is a set of techniques, or *machinery*, designed to automatically learn complex relationships between random variables. Typically there is some output that we are interested in predicting, and there are inputs that we hypothesize affect that output. The goal is to accurately model real world phenomena.

In the case of NLP, the underlying data generating process lies in the realm of human cognition, which practically speaking is a black box.

---

## Recall: The Discriminative Modeling Approach

From Lecture 03, we established the following framework:

$$
\mathbf{x} \rightarrow f(\mathbf{x}; \theta) \rightarrow \mathbf{z} \rightarrow \sigma_{\text{SOFTMAX}}(\mathbf{z}) \rightarrow \hat{\mathbf{y}}
$$

Where:
- $\nabla_{\theta} \text{NLL}$ guides parameter updates
- Loss: $\mathcal{L}(\theta; \hat{\mathbf{y}}, \mathbf{y}, \mathbf{x})$

**Softmax Regression:** $f(\mathbf{x}; \theta) = \mathbf{x}\mathbf{W}^T + \mathbf{b}$

Properties:
- Convex objective function
- Linear in $\mathbf{x}$
- Decision boundaries lie orthogonally in the plane of the input $\mathbf{x}$

---

## The XOR Problem

Consider a simple binary classification problem:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

⇒ A linear function fails to classify $\mathbf{x}$ correctly!

Linear models such as $\mathbf{x}\mathbf{W}^T + \mathbf{b}$ can only model linear functions. The activation functions allow ANNs to learn non-linear manifolds in the input space.

---

## Neural Networks with Single Hidden Layer

For a simple ANN with one hidden layer:

$$
f(\mathbf{x}; \theta) = \sigma_a\left(\mathbf{x}\mathbf{W}^{(1)T} + \mathbf{b}^{(1)}\right)\mathbf{W}^{(2)T} + \mathbf{b}^{(2)}
$$

Where:
- $\theta = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}\}$
- $\sigma_a(\cdot)$ is an "activation function"

---

## Applying ANNs to the XOR Problem

Let's define:
- $\mathbf{W}^{(1)} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$
- $\mathbf{b}^{(1)} = \begin{bmatrix} 0 & -1 \end{bmatrix} \in \mathbb{R}^{2}$
- $\mathbf{W}^{(2)} = \begin{bmatrix} 1 & -2 \end{bmatrix} \in \mathbb{R}^{2}$
- $\mathbf{b}^{(2)} = 0 \in \mathbb{R}$
- $y \in \{0, 1\}$
- $\mathbf{X} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}$

Using $\sigma_a(\mathbf{z}_i) = \max(0, \mathbf{z}_i)$ ⇒ "Rectified Linear Unit" or "ReLU"

---

## XOR Solution with ANN

Computing forward pass:

$$
\mathbf{X}\mathbf{W}^{(1)T} + \mathbf{b}^{(1)} = \begin{bmatrix} 0 & 0 \\ 1 & 1 \\ 1 & 1 \\ 2 & 2 \end{bmatrix} + \begin{bmatrix} 0 & -1 \\ 1 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix} \rightarrow \mathbf{z}^{(1)}
$$

$$
\sigma_a(\mathbf{z}^{(1)}) = \begin{bmatrix} 0 & 0 \\ 1 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix} \rightarrow \mathbf{a}^{(1)}
$$

$$
\mathbf{a}^{(1)}\mathbf{W}^{(2)T} + \mathbf{b}^{(2)} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix} \Leftrightarrow \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix} \leftarrow \text{Correct XOR class assignments!}
$$

**Key insight from the Universal Approximation Theorem:**
- An ANN with at least one activation layer can approximate ANY continuous function to finite dimensions over a finite domain. This enables ANNs to learn the derivatives of the function.

---

## Activation Functions

Activation functions introduce non-linearity into neural networks:

### Sigmoid
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
$$
\nabla_z \sigma(z) = \sigma(z)(1 - \sigma(z))
$$

### Hyperbolic Tangent (Tanh)
$$
\sigma_{\tanh}(z) = \frac{2}{1 + e^{-2z}} - 1
$$
$$
\nabla_z \sigma_{\tanh}(z) = 1 - \sigma_{\tanh}(z)^2
$$

---

## Activation Functions (Continued)

### Rectified Linear Unit (ReLU)
$$
\sigma_{\text{ReLU}}(z) = \begin{cases} z & z \geq 0 \\ 0 & \text{else} \end{cases}
$$
$$
\nabla_z \sigma_{\text{ReLU}}(z) = \begin{cases} 1 & z \geq 0 \\ 0 & \text{else} \end{cases}
$$

### Leaky ReLU
$$
\sigma_{\text{LeakyReLU}}(z) = \begin{cases} z & z \geq 0 \\ \alpha z & \text{else} \end{cases}
$$
$$
\nabla_z \sigma_{\text{LeakyReLU}}(z) = \begin{cases} 1 & z \geq 0 \\ \alpha & \text{else} \end{cases}
$$

where $\alpha$ is a small constant (e.g., 0.01)

---

## ANN Learning

Recall from Lecture 03:

$$
\hat{\theta} = \underset{\theta}{\text{argmin}} \text{ NLL}(\theta; \mathcal{D})
$$

$$
= \underset{\theta}{\text{argmin}} -\sum_{i=1}^{M} f(\mathbf{x}^{(i)}; \theta)_{y^{(i)}} - \log \sum_{k \in \mathcal{K}} e^{f(\mathbf{x}^{(i)}; \theta)_k}
$$

$\hat{\theta}$ found via Gradient Descent:

$$
\nabla_{\theta} \text{NLL} = -\sum_{i=1}^{M} \nabla_{\theta} f(\mathbf{x}^{(i)}; \theta)_{y^{(i)}} - \nabla_{\theta} \log \sum_{k \in \mathcal{K}} e^{f(\mathbf{x}^{(i)}; \theta)_k}
$$

⇒ **Softmax Regression:** $f(\mathbf{x}; \theta) = \mathbf{x}\mathbf{W}^T + \mathbf{b}$

Properties:
- NLL is convex
- Any local minimum is a global minimum

⇒ **ANN:**
- $\nabla_{\theta} f(\mathbf{x}; \theta)$ requires successive application of the chain rule
- AKA **Back Propagation**
- NLL is **non-convex**, many local minima

---

## Statistical Functions for Regularization

Recall that MLE finds $\hat{\theta}_{\text{MLE}}$ that maximizes the likelihood of the observed data:

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\text{argmax}} \sum_{\mathcal{D}} p(\mathcal{D}; \theta)
$$

Let's use **Bayes' Theorem** to express a related objective:

$$
p(\theta | \mathcal{D}^{(i)}) \leftarrow \mathcal{L}(\mathcal{D}; \theta) \leftarrow p(\mathbf{x}, \mathbf{y}) \rightarrow \mathcal{L}(\theta; \mathcal{D}) \rightarrow p(\mathcal{D} | \theta)
$$

**New Objective:**
$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta)
$$

$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log \left[\frac{p(\mathcal{D} | \theta)p(\theta)}{p(\mathcal{D})}\right]
$$

$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) + \log p(\theta)
$$

---

## Regularization: Common Parameterizations for $p(\theta)$

### (1) Uniform Distribution
$$
\theta \sim \text{Unif}(\lambda)
$$
$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) + \log \text{Unif}(\lambda)
$$
$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta)
$$
$$
= \hat{\theta}_{\text{MLE}}
$$

Thus, MAP is a **generalization** of MLE

### (2) Gaussian Prior on $\theta$
$$
\theta \sim \mathcal{N}(0, \sigma_{\theta}^2)
$$
$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) + \log \left[\frac{1}{\sqrt{2\pi\sigma_{\theta}^2}} e^{-\frac{\theta^2}{2\sigma_{\theta}^2}}\right]
$$
$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) - \frac{\theta^2}{2\sigma_{\theta}^2}
$$

⇒ **$L_2$ Regularization**

---

## Regularization (Continued)

### (3) Laplace Distribution Prior
$$
\theta \sim \text{Laplace}(0, \sigma_{\theta}^2)
$$
$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) + \log \left[\frac{1}{2\sigma} e^{-\frac{|\theta|}{\sigma}}\right]
$$
$$
= \underset{\theta}{\text{argmax}} -\sum_{\mathcal{D}} \log p(\mathcal{D} | \theta) - \frac{|\theta|}{\sigma}
$$

⇒ **$L_1$ Regularization**

---

## Dropout

**Dropout** randomly selects nodes in each hidden layer $\ell_i$ and sets them to zero during training. This is an element-wise operation:

$$
\mathbf{a}^{(\ell)} = \mathbf{y} \odot \mathbf{M} \odot \mathbf{a}^{(\ell)}
$$

Where:
- $\mathbf{M}_j \sim \begin{cases} 0 & \text{with } P = P_{\text{DROP}} \\ 1 & \text{with } P = 1 - P_{\text{DROP}} \end{cases}$
- $\odot$ is element-wise (Hadamard) product
- $\gamma$ is a cost function of $P_{\text{DROP}}$

**During training:** $\mathbf{z}_j^{(\ell+1)} = \mathbf{a}^{(\ell)} \mathbf{W}_j^{T(\ell+1)}$

**During inference:** $\mathbf{z}_j^{(\ell+1)} = \alpha^{(\ell)} \mathbf{W}_j^{T(\ell+1)}$

---

## The Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that characterizes the sources of prediction error. For a model predicting target $y$ from input $\mathbf{x}$, the expected squared error can be decomposed into three components.

### Decomposition of Expected Error

**Theorem 5.1 (Bias-Variance Decomposition)** For a learning algorithm that produces predictor $\hat{f}(\mathbf{x})$ from training data $\mathcal{D}$, the expected squared error at a point $\mathbf{x}$ can be written as:

$$
\mathbb{E}_{\mathcal{D}}[(\hat{f}(\mathbf{x}) - y)^2] = \text{Bias}^2(\hat{f}(\mathbf{x})) + \text{Var}(\hat{f}(\mathbf{x})) + \sigma^2
$$

where:
- **Bias:** $\text{Bias}(\hat{f}(\mathbf{x})) = \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})] - f^*(\mathbf{x})$
- **Variance:** $\text{Var}(\hat{f}(\mathbf{x})) = \mathbb{E}_{\mathcal{D}}[(\hat{f}(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})])^2]$
- **Irreducible Error:** $\sigma^2 = \mathbb{E}[(y - f^*(\mathbf{x}))^2]$

Here $f^*(\mathbf{x}) = \mathbb{E}[y|\mathbf{x}]$ is the true function we're trying to approximate.

### Interpretation

**Bias** measures systematic errors in the model:
- How far is the average prediction from the true value?
- High bias indicates the model is too simple (underfitting)
- Sources: model assumptions, limited capacity

**Variance** measures sensitivity to training data:
- How much do predictions vary across different training sets?
- High variance indicates overfitting to training data
- Sources: model complexity, limited data

**Irreducible Error** represents inherent noise:
- Cannot be eliminated by any model
- Represents stochasticity in the true data-generating process

---

## The Tradeoff in Neural Networks

For neural networks, model capacity is controlled by several factors:

### Network Depth and Width
- **Shallow, narrow networks:** High bias, low variance
- **Deep, wide networks:** Low bias, high variance

### Training Duration
- **Few epochs:** Underfitting (high bias)
- **Many epochs:** Overfitting (high variance)

### Regularization Strength
- **Strong regularization ($\lambda$ large):** Higher bias, lower variance
- **Weak regularization ($\lambda$ small):** Lower bias, higher variance

**Theorem 5.2 (Capacity Control)** The effective capacity of a neural network can be controlled by:
1. Architecture: number of layers $L$ and hidden units per layer $\{h_1, \ldots, h_L\}$
2. Regularization: $L_1$, $L_2$ penalties, dropout rate
3. Training: early stopping, learning rate schedule

### The Sweet Spot

**Definition 5.1** The optimal model complexity occurs where:
$$
\hat{\theta}^* = \arg\min_{\theta} \left[\text{Bias}^2(\hat{f}(\cdot; \theta)) + \text{Var}(\hat{f}(\cdot; \theta))\right]
$$

In practice:
- **Underfitting:** Both training and validation error are high
- **Optimal fit:** Low training error, low validation error
- **Overfitting:** Low training error, high validation error

---

## Learning Rate Scheduling and Annealing

The learning rate $\eta$ is perhaps the most important hyperparameter in neural network training. Too large, and training diverges; too small, and convergence is prohibitively slow. Learning rate scheduling dynamically adjusts $\eta$ during training to balance fast initial progress with fine-tuned convergence.

### The Learning Rate Problem

**Observation 5.1** Using a fixed learning rate presents a dilemma:
- **Large $\eta$:** Fast initial progress, but oscillates near minima
- **Small $\eta$:** Slow convergence, may get stuck in poor local minima

**Solution:** Start with a larger learning rate and gradually reduce it during training.

---

## Learning Rate Schedules

### Step Decay
$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}
$$

where:
- $\eta_0$ is the initial learning rate
- $\gamma \in (0, 1)$ is the decay factor (typically 0.1 or 0.5)
- $k$ is the step size (e.g., every 10 epochs)

**Properties:**
- Simple to implement
- Piecewise constant learning rate
- Requires tuning $\gamma$ and $k$

### Exponential Decay
$$
\eta_t = \eta_0 \cdot e^{-\lambda t}
$$

where $\lambda > 0$ controls the decay rate.

**Properties:**
- Smooth, continuous decay
- Asymptotically approaches zero
- Single hyperparameter $\lambda$

### Polynomial Decay
$$
\eta_t = \eta_0 \cdot \left(1 + \frac{t}{T}\right)^{-p}
$$

where:
- $T$ is the total number of training steps
- $p > 0$ is the polynomial degree (often $p=1$)

**Properties:**
- Decays to zero at specific endpoint $T$
- More control over decay profile via $p$

---

## Advanced Schedules

### Cosine Annealing
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)
$$

**Properties:**
- Smooth, gradual decay following cosine curve
- Popular in modern deep learning (e.g., ResNets)
- Can restart periodically (cosine annealing with warm restarts)

### Warm Restarts (SGDR)
Periodically reset learning rate to $\eta_{\max}$ after each cycle of cosine annealing:
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t_i\pi}{T_i}\right)\right)
$$

where $t_i$ is the number of steps since last restart, and $T_i$ is the cycle length.

**Motivation:** Restarts help escape local minima and saddle points.

---

## Learning Rate Warmup

**Definition 5.2** Learning rate warmup gradually increases the learning rate from a small value to the initial target value over the first few epochs.

$$
\eta_t = \begin{cases}
\eta_0 \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\
\eta_0 \cdot \text{schedule}(t - T_{\text{warmup}}) & t > T_{\text{warmup}}
\end{cases}
$$

**Motivation:**
- At initialization, parameters are random and gradients may be large and unstable
- Starting with a large learning rate can cause immediate divergence
- Warmup allows the network to stabilize before aggressive optimization

**Common Practice:** Use warmup for first 5-10% of training, then apply chosen schedule.

---

## Adaptive Learning Rate Methods

Rather than using a single global learning rate, adaptive methods maintain per-parameter learning rates.

### AdaGrad (Adaptive Gradient)
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_{\theta} \mathcal{L}
$$

where $G_t = \sum_{i=1}^{t} \nabla_{\theta}^{(i)} \odot \nabla_{\theta}^{(i)}$ accumulates squared gradients.

**Properties:**
- Automatically reduces learning rate for frequently updated parameters
- Good for sparse data
- Can become too aggressive (learning rate → 0)

### RMSProp (Root Mean Square Propagation)
$$
v_t = \beta v_{t-1} + (1-\beta)(\nabla_{\theta} \mathcal{L})^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla_{\theta} \mathcal{L}
$$

**Properties:**
- Uses exponential moving average instead of cumulative sum
- Addresses AdaGrad's aggressive decay
- Popular for RNNs

### Adam (Adaptive Moment Estimation)
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_{\theta} \mathcal{L}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_{\theta} \mathcal{L})^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

**Properties:**
- Combines momentum and adaptive learning rates
- Bias correction for moments
- Most popular optimizer in deep learning
- Typical values: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

---

## Practical Recommendations

### Choosing a Schedule

**For most applications:**
1. Start with Adam optimizer (handles learning rate adaptation automatically)
2. If using SGD with momentum:
   - Use cosine annealing for image tasks
   - Use step decay for NLP tasks
   - Include warmup for transformer models

**Hyperparameter Selection:**
- **Initial learning rate** $\eta_0$: Often 0.001 for Adam, 0.1 for SGD
- **Warmup steps**: 1-10% of total training
- **Decay schedule**: Tuned via validation performance

### Monitoring Training

Track both training and validation metrics to diagnose issues:

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Training loss not decreasing | Learning rate too small | Increase $\eta_0$ |
| Training loss exploding | Learning rate too large | Decrease $\eta_0$, add gradient clipping |
| Good train, poor validation | Overfitting | Increase regularization, reduce capacity |
| Poor train, poor validation | Underfitting | Increase capacity, train longer |

---

## Gradient Descent with Momentum

### Stochastic Gradient Descent (SGD) with Simple Momentum

**Set** momentum $\alpha$ and learning rate $\eta$

**Initialize** $\theta$, velocity $\mathbf{v}$

**Repeat:**
- $\mathbf{x}, \mathbf{y} \sim \mathcal{D}$ (mini-batch, size $m$)
- $\nabla_{\theta} \leftarrow \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}; \theta)$
- $\mathbf{v} = \alpha\mathbf{v} - \eta\nabla_{\theta}$
- $\theta = \theta + \mathbf{v}$

**Until:** Stopping condition

**Popular variants:**
- Nesterov
- RMSProp
- Adam

---

## Neural Networks as Computational Graphs

Consider a simple ANN with single hidden layer:

$$
P_{y|x} = \frac{e^{f(\mathbf{x}; \theta)}}{\sum_{\bar{z}} e^{f(\mathbf{x}; \theta)_{\bar{z}}}} \quad f(\mathbf{x}; \theta) = \mathbf{a}^{(0)} \mathbf{W}^{(1)T} + \mathbf{b}^{(1)}
$$

$$
\mathbf{a}^{(0)} = \max(\mathbf{x}\mathbf{z}^{(0)}, \mathbf{z}^{(0)})
$$

$$
\mathbf{z}^{(0)} = \mathbf{x}\mathbf{W}^{(0)T} + \mathbf{b}^{(0)}
$$

Where:
- $\mathbf{x} \in \mathbb{Z}_+^N$
- $\mathbf{z}^{(0)} \in \mathbb{R}^D$
- $\mathbf{z}^{(1)} \in \mathbb{R}^K$
- $P_{y|x} \in [0,1]^N$

---

## Recall: The Chain Rule from Calculus

For a nested function of $\mathbf{x}$:

$$
f_i(f_2(f_1(\mathbf{x})))
$$

⇒ A nested function of $\mathbf{x}$

The chain rule states:

$$
\frac{\partial f_i}{\partial \mathbf{x}} = \frac{\partial f_i}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial \mathbf{x}}
$$

---

## Backpropagation

Recall from Lecture 03 that: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = P_{y|x} - \mathbf{y} \in \mathbb{R}^K$

**Backpropagation** returns to the application of the chain rule ("multivariate") to compute the gradient of $\mathcal{L}$ w.r.t. $\theta$.

**For an example:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \cdot \frac{\partial \mathbf{z}^{(1)}}{\partial \mathbf{L}^{(0)}} = P_{y|x} - \mathbf{y} \in \mathbb{R}^K
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(0)}} \cdot \frac{\partial \mathbf{z}^{(0)}}{\partial \mathbf{W}^{(0)}} + \frac{\partial \mathcal{L}}{\partial \mathbf{S}^{(0)}} \cdot \frac{\partial \mathbf{S}^{(0)}}{\partial \mathbf{W}^{(0)}} = (P_{y|x} - \mathbf{y}) \cdot \mathbf{a}^{(0)} + \lambda\mathbf{W}^{(0)} \in \mathbb{R}^{K \times D}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(0)}} \cdot \frac{\partial \mathbf{z}^{(0)}}{\partial \mathbf{a}^{(0)}} \cdot \frac{\partial \mathbf{a}^{(0)}}{\partial \mathbf{z}^{(0)}} + \frac{\partial \mathcal{L}}{\partial \mathbf{S}^{(0)}} \cdot \frac{\partial \mathbf{S}^{(0)}}{\partial \mathbf{W}^{(0)}} = (P_{y|x} - \mathbf{y}) \cdot \mathbf{W}^{(1)} \cdot (\mathbf{x} \frac{\mathbf{z}^{(0)} \geq 0}{\mathbf{z}^{(0)} < 0}) \cdot 1 \in \mathbb{R}^D
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \cdot \frac{\partial \mathbf{z}^{(1)}}{\partial \mathbf{a}^{(0)}} \cdot \frac{\partial \mathbf{a}^{(0)}}{\partial \mathbf{z}^{(0)}} + \frac{\partial \mathcal{L}}{\partial \mathbf{S}^{(0)}} \cdot \frac{\partial \mathbf{S}^{(0)}}{\partial \mathbf{W}^{(0)}} = (P_{y|x} - \mathbf{y}) \cdot \mathbf{W}^{(1)} \cdot (\mathbf{x} \frac{\mathbf{z}^{(0)} \geq 0}{\mathbf{z}^{(0)} < 0}) \cdot \mathbf{x} + \lambda\mathbf{W}^{(0)} \in \mathbb{R}^{K \times D}
$$

---

## Summary

- Neural networks introduce non-linearity through activation functions
- Universal Approximation Theorem guarantees expressiveness
- Backpropagation enables efficient gradient computation through the chain rule
- Regularization techniques (L1, L2, Dropout) prevent overfitting
- The bias-variance tradeoff characterizes the fundamental tension between model simplicity and flexibility
- Learning rate scheduling and annealing are critical for efficient training and convergence
- Adaptive methods like Adam provide automatic per-parameter learning rate adjustment

**Next lecture:** We'll apply these neural network fundamentals to language modeling, exploring n-gram models, HMMs, convolutional filtering, energy-based models, and recurrent architectures.

