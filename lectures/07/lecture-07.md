# Lecture 07: Sequence Models and Attention Mechanisms

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

---

## Lesson Plan

- Recurrent Neural Networks (RNNs) revisited
- Sequence-to-Sequence (Seq2Seq) models
- The Information Bottleneck problem
- Bahdanau Attention Mechanism
- Memory Networks
- Neural Turing Machines
- Pointer Networks
- Modern perspectives on attention

---

## Recurrent Connections: A Brief Review

**Recurrent NNs (RNNs)** use **recurrent network** connections that span the **sequence** dimension.

**A traditional RNN consists of:**

$$
\mathbf{h}^{(T)} = \sigma_a\left(\mathbf{U}\mathbf{h}^{(T-1)} + \mathbf{W}\mathbf{x}^{(T)} + \mathbf{b}_h\right)
$$

$$
\mathbf{y}^{(T)} = \sigma_{SM}\left(\mathbf{V}\mathbf{h}^{(T)} + \mathbf{b}_y\right)
$$

**Key Properties:**
- Hidden state $\mathbf{h}^{(t)}$ maintains a summary of sequence history
- Sequential processing: cannot parallelize over time
- Vanishing/exploding gradients in long sequences

**Solutions:** LSTM and GRU architectures with gating mechanisms help maintain long-range dependencies.

---

## The Sequence-to-Sequence Framework

Many NLP tasks involve mapping one sequence to another:
- **Machine Translation:** English sentence → French sentence
- **Summarization:** Long document → Short summary
- **Dialogue:** User utterance → System response

**Challenge:** Input and output sequences have different lengths.

### The Encoder-Decoder Architecture

**Definition 7.1 (Sequence-to-Sequence Model)** A Seq2Seq model consists of two components:

1. **Encoder:** Maps input sequence $\mathbf{x} = (x_1, \ldots, x_T)$ to fixed-size context vector $\mathbf{c}$
2. **Decoder:** Generates output sequence $\mathbf{y} = (y_1, \ldots, y_{T'})$ conditioned on $\mathbf{c}$

---

## Encoder Architecture

The encoder is typically an RNN (LSTM or GRU) that processes the input sequence:

$$
\mathbf{h}^{t,\text{enc}} = f_{\text{enc}}(\mathbf{h}^{t-1,\text{enc}}, \mathbf{x}^t)
$$

**Context Vector:** After processing all inputs, the final hidden state becomes the context:
$$
\mathbf{c} = \mathbf{h}^{T,\text{enc}}
$$

**Bidirectional Encoder:** To capture both past and future context:
$$
\mathbf{h}^{t,\text{enc}} = [\overrightarrow{\mathbf{h}}^t; \overleftarrow{\mathbf{h}}^t]
$$

where $\overrightarrow{\mathbf{h}}^t$ processes forward and $\overleftarrow{\mathbf{h}}^t$ processes backward.

**Final Context:**
$$
\mathbf{c} = [\overrightarrow{\mathbf{h}}^T; \overleftarrow{\mathbf{h}}^1]
$$

---

## Decoder Architecture

The decoder generates output tokens autoregressively:

$$
\mathbf{h}^{t,\text{dec}} = f_{\text{dec}}(\mathbf{h}^{t-1,\text{dec}}, y^{t-1}, \mathbf{c})
$$

$$
P(y^t | y^{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}\mathbf{h}^{t,\text{dec}} + \mathbf{b})
$$

**Initialization:** The decoder's initial hidden state is set from the context:
$$
\mathbf{h}^{0,\text{dec}} = \tanh(\mathbf{W}_c \mathbf{c})
$$

**Training:** Use teacher forcing—feed ground truth $y^{t-1}$ as input at step $t$

**Inference:** Use model's own predictions autoregressively

---

## The Information Bottleneck Problem

**Observation 7.1:** The encoder must compress the entire input sequence into a single fixed-size vector $\mathbf{c}$.

**Problem:** For long sequences, this bottleneck becomes severe:
- All information must pass through $\mathbf{c}$
- Early tokens in input may be forgotten
- Performance degrades with sequence length

**Theorem 7.1 (Information Bottleneck)** The mutual information between input $\mathbf{x}$ and output $\mathbf{y}$ is bounded by the capacity of the bottleneck:

$$
I(\mathbf{x}; \mathbf{y}) \leq I(\mathbf{x}; \mathbf{c})
$$

For a deterministic encoder mapping to fixed-size $\mathbf{c} \in \mathbb{R}^d$, the right side is bounded by $d \log_2 e$ nats.

**Implication:** No matter how sophisticated the encoder/decoder, a fixed-size context vector fundamentally limits information transfer.

---

## Attention: The Solution to the Bottleneck

**Key Insight:** Instead of compressing everything into one vector, let the decoder access all encoder hidden states and dynamically select which to focus on.

**Attention Mechanism:** At each decoding step $t$, compute a weighted combination of all encoder states:

$$
\mathbf{c}^t = \sum_{i=1}^{T} \alpha_{t,i} \mathbf{h}_i^{\text{enc}}
$$

where $\alpha_{t,i}$ is the attention weight indicating how much to focus on input position $i$ when generating output $y^t$.

**Dynamic Context:** Each decoder step gets its own context vector $\mathbf{c}^t$, tailored to what's currently needed.

---

## Bahdanau Attention Mechanism

Introduced by Bahdanau, Cho, and Bengio (2015), this was the first successful attention mechanism for neural machine translation.

### Computing Attention Scores

**Step 1: Alignment Scores**

For each encoder position $i$, compute how well it aligns with decoder state $\mathbf{h}^{t,\text{dec}}$:

$$
e_{t,i} = a(\mathbf{h}^{t-1,\text{dec}}, \mathbf{h}_i^{\text{enc}})
$$

where $a(\cdot, \cdot)$ is an alignment model (small neural network).

**Common Choices for $a(\cdot, \cdot)$:**

1. **Additive (Bahdanau):**
$$
e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{h}^{t-1,\text{dec}} + \mathbf{W}_2 \mathbf{h}_i^{\text{enc}})
$$

2. **Multiplicative (Luong):**
$$
e_{t,i} = \mathbf{h}^{t-1,\text{dec}T} \mathbf{W} \mathbf{h}_i^{\text{enc}}
$$

3. **Dot Product:**
$$
e_{t,i} = \mathbf{h}^{t-1,\text{dec}T} \mathbf{h}_i^{\text{enc}}
$$

---

## Attention Weights and Context

**Step 2: Normalize to Probabilities**

Apply softmax to get attention distribution:
$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
$$

**Properties:**
- $\alpha_{t,i} \geq 0$ for all $i$
- $\sum_{i=1}^{T} \alpha_{t,i} = 1$
- Differentiable with respect to parameters

**Step 3: Weighted Sum**

Compute context vector as weighted average of encoder states:
$$
\mathbf{c}^t = \sum_{i=1}^{T} \alpha_{t,i} \mathbf{h}_i^{\text{enc}}
$$

**Step 4: Incorporate Context**

The decoder state is updated using both previous state and context:
$$
\mathbf{h}^{t,\text{dec}} = f_{\text{dec}}(\mathbf{h}^{t-1,\text{dec}}, y^{t-1}, \mathbf{c}^t)
$$

---

## Attention Visualization

Attention weights $\alpha_{t,i}$ form a matrix $\mathbf{A} \in \mathbb{R}^{T' \times T}$ where:
- Rows correspond to decoder (output) positions
- Columns correspond to encoder (input) positions
- Entry $(t, i)$ shows how much output $y^t$ attends to input $x_i$

**Interpretation:**
- Diagonal patterns: monotonic alignment (e.g., translation)
- Off-diagonal: reordering (e.g., adjective-noun order differs across languages)
- Vertical lines: one input word generates multiple output words
- Horizontal lines: multiple input words compressed into one output

**Linguistic Insights:** Attention visualizations often reveal syntactic and semantic correspondences learned by the model.

---

## Advantages of Attention

1. **No Fixed Bottleneck:** Decoder accesses all encoder states
2. **Better Long Sequences:** Performance doesn't degrade with length
3. **Interpretability:** Attention weights show model's focus
4. **Gradient Flow:** Direct paths from decoder to any encoder state
5. **Flexibility:** Can attend to any subset of inputs

**Empirical Results:** Attention-based models dramatically outperformed vanilla Seq2Seq on machine translation, especially for longer sentences.

---

## Memory Networks

Memory Networks, introduced by Weston et al. (2014), provide explicit memory storage for question answering and reasoning tasks.

### Architecture Components

**Definition 7.2 (Memory Network)** A Memory Network consists of four components:

1. **Memory $\mathbf{M}$:** Stores facts or knowledge as vectors $\{\mathbf{m}_1, \ldots, \mathbf{m}_N\}$
2. **Input Feature Map $I$:** Converts input to internal representation
3. **Generalization $G$:** Updates memory with new information
4. **Output Feature Map $O$:** Produces output from memory and input
5. **Response $R$:** Generates final response

---

## Memory Network Operations

### Input Processing

Given input (e.g., question) $\mathbf{q}$:
$$
\mathbf{u} = I(\mathbf{q})
$$

where $I$ embeds the question into the same space as memory.

### Memory Attention

Compute attention over memory slots:
$$
p_i = \text{softmax}(\mathbf{u}^T \mathbf{m}_i)
$$

Retrieve weighted memory:
$$
\mathbf{o} = \sum_{i=1}^{N} p_i \mathbf{c}_i
$$

where $\mathbf{c}_i$ is the output representation of memory $i$ (can differ from input representation $\mathbf{m}_i$).

---

## Multi-Hop Reasoning

For complex questions requiring multiple reasoning steps:

**Hop 1:** Retrieve first relevant memory
$$
\mathbf{o}_1 = \sum_i p_i^{(1)} \mathbf{c}_i^{(1)}
$$

**Update Query:**
$$
\mathbf{u}_2 = \mathbf{u}_1 + \mathbf{o}_1
$$

**Hop 2:** Retrieve second relevant memory
$$
\mathbf{o}_2 = \sum_i p_i^{(2)} \mathbf{c}_i^{(2)}
$$

**Repeat for $K$ hops**, then generate answer:
$$
\hat{y} = \text{softmax}(\mathbf{W}(\mathbf{u}_{K} + \mathbf{o}_K))
$$

**Example:** Question: "Where is John?" Given facts: "John picked up the ball.", "John went to the garden."
- Hop 1: Attend to "John" mentions
- Hop 2: Find location fact
- Output: "garden"

---

## End-to-End Memory Networks

**Simplified Architecture:** Remove explicit supervision of attention and train end-to-end.

**Advantages:**
- No need for attention labels
- Backprop through entire reasoning chain
- Learns what memories are relevant

**Position Encoding:** To maintain word order within memories:
$$
\mathbf{m}_i = \sum_j \mathbf{l}_j \odot \mathbf{w}_j^{(i)}
$$

where $\mathbf{l}_j$ is a position-dependent weight vector.

**Applications:**
- Question answering (bAbI tasks)
- Dialog systems
- Reading comprehension

---

## Neural Turing Machines

Neural Turing Machines (NTMs), introduced by Graves et al. (2014), extend memory networks with read/write operations inspired by classical Turing machines.

### Core Idea

**Differentiable Memory:** A matrix $\mathbf{M}_t \in \mathbb{R}^{N \times M}$ where:
- $N$ is number of memory locations
- $M$ is memory vector size
- All operations differentiable → trainable via backprop

**Controller:** An RNN/LSTM that produces read/write instructions

---

## NTM Reading

**Attention-Based Read:** At time $t$, controller produces attention weights $\mathbf{w}_t^r \in [0,1]^N$ with $\sum_i w_{t,i}^r = 1$.

Read vector:
$$
\mathbf{r}_t = \sum_{i=1}^{N} w_{t,i}^r \mathbf{M}_t[i]
$$

**Multiple Read Heads:** Can have $H$ independent read heads for parallel memory access:
$$
\mathbf{r}_t = [\mathbf{r}_t^{(1)}; \ldots; \mathbf{r}_t^{(H)}]
$$

---

## NTM Writing

**Two Operations:**

1. **Erase:** Remove old information
$$
\tilde{\mathbf{M}}_t[i] = \mathbf{M}_{t-1}[i] \odot (\mathbf{1} - w_{t,i}^w \mathbf{e}_t)
$$

where $\mathbf{e}_t \in [0,1]^M$ is the erase vector (produced by controller).

2. **Add:** Write new information
$$
\mathbf{M}_t[i] = \tilde{\mathbf{M}}_t[i] + w_{t,i}^w \mathbf{a}_t
$$

where $\mathbf{a}_t \in \mathbb{R}^M$ is the add vector.

**Combined:**
$$
\mathbf{M}_t[i] = \mathbf{M}_{t-1}[i] \odot (\mathbf{1} - w_{t,i}^w \mathbf{e}_t) + w_{t,i}^w \mathbf{a}_t
$$

---

## NTM Addressing Mechanisms

The controller produces attention weights via two mechanisms:

### Content-Based Addressing

Focus on memory locations similar to a key $\mathbf{k}_t$:
$$
w_t^c[i] = \frac{\exp(\beta_t \cdot \text{cosine}(\mathbf{k}_t, \mathbf{M}_t[i]))}{\sum_j \exp(\beta_t \cdot \text{cosine}(\mathbf{k}_t, \mathbf{M}_t[j]))}
$$

where $\beta_t > 0$ controls sharpness.

### Location-Based Addressing

**Interpolation:** Blend content addressing with previous attention:
$$
\mathbf{w}_t^g = g_t \mathbf{w}_t^c + (1 - g_t) \mathbf{w}_{t-1}
$$

where $g_t \in [0,1]$ is the interpolation gate.

**Shift:** Allow attention to move (e.g., sequential access):
$$
\tilde{w}_t[i] = \sum_j w_t^g[j] s_t[(i - j) \mod N]
$$

where $\mathbf{s}_t$ is a shift distribution (e.g., shift by -1, 0, or +1).

**Sharpen:** Focus attention:
$$
w_t[i] = \frac{\tilde{w}_t[i]^{\gamma_t}}{\sum_j \tilde{w}_t[j]^{\gamma_t}}
$$

where $\gamma_t \geq 1$ controls sharpness.

---

## NTM Controller

**LSTM Controller:** Receives:
- External input $\mathbf{x}_t$
- Previous read vectors $\mathbf{r}_{t-1}$

Produces:
- Read/write attention parameters
- Erase/add vectors
- Output $\mathbf{y}_t$

**Complete System:**
$$
\mathbf{h}^t, \mathbf{y}^t, \{\text{memory ops}\} = \text{Controller}(\mathbf{x}^t, \mathbf{r}^{t-1}, \mathbf{h}^{t-1})
$$

**End-to-End Training:** All parameters learned jointly via backpropagation through time.

---

## NTM Capabilities

**Algorithmic Tasks:**
- **Copy:** Read input sequence into memory, then output it
- **Repeat Copy:** Output sequence multiple times
- **Associative Recall:** Given key, retrieve associated value
- **Priority Sort:** Sort sequences by priority values
- **Dynamic N-Grams:** Predict sequences with varying context

**Key Advantage:** Unlike standard RNNs, NTMs can learn algorithms that generalize to longer sequences than seen during training.

**Limitations:**
- Computationally expensive ($O(N)$ per timestep for $N$ memory locations)
- Difficult to train (many interacting components)
- Attention can be unstable

---

## Pointer Networks

Pointer Networks (Vinyals et al., 2015) solve problems where the output is a sequence of positions in the input.

### Motivation

**Problem Class:** Output elements are drawn from the input:
- **Convex Hull:** Select boundary points from set of 2D points
- **Traveling Salesman:** Output tour is a permutation of input cities
- **Delaunay Triangulation:** Connect input points
- **Text Summarization:** Extract sentences from document

**Challenge:** Output vocabulary is input-dependent and can vary in size.

---

## Pointer Network Architecture

**Core Idea:** Use attention mechanism as a pointer to select input elements.

**Encoder:** Process input sequence $\mathbf{x} = (x_1, \ldots, x_n)$:
$$
\mathbf{h}_i^{\text{enc}} = \text{LSTM}(\mathbf{h}_{i-1}^{\text{enc}}, x_i)
$$

**Decoder:** At step $t$, produce attention over input positions:
$$
u_t^i = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{h}_i^{\text{enc}} + \mathbf{W}_2 \mathbf{h}^{t,\text{dec}})
$$

$$
p(y^t = i | y^1, \ldots, y^{t-1}, \mathbf{x}) = \text{softmax}(u_t)_i
$$

**Key Difference from Standard Attention:** The output IS the attention distribution (pointer), not a weighted combination.

---

## Training Pointer Networks

**Supervised Learning:** Given input-output pairs $(\mathbf{x}, \mathbf{y}^*)$ where $\mathbf{y}^* = (y^{1*}, \ldots, y^{m*})$ is a sequence of input indices:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{m} \log p(y^{t*} | y^{1*}, \ldots, y^{t-1*}, \mathbf{x}; \theta)
$$

**Inference:** Use beam search or greedy decoding:
$$
\hat{y}^t = \arg\max_i p(y^t = i | \hat{y}^1, \ldots, \hat{y}^{t-1}, \mathbf{x})
$$

---

## Pointer Network Applications

### Convex Hull

**Input:** $n$ points in 2D plane
**Output:** Sequence of point indices forming convex hull
**Training:** Supervised with ground-truth hulls

**Generalization:** Model trained on $n \leq 50$ can solve for $n > 50$ (algorithmic generalization!)

### Traveling Salesman Problem (TSP)

**Input:** $n$ cities with coordinates
**Output:** Tour (permutation) minimizing total distance
**Training:** Supervised (expensive) or reinforcement learning

**Results:** Near-optimal solutions, especially with beam search

### Extractive Summarization

**Input:** Document sentences
**Output:** Sequence of sentence indices to extract
**Advantage:** Output guaranteed to be grammatical (extracted from source)

---

## Comparison of Memory-Augmented Architectures

| Model | Memory | Addressing | Primary Use Case |
|-------|--------|------------|-----------------|
| **Attention (Bahdanau)** | Encoder hidden states | Learned (soft) | Seq2Seq tasks |
| **Memory Networks** | Explicit facts/knowledge | Content-based (soft) | QA, reasoning |
| **Neural Turing Machines** | Differentiable matrix | Content + location | Algorithm learning |
| **Pointer Networks** | Input sequence | Attention as output | Combinatorial problems |

**Common Thread:** All use attention mechanisms to access external information beyond the fixed capacity of hidden states.

---

## Information Flow Comparison

### Standard Seq2Seq (No Attention)
$$
\mathbf{x} \xrightarrow{\text{Encoder}} \mathbf{c} \xrightarrow{\text{Decoder}} \mathbf{y}
$$

Bottleneck: All information through fixed $\mathbf{c}$

### Attention-Based Seq2Seq
$$
\mathbf{x} \xrightarrow{\text{Encoder}} \{\mathbf{h}_1, \ldots, \mathbf{h}_T\} \xrightarrow{\text{Attention}} \mathbf{c}^t \text{ (dynamic)} \xrightarrow{\text{Decoder}} \mathbf{y}
$$

No bottleneck: Decoder accesses all encoder states

### Memory Networks
$$
\text{Facts} \rightarrow \{\mathbf{m}_1, \ldots, \mathbf{m}_N\} \xleftarrow{\text{Multi-hop Attention}} \mathbf{q} \rightarrow \text{Answer}
$$

Explicit memory: Reasoning over stored knowledge
