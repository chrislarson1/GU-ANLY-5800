# ANLY-5800: Natural Language Processing

*Updated 09/08/25*

### Course Information

| Item              | Details               |
|-------------------|-----------------------|
| **Course**        | ANLY-5800             |
| **Semester**      | Fall 2025             |
| **Instructor**    | [Chris Larson](https://www.linkedin.com/in/clrsn/) |
| **Credits**       | 3                     |
| **Prerequisites** | None                  |
| **Location**      | Car Barn 309          |
| **Time**          | Tue 3:30-6:00 pm EST  |
| **Office Hours**  | Virtual               |

---

### Course Overview

Natural language processing (NLP) lies at the heart of modern information systems. Over the last 30 years, it has transformed how humans acquire knowledge, interact with computers, and interact with other humans, multiple times over. This course presents these advancements through the lens of the machine learning methods that have enabled them. We explore how language understanding is framed as a tractable inference problem through *language modeling*, and trace the evolution of NLP from classical methods to the latest neural architectures, reasoning systems, and AI agents.

***What's new for Fall Semester 2025?***

- Expanded focus on LLM search and retrieval.
- Expanded focus on the practical and formal aspects of LLM reasoning and AI *Agents*.
- Expanded coverage of the latest NN architectures, including non-attention based models.
- Removed Labs, and have rolled some of that content into Assignments.

---

### Prerequisites

While this course has no course prerequisites, it is designed for students with mathematical maturity that is typically gained through course work in linear algebra, probability theory, first order optimization methods, and basic programming. The archetypal profile is a graduate or advanced undergraduate student in CS, math, engineering, or information sciences. But there have been many exceptions; above all other indicators, students displaying a genuine interest in the material tend to excel in the course. To assist with filling any gaps in the aforementioned technical areas, I devote the entire first lecture to mathematical concepts and tools that will be used throughout the class.

---

### Reference Texts

Many of the topics covered in this course have not been fully exposited in textbooks, and so in this course we make direct reference to papers from the literature. With that said, below are three excellent reference texts that cover a good portion of the topics in lectures 1-7.

1. [Jacob Eisenstein. Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
2. [Dan Jurafsky, James H. Martin. Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3)
3. [Ian Goodfellow, Yoshua Bengio, & Aaron Courville. Deep Learning](https://www.deeplearningbook.org)

---

### JetStream2 Access
As part of this course, you will have access to Jupyter notebooks with A100s (40GB) hosted on the [JetStream2 cluster](https://jetstream-cloud.org/documentation-support/index.html). This is a shared resource and will be made available ahead of the first assignment.

---

### Schedule

Date | Lecture | Topics | Key Readings |
|------|---------|---------|--------------|
| Sep 02 | **Lecture 1: Mathematical Foundations** | - Theorems in linear algebra<br>- Probability and Information theory<br>- Statistical parameter estimation<br>- Zipf's law | [Linear Algebra Done Right (Axler, 2015)](https://linear.axler.net/)<br>[Elements of Information Theory (Cover & Thomas, 2006)](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X)<br>[Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf)<br>[Zipf's Law in Natural Language (Piantadosi, 2014)](https://link.springer.com/article/10.3758/s13423-014-0585-6) |
| Sep 09 | **Lecture 2: Decision Boundary Learning** | - The Perceptron<br>- Support Vector Machines<br>- Kernel methods <br>- Regularization and generalization theory | [Pattern Recognition and Machine Learning (Bishop, 2006)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)<br>[Support Vector Machines (Cortes & Vapnik, 1995)](https://link.springer.com/article/10.1007/BF00994018)<br>[Statistical Learning Theory (Vapnik, 1998)](https://link.springer.com/book/9780387987804) |
| Sep 16 | **Lecture 3: Parameter Estimation Methods** | - Maximum likelihood estimation<br>- Discriminative modeling & softmax regression<br>- Generative modeling & Naive Bayes<br>- Maximum a posteriori estimation | [Machine Learning: A Probabilistic Perspective (Murphy, 2012)](https://probml.github.io/pml-book/)<br>[Pattern Recognition and Machine Learning (Bishop, 2006)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)<br>[Bayesian Data Analysis (Gelman et al., 2013)](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf) |
| Sep 23 | **Lecture 4: Distributional Semantics** | - TF-IDF and PMI<br>- Latent Semantic Analysis<br>- Latent Dirichlet Allocation <br>- Word2Vec | [Probabilistic Topic Models (Blei, 2012)](https://www.cs.columbia.edu/~blei/papers/Blei2012.pdf)<br>[Dynamic Topic Models (Blei & Lafferty, 2006)](https://dl.acm.org/doi/10.1145/1143844.1143859)<br>[Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) |
| Sep 30 | **Lecture 5: Neural Networks** | - Artificial neural networks<br>- Backpropagation algorithm<br>- Gradient descent<br>- Regularization methods<br>- Bias-variance tradeoff<br>- Learning rate scheduling and annealing | [Deep Learning (Goodfellow et al., 2016)](https://www.deeplearningbook.org/)<br>[Learning representations by back-propagating errors (Rumelhart et al., 1986)](https://www.nature.com/articles/323533a0)<br>[Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) |
| Oct 07 | **Lecture 6: Language Modeling** | - n-gram models<br>- HMMs<br>- Convolutional filtering<br>- EBMs and Hopfield Networks<br>- Recurrent networks<br>- Autoregression | [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)<br>[A Guide to the Rabiner HMM Tutorial (Rabiner, 1989)](https://ieeexplore.ieee.org/document/18626)<br>[An Empirical Study of Smoothing Techniques (Chen & Goodman, 1999)](https://www.aclweb.org/anthology/J99-2004/)<br>[Energy-Based Models (LeCun et al., 2006)](https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf)<br>[Neural networks and physical systems with emergent collective computational abilities (Hopfield, 1982)](https://www.pnas.org/doi/10.1073/pnas.79.8.2554) |
| Oct 14 | **Lecture 7: Sequence Models** | - Seq2Seq models<br>- Bahdanau attention<br>- Information bottleneck<br>- Neural Turing Machines<br>- Pointer networks<br>- Memory networks | [Sequence to Sequence Learning (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)<br>[Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)<br>[The Information Bottleneck Method (Tishby et al., 1999)](https://arxiv.org/abs/physics/0004057)<br>[Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401)<br>[Pointer Networks (Vinyals et al., 2015)](https://arxiv.org/abs/1506.03134)<br>[Memory Networks (Weston et al., 2014)](https://arxiv.org/abs/1410.3916) |
| Oct 21 | **Lecture 8: Transformers** | - Self-attention<br>- Scaled dot-product attention<br>- Multi-head attention<br>- Tokenization schemes <br>- Non-causal language models<br>- Causal language models | [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)<br>[Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)<br>[SentencePiece: A simple and language independent subword tokenizer (Kudo, 2018)](https://arxiv.org/abs/1808.06226)<br>[The Annotated Transformer (Rush, 2018)](http://nlp.seas.harvard.edu/2018/04/03/attention.html) |
| Oct 28 | **Lecture 9: LLM Finetuning, Alignment, Scaling, and Emergence** | - Supervised finetuning<br>- Instruction tuning<br>- Reinforcement learning fundamentals<br>- RLHF and DPO<br>- Scaling laws and compute-optimal training<br>- Phase transitions and grokking | [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)<br>[Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)<br>[Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)<br>[Chinchilla Scaling Laws (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)<br>[Double Descent (Belkin et al., 2019)](https://www.pnas.org/doi/10.1073/pnas.1903070116)<br>[Grokking: Generalization Beyond Overfitting (Power et al., 2022)](https://arxiv.org/abs/2201.02177) |
| Nov 04 | **Lecture 10: Modern Network Architectures** |- Mixture of Experts<br>- Sparsely activated models<br>- State Space models <br>- Joint Energy-Embedding models<br>- Contrastive learning | [Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)<br>[S4: State Space Layers for Sequence Modeling (Gu et al., 2021)](https://arxiv.org/abs/2111.00396)<br>[Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)<br>[JEM: Joint Energy-based Models (Grathwohl et al., 2019)](https://arxiv.org/abs/1912.03263)<br>[SimCSE: Simple Contrastive Learning of Sentence Embeddings (Gao et al., 2021)](https://arxiv.org/abs/2104.08821) |
| Nov 11 | **Lecture 11: Cross-Domain Applications** | - Vision models<br>- ASR and speech models<br>- Multi-modal models<br>- World models | [CLIP: Learning Transferable Visual Models (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)<br>[Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)<br>[DALL·E 2 (Ramesh et al., 2022)](https://arxiv.org/abs/2204.06125)<br>[World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122) |
| Nov 18 | **Lecture 12: Reasoning** | - Chain-of-Thought reasoning<br>- Tree-of-Thought search<br>- Causal inference and counterfactuals<br>- Program synthesis and self-debugging<br>- Inference-time scaling | [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)<br>[Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)<br>[The Book of Why (Pearl & Mackenzie, 2018)](https://bayes.cs.ucla.edu/WHY/)<br>[Reflexion: Language Agents with Verbal RL (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366) |
| Nov 25 | **Lecture 13: Search & Retrieval** | - Hierarchical retrieval<br>- Graph retrieval<br>- Multi-hop reasoning<br>- Adaptive retrieval strategies<br>- Contradiction detection & resolution | [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)<br>[Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)<br>[HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA (Yang et al., 2018)](https://arxiv.org/abs/1809.09600)<br>[Graph RAG (Hu et al., 2024)](https://arxiv.org/abs/2405.16506) |
| Dec 02 | **Lecture 14: Agents** | - Tools and function calling<br>- Model Context Protocol<br>- ReAct framework<br>- Experience replay and meta-learning<br>- Adversarial robustness<br>- Prompt injection attacks | [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)<br>[Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)<br>[Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/learn/architecture)<br>[MAML: Model-Agnostic Meta-Learning (Finn et al., 2017)](https://arxiv.org/abs/1703.03400)<br>[Prompt Injection Attacks (Greshake et al., 2023)](https://arxiv.org/abs/2302.12173) |
| Dec 09 | **Lecture 15: Training and Inference Computation** | - Quantization<br>- Model compression<br>- Advanced attention techniques<br>- KV-cache optimization<br>- Parallelism (data/tensor/pipeline)<br>- Inference-time scaling | [GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)<br>[QLoRA: Efficient Finetuning (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)<br>[FlashAttention 2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)<br>[PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)<br>[Speculative Decoding (Chen et al., 2023)](https://arxiv.org/abs/2211.17192) |
| Dec 16 | **Lecture 16: Final Project Presentations** |  | |

*Note: Lecture slides/notes are typically published the day of the lecture.*

---

### Deliverables

| Assignment | Weight | Group Size | Due |
|:------------:|:--------:|:------------:|:-----:|
| [Assignment 1]() | 10% | individual | - |
| [Assignment 2]() | 10% | individual | - |
| [Assignment 3]() | 10% | individual | - |
| [Assignment 4]() | 5% | individual | - |
| [Exam 1]() | 15% | individual | - |
| [Exam 2]() | 15% | individual | - |
| [Exam 3]() | 15% | individual | - |
| [Final Project]() | 20% | groups ≤ 4 | - |

---

### Grading Scale
| Grade | Cutoff ($\ge$)    |
|-------|-------------------|
|  A    |       92.5        |
|  A-   |       90          |
|  B+   |       87          |
|  B    |       83          |
|  B-   |       80          |
|  C+   |       77          |
|  C    |       73          |
|  C-   |       70          |
|  F    |        0          |

---

### Communication
Course content will be published to this [GitHub repository](https://github.com/chrislarson1/GU-ANLY-5800), while all deliverables will be submitted through [Canvas](https://georgetown.instructure.com/courses/213744). We also have a dedicated Discord server, which is the preferred forum for all course communications. **Please join our [Discord](https://discord.gg/uztAhx8P) server at your earliest convenience. In order for the teaching staff to associate your GU, GH, and Discord profiles, please enter your information into this [table](https://docs.google.com/spreadsheets/d/1MmJepoJmrjDOULWWtgQBb1BS7SvfRn70_MfWoeUH2Qg/edit?usp=sharing) to gain access to course materials and communications.**

---

### Course Policies

#### Attendance

Class attendance is required.

#### Tardy Submissions
- Assignments submitted within 24 hours of deadline: 10% penalty
- Assignments submitted within 48 hours of deadline: 35% penalty
- Assignments submitted after 48 hours: not accepted without prior approval

#### Use of AI
- You are encouraged to use language models as an aid in your assignments and final project. However, if a work submission contains LLM text verbatim, it will be rejected. You must submit your own work.
- Exams are open-note, but closed-book/internet.

#### Academic Integrity
All submissions in this class must be your original work. Plagiarism or academic dishonesty will result in course failure and potential disciplinary action.

---

### FAQs

***I'm not sure if I should take this class. How should I decide?***

If you are still deciding if anly-5800 is right for you, feedback from former students may be helpful. Over the past four years, ~200 students have taken the course and I've received enough feedback to give you the TL;DR:

- The course has been characterized as **challenging**, primarily due to the breadth and depth of concepts and tools covered, many of which are new to students.

- The course has been characterized as **rewarding**, with students feeling a sense of accomplishment after completing it. There have been a few common themes:
  - Students attributed improved performance in technical job interviews to this class.
  - Students mentioned new direction and insight into their own graduate research.
  - Students reported an improved ability to craft compelling research statements in graduate school applications.

- A minority of students have provided critical feedback. There have been a few common themes:
  - Students mentioned that course material and/or instruction was **overly theoretical**, and **not aimed at the practioner**.
  - Many students have mentioned that the course was too **time consuming**.
  - A small number of students have opted to **drop** the course.

<br>

***I am a law student. Can I enroll in ANLY-5800?***

If you are interested in LLMs and AI, you are more than welcome to attend lectures. However, to enroll in anly-5800, you will need at least some background in the areas mentioned in the [prerequisites section](#prerequisites). Please contact me if you have a non traditional background but feel you might meet these requirements.
