# Lecture 09: LLM Finetuning, Alignment, Scaling, and Emergence

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

Pretrained language models (from Lecture 08) are powerful but not directly aligned with human intent. This lecture covers the post-pretraining pipeline: supervised finetuning to follow instructions, reinforcement learning from human feedback to align with preferences, and the empirical scaling laws that govern model performance.

## The Alignment Problem

**Observation.** A base language model trained on internet text learns to:
- Complete text in the style of its training data
- Generate plausible continuations
- Perform next-token prediction

But it does NOT inherently:
- Follow instructions ("Translate this to French")
- Provide helpful, harmless, honest responses
- Refuse harmful requests
- Format outputs appropriately

**Solution pipeline:**
1. **Supervised finetuning (SFT):** Teach instruction-following
2. **Reward modeling:** Learn human preferences
3. **Reinforcement learning (RLHF):** Optimize for preferences while maintaining capabilities
4. **Alternative: Direct preference optimization (DPO):** Skip reward modeling

---

## Supervised Finetuning (SFT)

The first step adapts a pretrained model to follow instructions through supervised learning on high-quality demonstrations.

### Problem Setup

**Definition 1.1 (Instruction format).** An instruction example consists of:
- **System prompt** (optional): Role/context (e.g., "You are a helpful assistant")
- **User prompt** $x$: The instruction or question
- **Assistant response** $y^*$: The desired completion

**Example:**
```
System: You are a helpful assistant.
User: Translate "Hello" to French.
Assistant: Bonjour
```

### SFT Objective

**Definition 1.2 (SFT loss).** Let $\pi_\theta(y\mid x)$ be a pretrained causal LM. The SFT objective is standard supervised learning:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y^*)\sim \mathcal{D}}\bigg[\sum_{t=1}^{|y^*|} \log \pi_\theta\big(y^*_t\mid x, y^*_{<t}\big)\bigg]
$$

This is identical to the pretraining objective (Lecture 08), but now:
- Dataset $\mathcal{D}$ contains curated instruction-response pairs
- Responses are high-quality demonstrations of desired behavior
- Format is standardized (system/user/assistant structure)

### Data Quality Matters

**Observation 1.1.** SFT performance is dominated by data quality, not quantity. Key factors:
- **Diversity:** Cover many task types, domains, difficulty levels
- **Correctness:** Responses must be factually accurate
- **Style:** Consistent formatting, tone, helpfulness
- **Safety:** No harmful, biased, or inappropriate content

**Typical SFT dataset size:** 10k-100k high-quality examples (much smaller than pretraining corpus of trillions of tokens).

**Instruction tuning** (Wei et al., 2022) showed that finetuning on diverse instructions dramatically improves zero-shot performance on new tasks.

---

## Preference Modeling

SFT teaches the model to imitate demonstrations, but many responses can be correct with varying quality. Preference data provides richer supervision by comparing responses.

### From Demonstrations to Preferences

**Limitation of SFT.** For prompt "Explain quantum computing," multiple responses exist:
- Response A: Detailed, accurate, well-structured (best)
- Response B: Correct but terse (okay)
- Response C: Verbose, tangential (poor)

SFT treats all as equally valid if they appear in training data. We need to capture quality differences.

**Solution:** Collect pairwise preferences from human annotators.

### Pairwise Preference Data

**Definition 2.1 (Preference annotation).** For prompt $x$, sample two responses $y_1, y_2 \sim \pi_{\text{SFT}}(\cdot \mid x)$ from the SFT model. Human annotators choose which is better, yielding preference pair $(x, y^+, y^-)$ where $y^+ \succ y^-$ means "$y^+$ is preferred to $y^-$."

**Annotation criteria (InstructGPT):**
- Helpfulness: Does it follow the instruction?
- Harmlessness: Is it safe and appropriate?
- Honesty: Is it truthful and acknowledges uncertainty?

### The Bradley-Terry Model

**Definition 2.2 (Bradley-Terry model).** Assume each response has a latent scalar reward $r(y \mid x)$. The probability that $y^+$ is preferred to $y^-$ follows a logistic model:

$$
P\big(y^+ \succ y^- \mid x\big) = \sigma\big(r(y^+ \mid x) - r(y^- \mid x)\big) = \frac{1}{1 + \exp(-(r(y^+\mid x) - r(y^-\mid x)))}
$$

**Intuition:** The probability of preferring $y^+$ increases monotonically with the reward difference. If $r(y^+) \gg r(y^-)$, then $P(y^+ \succ y^-) \approx 1$.

### Reward Model Training

**Parameterization.** Use a neural network $r_\phi(y \mid x)$ (typically initialized from the SFT model) that outputs a scalar reward for each response.

**Architecture:** Start with SFT model, replace final token prediction head with scalar output:

$$
r_\phi(y \mid x) = \mathbf{w}^T \mathbf{h}_{\text{final}} + b
$$

where $\mathbf{h}_{\text{final}}$ is the last hidden state.

**Definition 2.3 (Reward model loss).** Maximize the log-likelihood of observed preferences:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{D}_p} \Big[\log \sigma\big(r_\phi(y^+\mid x) - r_\phi(y^-\mid x)\big)\Big]
$$

**Training:** Use pairwise ranking loss (similar to learning-to-rank in information retrieval). Typical dataset size: 50k-300k preference pairs.

---

## RL with KL Control (RLHF)

We learn a policy $\pi_\theta$ that maximizes reward while staying close to a reference policy $\pi_{\text{ref}}$ (e.g., the SFT model) to preserve helpful behaviors.

**Objective.**

$$
\max_{\theta} \, \mathbb{E}_{x\sim \mathcal{D},\, y\sim \pi_\theta(\cdot\mid x)}\big[ r_\phi(y\mid x) \big] - \beta\, \mathbb{E}_{x\sim \mathcal{D}}\Big[\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big]
$$

The KL term regularizes exploration and anchors style/format; $\beta$ controls the alignment–capability trade-off.

**Policy gradient (sequence-as-action).** With log-derivative trick,

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x, y\sim \pi_\theta}\Big[\nabla_\theta \log \pi_\theta(y\mid x)\, (r_\phi(y\mid x) - \beta\, \Delta_{\text{KL}}(y, x) - b(x))\Big],
$$

where $\Delta_{\text{KL}}$ denotes per-sample KL term (or advantage-style shaping) and $b(x)$ is a baseline.

**PPO surrogate (token-level).** Let $r_t(\theta) = \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{old}}(y_t\mid s_t)}$ and $A_t$ an advantage estimator; the clipped objective is

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}\Big[ \min\big( r_t(\theta) A_t, \, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \Big]
\; + \; \beta\, \mathrm{KL}\big(\pi_\theta\Vert\pi_{\text{ref}}\big) \; + \; \lambda \|\theta\|^2.
$$

---

## Direct Preference Optimization (DPO)

DPO avoids explicit reward learning and RL by directly fitting the policy to pairwise preferences relative to a reference policy.

**Derivation (sketch).** Under a KL-regularized objective with an optimal reward model consistent with Bradley–Terry and with $r(y\mid x) \propto \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, maximizing preference likelihood yields the logistic loss

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y^+,y^-)}\Big[ \log \sigma\big( \beta\,[\underbrace{\log \pi_\theta(y^+\mid x) - \log \pi_\theta(y^-\mid x)}_{\text{policy logit gap}} - \underbrace{\log \pi_{\text{ref}}(y^+\mid x) - \log \pi_{\text{ref}}(y^-\mid x)}_{\text{reference gap}}] \big) \Big].
$$

**Observation.** DPO is stable, simple (no rollouts), and competitive with RLHF given high-quality preference data; $\beta$ tunes adherence to preferences versus the reference style.

---

## Scaling Laws and Compute-Optimal Training

Empirical laws relate model size, dataset size, and compute to validation loss.

**Phenomenological form.** For loss $\mathcal{L}$ (e.g., cross-entropy) and large-scale regimes,

$$
\mathcal{L}(N, D, C) \approx \mathcal{L}_\infty + a\,N^{-\alpha} + b\,D^{-\beta} + c\,C^{-\gamma}, \quad \alpha,\beta,\gamma>0,
$$

where $N$ is parameter count, $D$ tokens seen, and $C$ training compute.

**Compute-optimal allocation.** With FLOPs roughly proportional to $\mathrm{FLOPs} \propto N\cdot D$ for dense decoders, minimizing loss at fixed compute yields a rule of thumb $D^* \propto N$. Oversized models undertrained on tokens waste capacity; modestly sized models trained longer often win ("data-limited" vs. "parameter-limited" regimes).

**Practical implications.**
- Match dataset size to parameter count (token-budgeting).
- Use curriculum/mixture-of-quality datasets; deduplicate to reduce memorization.
- Longer training with proper regularization and schedule can outperform larger short-trained models at fixed compute.

---

## Emergent Phenomena

Large LMs exhibit qualitative shifts in behavior as scale grows.

**Phase transitions.** Certain capabilities (e.g., multi-step arithmetic) show abrupt accuracy increases when $\log$ compute crosses a threshold; indicative of representation reorganizations.

**Double descent.** Test risk as a function of model size or training time can first decrease, then increase near interpolation, then decrease again with further scale; reflects bias–variance and optimization interplay in overparameterized regimes.

**Grokking.** Models may memorize training data with poor generalization for long periods and then suddenly generalize after extended training, often accompanied by compression/structure in internal representations. Regularization and optimization implicitly bias toward simpler solutions that eventually win.

---

## Practical Alignment Recipe

1. Pretrain or start from a strong base decoder (causal LM) sized to token budget.
2. SFT on instruction-formatted high-quality data; mix in rejection-sampled or curated examples.
3. Collect preferences with diverse annotators; train BT reward or apply DPO directly.
4. If using RLHF: stabilize with KL control to $\pi_{\text{ref}}$, small PPO steps, and reward normalization.
5. Evaluate with capability and safety benchmarks; iterate on data quality (it dominates).

---

## Appendix A: Reinforcement Learning Foundations

This appendix reviews core reinforcement learning (RL) concepts that underlie RLHF, DPO, and modern policy optimization methods.

### A.1 Markov decision processes

**Definition A.1 (MDP).** A Markov decision process is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ where:
- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $P(s' \mid s,a)$: transition model
- $r(s,a)$: reward
- $\gamma \in [0,1)$: discount factor

An agent following policy $\pi_\theta(a\mid s)$ induces trajectories $\tau = (s_0,a_0,s_1,a_1,\ldots)$ with return

$$
G_0 = \sum_{t=0}^{\infty} \gamma^t r(s_t,a_t).
$$

**Objective.** Find $\pi_\theta$ that maximizes expected return
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[G_0\big].
$$

In RLHF, we treat the prompt as state, the token sequence as action, and the reward model score as (proxy) return.

### A.2 Value functions and Q-learning

**Definition A.2 (Value functions).** For fixed policy $\pi$,

$$
V^\pi(s) = \mathbb{E}_\pi\big[ G_t \mid s_t = s\big], \qquad
Q^\pi(s,a) = \mathbb{E}_\pi\big[ G_t \mid s_t = s, a_t = a\big].
$$

These satisfy the Bellman equations:

$$
V^\pi(s) = \sum_a \pi(a\mid s)\Big[r(s,a) + \gamma \sum_{s'} P(s'\mid s,a)V^\pi(s')\Big],
$$

$$
Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\sum_{a'} \pi(a'\mid s')Q^\pi(s',a').
$$

**Optimal value functions.** Define

$$
Q^{*}(s,a) = \max_\pi Q^\pi(s,a), \qquad V^{*}(s) = \max_a Q^{*}(s,a).
$$

Then

$$
Q^{*}(s,a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\max_{a'} Q^{*}(s',a').
$$

**Q-learning (tabular).** Given transition samples $(s_t,a_t,r_t,s_{t+1})$, the Q-learning update is

$$
Q_{t+1}(s_t,a_t) \leftarrow Q_t(s_t,a_t) +
\alpha \Big[r_t + \gamma \max_{a'} Q_t(s_{t+1},a') - Q_t(s_t,a_t)\Big],
$$

with step-size $\alpha>0$. Under suitable conditions, $Q_t \to Q^{*}$.

For language modeling, direct tabular Q-learning is intractable due to huge state/action spaces, but the Bellman perspective informs actor–critic and value-based methods used in sequence RL.

### A.3 Score-function (REINFORCE) policy gradients

Policy gradient methods directly optimize $J(\theta)$ over parametrized policies $\pi_\theta$.

**Theorem A.3 (Score-function gradient estimator).** For any differentiable policy $\pi_\theta(a\mid s)$,

$$
\nabla_\theta J(\theta)
 = \mathbb{E}_{\tau \sim \pi_\theta}\Bigg[
 \sum_{t=0}^{\infty}
 \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, G_t
 \Bigg].
$$

This is the **score-function gradient estimator** (SFGE), also known as REINFORCE (Williams, 1992).

**Variance reduction (baselines).** Using any baseline $b(s_t)$ that does not depend on $a_t$,

$$
\mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, b(s_t)\big] = 0,
$$

so we can replace $G_t$ with an advantage term

$$
A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t),
$$

yielding

$$
\nabla_\theta J(\theta)
 = \mathbb{E}\Big[
 \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, A^\pi(s_t,a_t)
 \Big],
$$

which has lower variance. Actor–critic methods approximate $V^\pi$ (critic) and update $\pi_\theta$ (actor).

Generalized advantage estimation (GAE) further trades off bias and variance via a smoothing parameter $\lambda \in [0,1]$; GAE-style estimators are standard in modern policy gradient methods (including PPO in RLHF).

### A.4 Monte Carlo Tree Search (MCTS)

MCTS is a planning algorithm used in environments with a simulator (e.g., games).

**High-level algorithm.** From a root state $s_0$, repeat:

1. **Selection:** Traverse the tree using a selection policy (e.g., UCT) until reaching a leaf.
2. **Expansion:** Add one or more child nodes corresponding to previously unvisited actions/states.
3. **Simulation (rollout):** From the leaf, simulate to a terminal state using a default policy; obtain return $G$.
4. **Backpropagation:** Propagate $G$ up the tree, updating value estimates and visit counts.

**UCT rule.** For node $s$, choose action $a$ maximizing

$$
U(s,a) = \hat{Q}(s,a) + c \sqrt{\frac{\log N(s)}{N(s,a)+1}},
$$

where $\hat{Q}(s,a)$ is an empirical value estimate, $N(s)$ is visit count for state $s$, $N(s,a)$ is visit count for edge $(s,a)$, and $c>0$ controls exploration.

MCTS underlies systems such as AlphaZero, where policy and value networks guide search; conceptually, it is a model-based complement to model-free RL.

### A.5 Trust regions, TRPO, and proximal methods

Vanilla policy gradient updates can be unstable: large steps in parameter space can dramatically change $\pi_\theta$. Trust-region methods constrain each update to keep the new policy close (in KL) to the old one.

**TRPO (Trust Region Policy Optimization).** Optimize a surrogate

$$
\max_\theta \; \mathbb{E}_{s,a\sim \pi_{\text{old}}}
\bigg[
 \frac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)} A_{\text{old}}(s,a)
\bigg]
$$

subject to the KL constraint

$$
\mathbb{E}_{s\sim \pi_{\text{old}}}
\big[\mathrm{KL}(\pi_{\text{old}}(\cdot\mid s)\,\Vert\,\pi_{\theta}(\cdot\mid s))\big]
 \le \delta.
$$

This yields **small, trust-region steps** that improve performance while controlling divergence from $\pi_{\text{old}}$.

**PPO (Proximal Policy Optimization).** PPO replaces the hard KL constraint with a clipped objective:

$$
L^{\text{CLIP}}(\theta)
 = \mathbb{E}\Big[
 \min\big( r_t(\theta) A_t,\,
           \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
 \big)
 \Big],
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$ and $\epsilon>0$ is a small constant. An explicit KL penalty to a reference policy is often added.

PPO is simpler to implement than TRPO, retains the trust-region intuition, and is the de facto standard in modern deep RL—hence its widespread use in RLHF pipelines.

### A.6 Connection to RLHF and DPO

- RLHF uses a **KL-regularized policy objective** together with PPO-style updates:
  - The reward model $r_\phi$ provides per-trajectory or per-token rewards.
  - The KL term to $\pi_{\text{ref}}$ acts as a trust-region regularizer.
  - Advantage estimators (often GAE) and clipping implement a practical proximal update.
- DPO can be viewed as optimizing a related KL-regularized objective **in closed form** at the level of preference pairs:
  - It avoids explicit rollouts and explicit reward models.
  - It still encodes a comparison to a reference policy via log-probability gaps.

Thus, RLHF and DPO sit on top of classical RL ideas: value functions and Bellman equations, score-function policy gradients and variance reduction, trust-region/proximal methods (TRPO/PPO), and preference-based modeling (Bradley–Terry).

---

## References (suggested)

- Ouyang et al. (2022) Instruction following with human feedback
- Christiano et al. (2017) Deep RL from human preferences
- Rafailov et al. (2023) Direct Preference Optimization
- Hoffmann et al. (2022) Chinchilla compute-optimal scaling
- Belkin et al. (2019) Double descent; Power et al. (2022) Grokking
