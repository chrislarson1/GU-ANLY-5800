# Week 1 Baseline Checklist – ANLY-5800 Final Project

This checklist is for your **end-of-Week-1** milestone. You can keep it in your repo and check items off as you complete them.

---

## 1. Repository setup

- [ ] Project repo created (or course repo forked/cloned for your group)
- [ ] Clear directory structure (e.g., `data/`, `src/`, `notebooks/`, `experiments/`)
- [ ] `README.md` includes:
  - [ ] Short project description
  - [ ] Environment / dependency info (e.g., `requirements.txt` or `environment.yml`)
  - [ ] How to run the baseline experiment

---

## 2. Data pipeline

- [ ] Dataset(s) chosen and documented:
  - [ ] Source link(s)
  - [ ] Size (number of examples, languages, domains)
- [ ] Download / access scripts implemented (or clearly documented if manual)
- [ ] Preprocessing implemented:
  - [ ] Tokenization / text cleaning
  - [ ] Train / validation / test split defined
- [ ] Sanity checks:
  - [ ] Printed examples of preprocessed inputs/labels
  - [ ] No obvious label/data leakage

---

## 3. Baseline model

- [ ] Baseline approach implemented, for example:
  - [ ] TF-IDF + logistic regression or linear SVM
  - [ ] Zero-shot or prompt-only LLM
  - [ ] Simple RNN/CNN/Transformer model
- [ ] Training script or notebook:
  - [ ] Trains without crashing on a subset of data
  - [ ] Logs loss/metrics during training
- [ ] Evaluation:
  - [ ] Runs on validation set
  - [ ] Reports at least one appropriate metric (accuracy, F1, BLEU, ROUGE, perplexity, etc.)

---

## 4. Jetstream2 / compute (if applicable)

If you plan to use Jetstream2:

- [ ] You can successfully connect to your Jupyter environment (see `project/jetstream2.md`)
- [ ] A small test notebook or script runs on GPU (e.g., quick PyTorch check)
- [ ] You understand where to store data and models (`/workspace/anly5800`)

If using other compute:

- [ ] Environment verified on that platform (local GPU, other cloud, etc.)

---

## 5. Baseline results & Week 2 plan

- [ ] Baseline metrics recorded in your repo (e.g., `experiments/baseline-results.md` or similar)
- [ ] At least one simple plot or table summarizing baseline performance
- [ ] Short progress note (1–2 pages max) including:
  - [ ] What you implemented in Week 1
  - [ ] Baseline performance
  - [ ] **At least two concrete improvements** you will implement in Weeks 2–3

Examples of improvements:

- Architecture changes (deeper model, different attention pattern)
- Finetuning strategy changes (LoRA ranks, learning rates, dataset size)
- Agent capabilities (additional tools, better planning, memory, etc.)

---

Once all of the above are checked, you are in good shape for Week 2, where the focus shifts to your **core model/system** and **main experiments**.


