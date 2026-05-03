# Bahdanau Attention Neural Machine Translation

Implementation of the paper:

**Neural Machine Translation by Jointly Learning to Align and Translate**

using:
- PyTorch
- GRU-based Seq2Seq
- Bahdanau Attention
- Teacher Forcing
- Attention Visualization

Built and trained on Kaggle GPU notebooks.

---

# Project Overview

This project implements an end-to-end Attention-based Neural Machine Translation (NMT) system following the original Bahdanau Attention paper.

The model translates:
- German → English

using:
- Bidirectional GRU Encoder
- Bahdanau Additive Attention
- GRU Decoder

The goal of this project is to deeply understand:
- Seq2Seq architectures
- Encoder–Decoder systems
- Attention mechanisms
- Neural Machine Translation
- Foundations of modern LLM attention systems

---

# Paper

### Title
Neural Machine Translation by Jointly Learning to Align and Translate

### Authors
- Dzmitry Bahdanau
- Kyunghyun Cho
- Yoshua Bengio

### Paper Link
https://arxiv.org/abs/1409.0473

---

# Architecture

```text
Source Sentence
       ↓
Word Embedding
       ↓
Bidirectional GRU Encoder
       ↓
Encoder Hidden States
       ↓
Bahdanau Attention
       ↓
Context Vector
       ↓
GRU Decoder
       ↓
Target Translation
```

---

# Key Concepts Implemented

## 1. Seq2Seq Learning

Implemented a full sequence-to-sequence translation pipeline.

---

## 2. Bidirectional GRU Encoder

The encoder processes sentences:
- left → right
- right → left

to capture contextual information from both directions.

---

## 3. Bahdanau Attention

Implemented additive attention mechanism where decoder dynamically focuses on important source words during translation.

Core equations:

```math
eij = a(si−1, hj)
```

```math
αij = softmax(eij)
```

```math
ci = Σ αij hj
```

---

## 4. Teacher Forcing

Used teacher forcing during training for stable sequence generation.

---

## 5. Gradient Clipping

Implemented gradient clipping to prevent exploding gradients in RNN training.

---

## 6. Attention Visualization

Generated attention heatmaps to visualize alignment between:
- source words
- generated target words

---

# Dataset

Dataset used:
- Multi30k German-English Translation Dataset

Languages:
- German (source)
- English (target)

---

# Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| NLP Processing | spaCy |
| Dataset Handling | TorchText |
| Visualization | Matplotlib |
| Training Environment | Kaggle GPU |

---

# Model Components

## Encoder
- Embedding Layer
- Bidirectional GRU
- Linear Projection Layer

---

## Attention
- Additive Attention
- Alignment Score Computation
- Softmax Attention Weights

---

## Decoder
- Embedding Layer
- Context Vector Integration
- GRU Decoder
- Output Projection Layer

---

# Training Details

| Parameter | Value |
|---|---|
| Encoder Embedding Dim | 256 |
| Decoder Embedding Dim | 256 |
| Encoder Hidden Dim | 512 |
| Decoder Hidden Dim | 512 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Gradient Clipping | 1 |
| Epochs | 10 |

---

# Features

- Full Seq2Seq implementation from scratch
- Bahdanau Attention mechanism
- GPU training support
- Translation inference
- Attention heatmap visualization
- Teacher forcing
- Gradient clipping
- Model checkpoint saving

---

# Example Translation

## Input (German)

```text
ein mann läuft
```

## Predicted Translation

```text
a man is running
```

---

# Attention Visualization

The model learns dynamic alignment between source and target words.

Example:

| Target Word | Focused Source Word |
|---|---|
| man | mann |
| running | läuft |

---

# Project Structure

```text
bahdanau_attention_nmt/
│
├── bahdanau_attention_nmt.ipynb
├── README.md
├── requirements.txt
└── saved_models/
```

---

# How To Run

## 1. Clone Repository

```bash
git clone https://github.com/ranastudent/bahdanau_attention_nmt.git
```

---

## 2. Install Dependencies

```bash
pip install torch torchtext spacy matplotlib
```

---

## 3. Download spaCy Models

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## 4. Run Notebook

Open:

```text
bahdanau_attention_nmt.ipynb
```

using:
- Jupyter Notebook
- Kaggle
- Google Colab

---

# Learning Outcomes

This project helped build deep understanding of:
- Attention mechanisms
- Encoder–Decoder architectures
- RNN-based NMT systems
- Sequence generation
- Autoregressive decoding
- Foundations of Transformer models

---

# Future Improvements

- Beam Search Decoding
- BLEU Score Evaluation
- Transformer Architecture
- English ↔ Bangla Translation
- Luong Attention
- Subword Tokenization (BPE)
- Mixed Precision Training

---

# Why This Project Matters

Bahdanau Attention was a foundational breakthrough that influenced:
- Transformers
- BERT
- GPT
- Modern Large Language Models

This project recreates that historical architecture from scratch for educational and research purposes.

---

# Author

Reduanul Islam

AI / NLP / Deep Learning Enthusiast

---

# Acknowledgements

Special thanks to:
- PyTorch
- spaCy
- Kaggle
- Authors of the Bahdanau Attention paper

for making open AI research and tooling accessible.
