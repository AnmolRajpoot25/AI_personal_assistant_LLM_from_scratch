# 🤖 AI Personal Assistant — Building an LLM From Scratch

## 🌟 Introduction

This repository is a **deep, hands-on exploration of Large Language Models (LLMs)** built **from scratch**, focusing on *how modern AI systems like GPT actually work internally*.

Instead of treating LLMs as black boxes, this project breaks them down into **clear, understandable components** — from raw text processing to attention mechanisms, GPT-style text generation, pretraining, and fine-tuning on custom instruction data.

This repository is ideal for:
- Students learning NLP and deep learning  
- Engineers curious about LLM internals  
- Anyone aiming to build or understand AI assistants  

---

## 🎯 Project Goals

- Understand **LLM architecture from first principles**
- Implement **tokenization, attention, and GPT-style models**
- Train models on **unlabeled text**
- Fine-tune LLMs using **instruction datasets**
- Visualize **training dynamics and model behavior**
- Build a strong conceptual foundation for AI assistants

---

## 🧠 System Architecture Overview

### 🔹 End-to-End LLM Pipeline

Raw Text

↓

Tokenization

↓

Token IDs

↓

Embedding Layer

↓

Transformer Blocks

↓

Output Head

↓

Generated Text / Prediction


This pipeline mirrors how real-world LLMs process and generate language, step by step.

---

## 🔬 Transformer Block Breakdown

Input Embeddings
↓
Multi-Head Self Attention
↓
Add & Normalize
↓
Feed Forward Network
↓
Add & Normalize
↓
Output to Next Layer


Each transformer block refines contextual understanding while preserving information flow using residual connections.

---

## 🧩 Attention Mechanism Explained

Query ──┐

├──► Attention Scores ──► Weighted Sum ──► Context Vector

Key ─────┘

Value ──────────────────────────┘


The attention mechanism allows the model to dynamically focus on relevant parts of the input sequence, enabling contextual understanding.

---

## 🏗 GPT-Style Text Generation Flow

Input Tokens

↓

Token Embeddings

↓

Positional Encoding

↓

Masked Self Attention

↓

Feed Forward Layer

↓

Linear Layer + Softmax

↓

Next Token Prediction



This autoregressive design enables fluent and coherent text generation.

---

## 🔁 Training & Fine-Tuning Strategy

Unlabeled Text Data
↓
Pretraining Phase
↓
Base Language Model
↓
Instruction Dataset
↓
Fine-Tuning Phase
↓
Task-Specific AI Model


Pretraining builds language understanding, while fine-tuning adapts the model to specific tasks such as classification or instruction following.

---

## 📂 Repository Structure

| File / Notebook | Description |
|-----------------|-------------|
| `tokenization_of_data_for_LLM_processing.ipynb` | Text preprocessing & tokenization |
| `attention_mechanism_with_and_without_training_weights.ipynb` | Attention visualization |
| `GPT_implementation_from_scratch_to_generate_text.ipynb` | GPT-style model |
| `Pretraing_model_on_unlabeled_data.ipynb` | Language model pretraining |
| `finetuning_of_LLM_models_and_use_as_spam_classifier.ipynb` | Fine-tuning & evaluation |
| `AI_personal_trainer_using_LLMs.ipynb` | Training workflows |
| `gpt_download.py` | Pretrained model downloader |
| `instruction-data.json` | Instruction-response dataset |
| `loss-plot.pdf` | Training loss |
| `accuracy-plot.pdf` | Model accuracy |
| `temperature-plot.pdf` | Sampling temperature effects |

---

## 🛠️ Tech Stack

### 🐍 Python
Core language for implementing LLM logic and training workflows.

<img src="https://img.icons8.com/color/48/python--v1.png"/>

---

### 📓 Jupyter Notebook
Interactive experimentation, visualization, and model development.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1280px-Jupyter_logo.svg.png" width="55"/>

---

### 🔥 PyTorch
Used for tensor operations, neural networks, and training loops.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo_icon.svg/256px-Pytorch_logo_icon.svg.png" width="55"/>

---

### 📊 Matplotlib
Visualization of training loss, accuracy, and model behavior.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Matplotlib_Logo.svg/1024px-Matplotlib_Logo.svg.png" width="65"/>

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/AnmolRajpoot25/AI_personal_assistant_LLM_from_scratch.git
cd AI_personal_assistant_LLM_from_scratch
pip install jupyter numpy pandas torch matplotlib scikit-learn
jupyter notebook
```
📈 Key Learnings from This Project
How LLMs tokenize and represent text

Why attention is the core of transformer models

How GPT models generate text autoregressively

Differences between pretraining and fine-tuning

How hyperparameters affect model behavior

🚀 Future Improvements
Add reinforcement learning from human feedback (RLHF)

Build a chat-style personal assistant interface

Integrate vector databases for memory

Add evaluation benchmarks and metrics

🤝 Contributions
Contributions, suggestions, and improvements are welcome!
Feel free to fork the repository and submit pull requests.

