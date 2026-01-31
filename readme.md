# 🤖 AI Personal Assistant – LLM From Scratch

## 🚀 Overview

This repository is a hands-on exploration of **Large Language Models (LLMs)** built from scratch. It focuses on understanding the internal working of modern language models such as **tokenization, attention mechanisms, GPT-style text generation, pretraining, and fine-tuning** using Python and Jupyter notebooks.

The project is designed for learners who want to move beyond black-box APIs and gain a **deep, practical understanding of how LLMs work internally**.

---

## 🧠 High-Level Architecture

### 🔹 Overall LLM Pipeline

```mermaid
flowchart LR
    A[Raw Text Input] --> B[Tokenizer]
    B --> C[Token IDs]
    C --> D[Embedding Layer]
    D --> E[Transformer Blocks]
    E --> F[Output Head]
    F --> G[Generated Text / Prediction]
🔍 Transformer Block Architecture
Diagram
flowchart TB
    A[Input Embeddings] --> B[Multi-Head Self Attention]
    B --> C[Add & Normalize]
    C --> D[Feed Forward Network]
    D --> E[Add & Normalize]
    E --> F[Output to Next Layer]
🧩 Attention Mechanism Flow
Diagram
flowchart LR
    Q[Query] --> A[Attention Score]
    K[Key] --> A
    V[Value] --> B[Weighted Sum]
    A --> B
    B --> C[Context Vector]
🏗️ GPT-Style Text Generation Architecture
Diagram
flowchart TB
    A[Input Tokens] --> B[Token Embedding]
    B --> C[Positional Encoding]
    C --> D[Masked Multi-Head Attention]
    D --> E[Feed Forward Layer]
    E --> F[Linear + Softmax]
    F --> G[Next Token Prediction]
🔁 Training & Fine-Tuning Workflow
Diagram
flowchart LR
    A[Unlabeled Text Data] --> B[Pretraining LLM]
    B --> C[Base Language Model]
    C --> D[Instruction Dataset]
    D --> E[Fine-Tuning]
    E --> F[Task-Specific Model]
📂 Project Structure
File / Notebook	Description
tokenization_of_data_for_LLM_processing.ipynb	Text tokenization for LLM pipelines
attention_mechanism_with_and_without_training_weights.ipynb	Understanding attention mechanism
GPT_implementation_from_scratch_to_generate_text.ipynb	GPT-style text generation from scratch
Pretraing_model_on_unlabeled_data.ipynb	Pretraining LLM on raw text
finetuning_of_LLM_models_and_use_as_spam_classifier.ipynb	Fine-tuning LLM for classification
AI_personal_trainer_using_LLMs.ipynb	Training workflow experiments
gpt_download.py	Script to download pretrained models
instruction-data.json	Custom instruction-response dataset
loss-plot.pdf	Training loss visualization
accuracy-plot.pdf	Training accuracy visualization
temperature-plot.pdf	Sampling temperature effects
previous_chapters.py	Supporting utilities
the-verdict.txt	Notes / summaries
🛠️ Tech Stack
🐍 Python
<img src="https://img.icons8.com/color/48/python--v1.png"/>
Core programming language for implementing models and pipelines.

📓 Jupyter Notebook
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1280px-Jupyter_logo.svg.png" width="50"/>
Interactive experimentation and visualization.

🔥 PyTorch
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo_icon.svg/256px-Pytorch_logo_icon.svg.png" width="50"/>
Deep learning framework for tensor operations and neural networks.

📊 Matplotlib
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Matplotlib_Logo.svg/1024px-Matplotlib_Logo.svg.png" width="60"/>
Used for plotting training metrics and behavior.

⚙️ Setup & Installation
git clone https://github.com/AnmolRajpoot25/AI_personal_assistant_LLM_from_scratch.git
cd AI_personal_assistant_LLM_from_scratch
Create virtual environment:

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
Install dependencies:

pip install jupyter numpy pandas torch matplotlib scikit-learn
Run notebooks:

jupyter notebook
🎯 Learning Objectives
Understand LLM internals deeply

Implement GPT-style models from scratch

Learn pretraining and fine-tuning pipelines

Visualize attention and training dynamics

Build intuition behind language model behavior

🤝 Contributing
Contributions are welcome!

Add more experiments

Improve documentation

Optimize training workflows

Add evaluation metrics

