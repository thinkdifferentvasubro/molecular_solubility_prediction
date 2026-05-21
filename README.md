# Molecular Solubility Prediction using Graph Neural Networks + LLM Interpretation

An end-to-end deep learning system for predicting molecular solubility using molecular graph representations and interpretable AI outputs.

This project began with a simple question:

> *Can we teach neural networks to understand molecules the way chemists do?*

Instead of treating molecules as plain numerical vectors, this system represents them as **graphs** — where atoms become nodes and chemical bonds become edges. The project explores multiple graph neural network architectures before arriving at the finalsolution using a **Directed Message Passing Neural Network (D-MPNN)**.

The result is a complete AI pipeline capable of:
- Predicting molecular log solubility values
- Learning atom-level chemical relationships
- Explaining predictions in human-readable chemical language using an LLM
- Running through an automated CI/CD and Dockerized deployment pipeline

---

# Project Story

The project initially started with traditional Graph Neural Network architectures:

### 1. Graph Convolutional Networks (GCN)
The first implementation used Graph Convolution Networks for molecular representation learning.

While GCNs performed reasonably well, they introduced a major limitation:

- Difficulty integrating richer edge-level and directional chemical information
- Limited flexibility for incorporating additional molecular interaction features
- Message aggregation became overly simplified for complex molecular structures

This reduced the model's ability to capture nuanced chemical relationships.

---

### 2. Graph Attention Networks (GAT)
The next attempt involved Graph Attention Networks to improve representation learning through attention mechanisms.

Although GATs improved feature weighting, they introduced new problems:

- Training instability and attention fluctuation ("tottering")
- High sensitivity to graph size and connectivity
- Increased computational complexity
- Less stable convergence during experimentation

The attention mechanism sometimes over-focused on noisy structural relationships, reducing consistency.

---

### 3. Final Architecture — Directed Message Passing Neural Network (D-MPNN)

To overcome these limitations, the architecture was redesigned using a **Directed Message Passing Neural Network (D-MPNN)**.

D-MPNN processes molecular graphs through directed bond-level message propagation, allowing the model to learn more chemically meaningful representations.

This approach provided:
- Better molecular feature extraction
- Stable training behavior
- Stronger atom-bond interaction learning
- Improved generalization on molecular property prediction tasks

The final model was implemented using **Chemprop** and **PyTorch** for optimized molecular deep learning workflows.

---

# Features

## Molecular Graph Learning
- Directed Message Passing Neural Network (D-MPNN)
- Bond-aware message propagation
- Atom-level representation learning
- Molecular graph encoding using RDKit

## Solubility Prediction
- Predicts molecular log solubility values
- Regression-based molecular property prediction
- Chemprop-powered molecular deep learning pipeline

## LLM Interpretation Layer
Integrated Google GenAI through LangChain to convert raw numerical outputs into understandable explanations.

---

# CI/CD Pipeline

The project includes a complete CI/CD workflow for automated testing and deployment.

### Pipeline Workflow

```text
Code Push
   │
   ▼
GitHub Actions Trigger
   │
   ▼
Run Automated Tests
   │
   ▼
Build Docker Image
   │
   ▼
Save Docker Artifact
```

# Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Molecular AI | Chemprop |
| Cheminformatics | RDKit |
| API Framework | FastAPI |
| UI | Gradio |
| LLM Integration | LangChain, Google GenAI |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

# Model Experimentation

| Model | Outcome |
|---|---|
| Graph Convolution Network (GCN) | Limited support for richer molecular feature integration |
| Graph Attention Network (GAT) | Training instability and attention tottering |
| Directed Message Passing Neural Network (D-MPNN) | Final production architecture with stable and superior performance |

# API, Deployment & Interface

The project also includes a lightweight production-ready serving stack:

- **FastAPI** for exposing prediction endpoints through REST APIs
- **Gradio** for an interactive web interface to test molecular predictions visually
- **Docker** for containerized deployment and environment reproducibility

This allows the model to be served, tested, and deployed consistently across different systems.

# Author
Vasu
