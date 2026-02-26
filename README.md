# Domain-Specific LLM Fine-Tuning & RAG System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)

---

## Overview

This project implements a domain-adapted Large Language Model (LLM) pipeline combining **parameter-efficient fine-tuning (LoRA)** with **Retrieval-Augmented Generation (RAG)** for knowledge-grounded text generation.

Inspired by practical industrial AI use cases where document-grounded reasoning and low hallucination rates are critical requirements.

**Key objectives:**
- Adapt an open-source transformer model to structured domain reasoning tasks
- Reduce hallucinations through knowledge-grounded retrieval at inference time
- Benchmark fine-tuned vs base model performance quantitatively
- Establish a reproducible, modular experimentation workflow

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Query                        │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   SentenceTransformer   │  ← Embedding Layer
          │      (Encoder)          │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      FAISS Index        │  ← Vector Store
          │   Top-k Retrieval       │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Context Injection     │  ← RAG Prompt Builder
          │   (Query + Documents)   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   LoRA Fine-Tuned LLM   │  ← TinyLlama / Mistral
          │   (PEFT Adapted)        │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Generated Response    │
          └─────────────────────────┘
```

### Component Breakdown

**Base Model**
- Open-source causal language model: TinyLlama / Mistral 7B
- Loaded via HuggingFace Transformers with configurable quantization

**Parameter-Efficient Fine-Tuning (LoRA)**
- LoRA adaptation via the `peft` library
- Targeted injection into attention projection layers (`q_proj`, `v_proj`)
- Drastically reduced trainable parameter footprint vs full fine-tuning

**Retrieval-Augmented Generation (RAG)**
- Document encoding with `sentence-transformers`
- FAISS vector index for fast approximate nearest-neighbor search
- Top-k context retrieval injected directly into generation prompt

**Evaluation Pipeline**
- Side-by-side comparison: Base vs LoRA-Tuned vs LoRA + RAG
- Qualitative output inspection
- Structured hallucination assessment
- Prompt ablation experiments

---

## Repository Structure

```
domain-llm-rag-system/
│
├── data/
│   ├── train/                    # Domain-specific training corpus
│   └── knowledge_base/           # Documents for RAG indexing
│
├── src/
│   ├── finetune.py               # LoRA fine-tuning pipeline
│   ├── rag_pipeline.py           # FAISS index + retrieval logic
│   ├── evaluate.py               # Benchmark and comparison utilities
│   └── dataset.py                # Dataset loading and preprocessing
│
├── notebooks/
│   └── experiments.ipynb         # Exploratory analysis and ablations
│
├── assets/
│   └── architecture_diagram.png  # System architecture visual
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/domain-llm-rag-system.git
cd domain-llm-rag-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements (`requirements.txt`):**

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
datasets>=2.14.0
numpy>=1.24.0
accelerate>=0.24.0
```

> For GPU-accelerated FAISS: replace `faiss-cpu` with `faiss-gpu`

---

## Training (LoRA Fine-Tuning)

```bash
python src/finetune.py
```

**Configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model_name` | `TinyLlama/TinyLlama-1.1B` | Base HuggingFace model |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--lora_dropout` | `0.05` | LoRA dropout rate |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--lr` | `2e-4` | Learning rate |
| `--device` | `cuda` | Training device |

**Example:**

```bash
# Fine-tune TinyLlama with LoRA
python src/finetune.py --model_name TinyLlama/TinyLlama-1.1B --lora_r 16 --epochs 3

# Fine-tune Mistral 7B (GPU recommended)
python src/finetune.py --model_name mistralai/Mistral-7B-v0.1 --lora_r 8 --batch_size 2
```

---

## RAG Pipeline

```bash
python src/rag_pipeline.py
```

**Pipeline steps:**

1. **Encode** — Documents in `data/knowledge_base/` encoded with `SentenceTransformer`
2. **Index** — FAISS index built over document embeddings
3. **Retrieve** — Top-k most relevant documents fetched for each input query
4. **Inject** — Retrieved context prepended to generation prompt
5. **Generate** — Grounded response produced by fine-tuned LLM

**Configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--top_k` | `3` | Number of retrieved documents |
| `--embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `--index_path` | `data/faiss.index` | Path to saved FAISS index |

---

## Evaluation

```bash
python src/evaluate.py
```

**Comparison matrix:**

| Variant | Fine-Tuned | Retrieval | Expected Behavior |
|---|---|---|---|
| Base LLM | ❌ | ❌ | General-purpose, prone to hallucination |
| LoRA-Tuned | ✅ | ❌ | Domain-adapted, reduced off-topic outputs |
| LoRA + RAG | ✅ | ✅ | Knowledge-grounded, lowest hallucination rate |

**Current metrics:**
- Qualitative side-by-side output inspection
- Response relevance scoring

**Planned metrics:**
- ROUGE / BLEU scoring
- Factual consistency via NLI models
- Automated hallucination detection (SelfCheckGPT)

---

## Experiment Results

> Early-stage experiments. Full benchmark results in progress.

| Model Version | Fine-Tuning | Retrieval | Notes |
|---|---|---|---|
| Base LLM | ❌ | ❌ | Baseline reference |
| LoRA-Tuned | ✅ | ❌ | Domain-specific adaptation |
| LoRA + RAG | ✅ | ✅ | Knowledge-grounded generation |

Hyperparameter sweeps (`lora_r`, `top_k`, `lr`) and prompt ablation experiments are actively in progress.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.0+ | Core deep learning framework |
| Transformers | 4.35+ | LLM loading, tokenization, training |
| PEFT | 0.6+ | LoRA parameter-efficient fine-tuning |
| FAISS | 1.7+ | Vector similarity search |
| Sentence-Transformers | 2.2+ | Document and query embeddings |
| Accelerate | 0.24+ | Multi-GPU / mixed precision training |
| NumPy | 1.24+ | Numerical operations |

---

## Design Principles

- **Parameter efficiency** — LoRA trains <1% of model parameters vs full fine-tuning
- **Grounded generation** — RAG retrieval reduces hallucination at inference time
- **Modularity** — Fine-tuning, retrieval, and evaluation are fully decoupled components
- **Reproducibility** — Seeded training, versioned configs, logged experiments
- **GPU compatibility** — CUDA-optimized throughout with CPU fallback

---

## Roadmap

- [x] LoRA fine-tuning pipeline
- [x] FAISS-based RAG retrieval
- [x] Base vs fine-tuned vs RAG comparison
- [ ] QLoRA (4-bit quantized fine-tuning for consumer GPUs)
- [ ] ROUGE / BLEU automated evaluation
- [ ] Hallucination detection (SelfCheckGPT integration)
- [ ] Experiment tracking (Weights & Biases / MLflow)
- [ ] Domain-specific larger training corpus
- [ ] REST inference API (FastAPI)
- [ ] Dockerize full pipeline
- [ ] CI/CD for regression testing on model updates

---

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

Built and maintained by **[Your Name]**  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username) · [Email](mailto:your@email.com)

---

> *Designed as a scalable foundation for domain-adapted LLM systems in production environments — where factual grounding, low hallucination rates, and modular extensibility are non-negotiable.*
