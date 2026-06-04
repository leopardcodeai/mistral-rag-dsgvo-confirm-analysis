# Mistral RAG vs Fine-Tuning — DSGVO/GDPR Compliance Analysis

> A case study comparing **Retrieval-Augmented Generation** and **Fine-Tuning** for GDPR-compliant AI in the mobility sector. Built with agentic coding workflows by [LeopardCode.AI](https://leopardcode.ai).

[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![DSGVO](https://img.shields.io/badge/DSGVO-Konform-green)](https://gdpr.eu)

---

## Abstract

This project compares two approaches for privacy-compliant AI model adaptation within the European legal framework: **Retrieval-Augmented Generation (RAG)** via AWS Bedrock using a European language model (Mistral Pixtral Large) and **Fine-Tuning** of an open-weights language model (Mistral 7B) using LoRA on a free T4 GPU in Google Colab.

A synthetic dataset of 20 ride-sharing customers with twelve KPI question-answer pairs serves as the case study. **Results show that RAG achieves 100% accuracy (223 out of 223 correctly identified rides), outperforming the fine-tuned approach, whose 4-bit quantized inference produced unintelligible output due to rounding errors.**

[📄 Read the full paper →](docs/PAPER.md)

---

## Results

| Approach | Accuracy | Duration | GDPR | Production Ready |
|----------|----------|----------|------|:----------------:|
| **Mistral RAG** (Bedrock, eu-central-1) | **100%** | ~3 sec | ✅ | ✅ Yes |
| **Mistral 7B Fine-Tuned** (4-bit, Colab T4) | **0%** | ~15 min | ✅ | ❌ No |
| distilgpt2 Fine-Tuned | 100% | ~15 min | ✅ | ⚠️ Toy only |
| Gemma 3 1B Fine-Tuned | not tested | — | ✅ | — |

**Key finding:** RAG delivers more accurate results under realistic resource constraints. Fine-Tuning with LoRA on a T4 GPU is technically feasible but suffers from hardware limitations during inference (4-bit quantization rounding errors).

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │      Data: eu-central-1 (Frankfurt)   │
                    │      AWS S3: rsi-test-data           │
                    └─────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ PATH 1: Mistral RAG    │       │ PATH 2: Mistral 7B FT   │
│ (Production, no GPU)   │       │ (Training + Deployment)  │
├─────────────────────────┤       ├─────────────────────────┤
│ A. Question from user   │       │ A. Question from user   │
│ B. Load data from S3   │       │ B. Install dependencies │
│ C. Call Bedrock API   │       │ C. HF Login with token  │
│ D. Mistral generates   │       │ D. Load data from S3   │
│ E. Return: 223 rides   │       │ E. Load Mistral 7B      │
│                        │       │ F. Fine-Tune on T4 GPU │
│ Ready: Python script   │       │ G. Save model locally   │
│ query_mistral_db.py    │       │ H. Upload to S3:        │
│                        │       │    rsi-7b-finetuned/    │
│ No training needed!    │       │ I. Deploy to SageMaker  │
│                        │       │    Endpoint in Frankfurt│
│                        │       │ J. Query endpoint:      │
│                        │       │    query_mistral7b_endpoint │
│                        │       │ K. Compare with RAG     │
└─────────────────────────┘       └─────────────────────────┘
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              RSI KPI ANALYZER - DSGVO Flowchart                    │
│              All data in eu-central-1 (Frankfurt)                  │
└─────────────────────────────────────────────────────────────────────┘

A [Customer Question]
         │
         ▼
┌──────────────────────────────────────────────┐
│  S3 Bucket (rsi-test-data)                  │
│    ├── customer_data.json (20 customers)     │
│    └── fine-tuning/kpi_training_dataset.jsonl│
└──────────────────────┬──────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌───────────────────┐   ┌───────────────────┐
│ PATH 1: Mistral   │   │ PATH 2: Fine-Tune │
│ RAG (Production)  │   │ (Colab Training)  │
├───────────────────┤   ├───────────────────┤
│ AWS Bedrock API   │   │ Google Colab T4   │
│ → 223 rides ✅    │   │ → LoRA adapters   │
└───────────────────┘   └────────┬──────────┘
                                 │
                                 ▼
                        ┌───────────────────┐
                        │ SageMaker Endpoint│
                        │ eu-central-1      │
                        └────────┬──────────┘
                                 │
                                 ▼
                        ┌──────────────────────────┐
                        │ COMPARE: Fine-Tuned vs.  │
                        │ Mistral RAG              │
                        └──────────────────────────┘
```

---

## DSGVO/GDPR Compliance Check

| Criterion | Mistral RAG (Path 1) | Fine-Tuning (Path 2) |
|-----------|:--------------------:|:--------------------:|
| Data in EU (Training) | ✅ Yes | ⚠️ Colab US → returned to EU |
| Data in EU (Inference) | ✅ Yes (Frankfurt) | ✅ Yes (Frankfurt) |
| EU-based model | ✅ Yes (Mistral, France) | ✅ Yes (Mistral 7B open-weights) |
| Production-ready | ✅ Yes | ⚠️ Needs GPU quota |
| AWS infrastructure | ✅ Art. 28 GDPR compliant | ✅ Art. 28 GDPR compliant |

**Result:** Both architectures meet GDPR requirements with proper configuration. RAG is immediately production-ready.

---

## Project Structure

```
aws/
├── README.md                        # This file
├── .env.example                     # AWS credentials template
│
├── data/                            # Datasets
│   ├── customer_data.json            # 20 synthetic ride-share customers
│   └── kpi_training_dataset.jsonl    # 12 Q&A pairs for fine-tuning
│
├── scripts/                         # All Python & Bash scripts
│   ├── create_fake_data.py           # Generate 20 ride-share customers
│   ├── upload_to_s3.py              # Upload data to S3
│   ├── query_mistral_db.py          # PATH 1: Mistral RAG ✅
│   ├── query_claude_rag.py          # Comparison: Claude RAG
│   ├── query_llama_rag.py           # Comparison: Llama RAG
│   ├── query_mistral7b_endpoint.py  # Mistral 7B Fine-Tuned Query
│   ├── compare_pipelines.py         # Compare both paths
│   ├── prepare_fine_tuning.py       # Upload dataset to S3 for FT
│   ├── colab_mistral7b_final.py     # Colab: Mistral 7B Fine-Tuning
│   ├── colab_fine_tuning_gemma3.py  # Colab: Gemma 3 1B
│   ├── colab_gemma3_full.py         # Colab: Gemma 3 Complete
│   ├── aws_mistral7b_endpoint.py    # Deploy Mistral 7B to SageMaker
│   ├── setup.sh                     # Automated setup
│   └── check_bedrock_custom.py      # Check Bedrock Custom Models
│
├── configs/                         # Configuration files
│   ├── config.json                  # Bedrock configuration
│   ├── bedrock_and_s3_policy.json   # IAM policy
│   ├── fine_tuning_config.json      # Bedrock customization config
│   └── ec2-trust-policy.json        # EC2 trust policy
│
├── models/                          # Fine-tuned model weights
├── docs/                            # Full documentation
│   ├── PAPER.md                     # Full academic paper 📄
│   ├── ARCHITEKTUR.md               # Detailed architecture
│   ├── DSGVO_ARCHITEKTUR.md         # GDPR status & limitations
│   └── COLAB_GUIDE.md              # Google Colab guide
│
└── tests/
```

---

## Quick Start

### 1. Mistral RAG (No training needed)
```bash
python3 scripts/query_mistral_db.py
# → "The sum of all rides is 223."
```

### 2. Fine-Tune Mistral 7B in Google Colab
```bash
# 1. Enable T4 GPU in Colab
# 2. Add AWS keys to colab_mistral7b_final.py
# 3. Run: !python colab_mistral7b_final.py
# Training: 5-10 min → model uploaded to S3
```

### 3. Deploy Fine-Tuned Model to SageMaker
```bash
python3 scripts/aws_mistral7b_endpoint.py
# Deployed in eu-central-1 (Frankfurt) ✅ GDPR
```

### 4. Compare Results
```bash
python3 scripts/compare_pipelines.py
# Shows comparison table
```

---

## Discussion

### When to use RAG
- Small to medium structured datasets
- When data changes frequently
- Zero GPU infrastructure required
- Deterministic computation based on source data
- **Hallucination resistance** — model receives facts as context

### When to use Fine-Tuning
- Stylistic adaptation (specific response formats)
- Large unstructured text corpora
- Latency-critical applications (no external retrieval step)

### Limitations of this study
- Small dataset: 20 customers, 12 Q&A pairs
- Synthetic data, not production
- T4 GPU (15 GB VRAM) forced 4-bit quantization
- Single run per configuration, no A/B testing

---

## Resources

- [Full Paper (PDF-style Markdown)](docs/PAPER.md)
- [Architecture Overview](docs/ARCHITEKTUR.md)
- [GDPR Compliance Details](docs/DSGVO_ARCHITEKTUR.md)
- [Google Colab Guide](docs/COLAB_GUIDE.md)
- [LeopardCode.AI — AI Engineering & Consulting](https://leopardcode.ai)

---

<p align="center">
  <sub>Built with agentic coding workflows by <a href="https://leopardcode.ai">LeopardCode.AI</a></sub><br>
  <sub>Dr. Alexander Brunker — AI Engineering &amp; Consulting</sub>
</p>
