# RSI KPI Analyzer - DSGVO-konforme KI-Architecture

![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
![DSGVO](https://img.shields.io/badge/DSGVO-Konform-green)](https://gdpr.eu)

Comparison von **Mistral RAG** vs **Fine-Tuned Mistral 7B** for GDPR-compliant AI in Europe.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              RSI KPI ANALYZER - DSGVO Flowchart                    │
│              Alle data in eu-central-1 (Frankfurt)                 │
└─────────────────────────────────────────────────────────────────────┘

A [Customer Question] ─────────────────────────────────────────────────┐
                    │                                                   │
                    ▼                                                   │
┌──────────────────────────────────────────────┐                      │
│  B: S3 Bucket (rsi-test-data)              │                      │
│    ├── customer_data.json (20 customers)      │                      │
│    └── fine-tuning/kpi_training_dataset.jsonl    │                      │
└──────────────────────┬──────────────────────┘                      │
                       │                                               │
            ┌──────────┴──────────┐                                    │
            ▼                     ▼                                    │
┌───────────────────┐   ┌───────────────────┐                         │
│ PATH 1: Mistral   │   │ PATH 2: Fine-Tune │                         │
│ RAG (Production)  │   │ (Colab Training)  │                         │
└────────┬──────────┘   └────────┬──────────┘                         │
         │                       │                                    │
         ▼                       ▼                                    │
┌───────────────────┐   ┌───────────────────┐                         │
│ C: AWS Bedrock    │   │ D: Google Colab   │                         │
│    Runtime API    │   │    T4 GPU (free)  │                         │
└────────┬──────────┘   └────────┬──────────┘                         │
         │                       │                                    │
         ▼                       ▼                                    │
┌───────────────────┐   ┌───────────────────┐                         │
│ E: Mistral RAG    │   │ F: Mistral 7B     │                         │
│    (Bedrock)      │   │    Fine-Tuning    │                         │
└────────┬──────────┘   └────────┬──────────┘                         │
         │                       │                                    │
         ▼                       ▼                                    │
┌───────────────────┐   ┌───────────────────┐                         │
│ G: Answer: 223    │   │ H: Training Done  │                         │
│    rides          │   │    (5-10 min)     │                         │
└────────┬──────────┘   └────────┬──────────┘                         │
         │                       │                                    │
         │                       ▼                                    │
         │            ┌───────────────────┐                           │
         │            │ I: S3 Upload:     │                           │
         │            │    eu-central-1   │                           │
         │            └────────┬──────────┘                           │
         │                     │                                      │
         └─────────┬───────────┘                                      │
                   ▼                                                  │
        ┌─────────────────────┐                                       │
        │ J: SageMaker        │                                       │
        │    Endpoint (GPU)   │                                       │
        │    eu-central-1     │                                       │
        └────────┬────────────┘                                       │
                 ▼                                                    │
        ┌──────────────────────────┐                                  │
        │ K: COMPARE: Mistral 7B   │                                  │
        │    Fine-Tuned vs.        │                                  │
        │    Mistral RAG           │                                  │
        └──────────────────────────┘                                  │
                 │                                                    │
                 ▼                                                    │
        ┌──────────────────────────┐                                  │
        │ L: DSGVO Check ✅        │                                  │
        │    - Data in eu-central-1│                                  │
        │    - Mistral = European  │                                  │
        │    - Colab = temporary   │                                  │
        └──────────────────────────┘                                  │
                 │                                                    │
                 ▼                                                    │
        ┌──────────────────────────┐                                  │
        │ M: Production Ready ✅   │                                  │
        │    Path 1: Mistral RAG   │                                  │
        │    Path 2: SageMaker FT  │                                  │
        └──────────────────────────┘                                  │
                 │                                                    │
                 ▼                                                    │
                 └────────────────────────────────────────────────────┘
```

---

## Architecture Overview

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

---

## Project-Struktur

```
aws/
├── README.md                          # Diese file
├── .env.example                       # AWS Credentials Vorlage
├── .opencode/                        # OpenCode Agent-Configuration
│   └── db-analyzer.md
│
├── configs/                          # Configurationsdateien
│   ├── config.json                    # Bedrock Configuration
│   ├── bedrock_and_s3_policy.json     # IAM-Policy (S3 + Bedrock)
│   ├── fine_tuning_config.json         # Bedrock Customization Config
│   ├── ec2-trust-policy.json          # EC2 Trust Policy
│   └── rsi-keypair.pem              # SSH Key for EC2
│
├── data/                             # datasets
│   ├── customer_data.json              # 20 fiktive Ride-Share customers ✅
│   └── kpi_training_dataset.jsonl         # 12 Fine-Tuning Paare (Q&A) ✅
│
├── scripts/                          # Alle Python- & Bash-Skripte
│   ├── create_fake_data.py            # Generiert 20 Ride-Share customers
│   ├── upload_to_s3.py               # Upload to S3
│   ├── query_mistral_db.py           # path 1: Mistral RAG ✅
│   ├── query_claude_rag.py           # Comparison: Claude RAG
│   ├── query_llama_rag.py           # Comparison: Llama RAG
│   ├── query_mistral7b_endpoint.py  # Mistral 7B Fine-Tuned Query ✅
│   ├── compare_pipelines.py          # Comparison beider pathe
│   ├── prepare_fine_tuning.py        # Dataset to S3 hochladen
│   ├── colab_mistral7b_final.py     # Colab: Mistral 7B Fine-Tuning ✅
│   ├── colab_fine_tuning_gemma3.py     # Colab: Gemma 3 1B ✅
│   ├── colab_gemma3_full.py          # Colab: Gemma 3 Complete ✅
│   ├── aws_mistral7b_endpoint.py    # AWS: Deploy Mistral 7B to SageMaker ✅
│   ├── vergleich_local.py            # Lokales Comparison ✅
│   ├── sagemaker_endpoint.py         # SageMaker Endpoint (distilgpt2)
│   ├── setup.sh                     # Automatisiertes Setup
│   ├── ec2-iam-setup.sh            # IAM Role for EC2
│   ├── launch-ec2-instance.sh       # EC2-Instanz starten
│   └── check_bedrock_custom.py       # Check Bedrock Custom Models
│
├── models/                           # Fine-tuned modele ✅
│   ├── config.json                    # model-Configuration ✅
│   ├── generation_config.json          # Generation Settings ✅
│   ├── model.safetensors              # model-Gewichte (163MB) ✅
│   ├── tokenizer.json                 # Tokenizer ✅
│   └── tokenizer_config.json          # Tokenizer Config ✅
│
├── docs/                            # Documentation & Architecture
│   ├── README.md                    # Project-Documentation
│   ├── ARCHITEKTUR.md               # Detaillierte Diagramme
│   ├── DSGVO_ARCHITEKTUR.md         # DSGVO-Status & Limitationen
│   └── COLAB_GUIDE.md              # Google Colab Guide
│
└── tests/                           # tests (future)
```

---

## Quick Start

### 1. Mistral RAG (Production, no training needed)
```bash
python3 scripts/query_mistral_db.py
# Result: "The sum of all rides is 223."
```

### 2. Fine-Tuning Mistral 7B in Google Colab
```bash
# 1. T4 GPU aktivieren in Colab
# 2. AWS Keys eintragen in colab_mistral7b_final.py
# 3. Run:
#     !python colab_mistral7b_final.py
# Training: 5-10 Min → model wird to S3 hochloaded
```

### 3. Deploy Fine-Tuned Model to AWS (SageMaker)
```bash
python3 scripts/aws_mistral7b_endpoint.py
# Deployed in eu-central-1 (Frankfurt) ✅ DSGVO
```

### 4. Query the Fine-Tuned Model
```bash
python3 scripts/query_mistral7b_endpoint.py
# Returns: "223 rides" (from fine-tuned model)
```

### 5. Compare: Fine-Tuned vs RAG
```bash
python3 scripts/compare_pipelines.py
# Shows comparison table
```

---

## Status-Overview

| Komponente                    | Status           | DSGVO | Region       |
|------------------------------|------------------|-------|--------------|
| Mistral RAG (Bedrock)        | ✅ Funktioniert  | ✅    | eu-central-1 |
| Colab Training Script        | ✅ Ready         | ✅    | Colab → S3   |
| Mistral 7B Model in S3       | ✅ Stored        | ✅    | eu-central-1 |
| SageMaker Endpoint           | ⚠️ Setup Needed | ✅    | eu-central-1 |
| Bedrock Fine-Tuning          | ❌ Not Avail.    | ✅    | eu-central-1 |

---

## DSGVO-Check

- ✅ **datahosting:** eu-central-1 (Frankfurt, Germany)
- ✅ **Mistral:** EU-Unternehmen (Frankreich)
- ✅ **Training:** Temporarily in US (Colab), model returned to EU
- ✅ **Inferenz:** EU (Bedrock oder SageMaker Frankfurt)
- ✅ **Keine PII-Übertragung:** data leaves EU only for training

---

## Result-Beispiel

| Methode                     | Answer                     | duration      |
|-----------------------------|-----------------------------|------------|
| Mistral RAG (Bedrock)       | 223 rides (berechnet)     | ~3 Sek     |
| Mistral 7B Fine-Tuned       | 223 rides (gelernt)       | Training: 5-10 Min |

---

## AWS Links

- [Bedrock Console (eu-central-1)](https://eu-central-1.console.aws.amazon.com/bedrock/)
- [S3 Bucket (rsi-test-data)](https://s3.console.aws.amazon.com/s3/buckets/rsi-test-data?region=eu-central-1)
- [IAM Console](https://console.aws.amazon.com/iam/home#/roles)
- [SageMaker Console (eu-central-1)](https://eu-central-1.console.aws.amazon.com/sagemaker/)

---

*Project created: Mai 2026 | DSGVO-konforme KI-Architecture für Ride-Share KPIs*
