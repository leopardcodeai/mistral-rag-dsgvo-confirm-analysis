# RSI KPI Analyzer - DSGVO-konforme KI-Architecture

## Overview
Comparison zweier Ansätze für DSGVO-konforme KI-modelanpassung mit europäischen datahosting.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RSI KPI ANALYZER (DSGVO)                         │
│                     data verbleiben in eu-central-1                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐              ┌─────────────────────────────┐
│   PFAD 1: Mistral RAG      │              │  PFAD 2: Meta Llama FT     │
│   (Europäisches model)    │              │  (US-model, EU-Hosting)    │
└─────────────────────────────┘              └─────────────────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────────┐              ┌─────────────────────────────┐
│  S3: rsi-test-data       │              │  S3: rsi-test-data       │
│  ├── customer_data.json     │              │  ├── customer_data.json     │
│  └── fine-tuning/          │              │  └── kpi_training_dataset.jsonl│
│      └── (nicht genutzt)   │              │                           │
└─────────────────────────────┘              └─────────────────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────────┐              ┌─────────────────────────────┐
│  RAG (Retrieval)           │              │  Fine-Tuning Job            │
│  - Lädt JSON aus S3        │              │  - Basis: meta.llama3-2-3b │
│  - Übergibt Kontext        │              │  - Training: 12 KPI-Paare   │
│    an Mistral              │              │  - Region: eu-central-1      │
└─────────────────────────────┘              └─────────────────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────────┐              ┌─────────────────────────────┐
│  AWS Bedrock               │              │  AWS Bedrock                │
│  mistral.pixtral-large     │              │  meta.llama3-2-3b-instruct  │
│  eu-central-1 (Frankfurt)  │              │  eu-central-1 (Frankfurt)   │
│  Inference Profile         │              │  Custom Model nach Training  │
└─────────────────────────────┘              └─────────────────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Answer: "294 rides gesamt"                        │
└─────────────────────────────────────────────────────────────────────────┘

DSGVO-Check:
✅ Beide pathe: data in eu-central-1 (Germany)
✅ path 1: Mistral (französisches Unternehmen) + RAG
✅ path 2: Meta Llama (open-source) + Fine-Tuning in EU
```

## Detaillierte Architecture pro path

### path 1: Mistral + RAG (Instruction-Tuning via Retrieval)

```
[customer] --HTTPS--> [query_mistral_db.py]
                            │
                            ▼
                   [S3: customer_data.json]
                            │
                            ▼
                   [Bedrock Runtime API]
                            │
                            ▼
                   [Mistral Pixtral Large]
                   [eu-central-1 Frankfurt]
                            │
                            ▼
                   [Answer mit KPI-calculation]
```

**Vorteile:**
- Kein Fine-Tuning nötig (schneller Start)
- data bleiben in S3 (kein model-Training mit PII)
- Mistral = europäisches Unternehmen (Frankreich)

### path 2: Meta Llama + Fine-Tuning (Custom Model)

```
[customer] --HTTPS--> [query_llama_ft.py]
                            │
                            ▼
                   [S3: kpi_training_dataset.jsonl]
                            │
                            ▼
                   [Bedrock CreateModelCustomizationJob]
                            │
                            ▼
                   [Meta Llama 3.2 3B Instruct]
                   [Training in eu-central-1]
                            │
                            ▼
                   [Custom Model: rsi-kpi-llama]
                   [Inferenz in eu-central-1]
                            │
                            ▼
                   [Answer mit trainierten KPIs]
```

**Vorteile:**
- model lernt Ride-Share-spezifische Answermuster
- Open-source model (keine US-Provider-Abhängigkeit)
- Fine-Tuning in EU (DSGVO-konform)

## file-Struktur

```
aws/
├── .env.example                    # AWS Credentials Vorlage
├── config.json                    # Bedrock Configuration
├── .opencode/
│   └── db-analyzer.md             # Agent-Configuration
├── create_fake_data.py            # Generiert 20 Ride-Share customers
├── customer_data.json              # Fake-customer data (local)
├── upload_to_s3.py                # Upload to S3
├── bedrock_and_s3_policy.json     # IAM-Policy (S3 + Bedrock)
├── datenbank_auswertung.py        # JSON-DB Analysis
├── test_mistral.py                # Mistral API-Test
│
├── query_mistral_db.py            # path 1: Mistral + RAG
├── query_claude_rag.py            # (Optional) Claude + RAG
├── kpi_training_dataset.jsonl         # Fine-Tuning data (12 Paare)
├── prepare_fine_tuning.py         # Upload Dataset to S3
├── fine_tuning_config.json        # Bedrock Customization Job Config
├── start_fine_tuning.py           # Startt Fine-Tuning Job
└── compare_pipelines.py           # Comparison beider Pipelines
```

## DSGVO-Konformität

| Kriterium                    | Mistral RAG | Meta Llama FT |
|------------------------------|-------------|---------------|
| dataregion                   | eu-central-1| eu-central-1  |
| PII in Training-data        | Nein        | Ja (anonymisiert) |
| EU-Anbieter                   | Ja (Frankreich) | Nein (US, aber open-source) |
| datakontrolle                | Voll        | Voll          |
| Fine-Tuning in EU             | Nicht nötig | Ja            |

## Start der Pipelines

### path 1: Mistral RAG
```bash
python3 query_mistral_db.py
```

### path 2: Meta Llama Fine-Tuning (sobald verfügbar)
```bash
# Dataset hochladen
python3 prepare_fine_tuning.py

# Fine-Tuning Job starten (benötigt Unterstützung in eu-central-1)
python3 start_fine_tuning.py

# Abfrage mit custom model
python3 query_llama_ft.py
```

### Comparison
```bash
python3 compare_pipelines.py
```
