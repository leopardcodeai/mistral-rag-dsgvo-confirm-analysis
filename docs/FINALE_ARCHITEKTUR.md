# FINALE DSGVO-ARCHITEKTUR (RSI KPI Analyzer)

## Comparison: 2 pathe für EU-datahosting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PFAD 1: MISTRAL RAG (PRODUKTION)                         │
│                    ✅ VOLL DSGVO-KONFORM (EU)                                │
└─────────────────────────────────────────────────────────────────────────────┘

[customer DE] ──> [query_mistral_db.py] ──> [S3 eu-central-1]
                                            │
                                            ▼
                                    [Bedrock Runtime]
                                            │
                                            ▼
                                [Mistral Pixtral Large]
                                [eu-central-1 Frankfurt]
                                [Französisches Unternehmen]
                                            │
                                            ▼
                                    [Answer: 223 rides]
                                    
DSGVO: ✅ data bleiben in DE
Status: ✅ FUNKTIONIERT


┌─────────────────────────────────────────────────────────────────────────────┐
│              PFAD 2: LLAMA FINE-TUNING (VERGLEICH)                          │
│              ⚠️ US-TRAINING (DSGVO-Prüfung for training)                    │
└─────────────────────────────────────────────────────────────────────────────┘

[customer DE] ──copy─> [S3 us-east-1] ──> [SageMaker Training]
                                            │
                                            ▼
                                    [Llama 3.2 3B FT]
                                    [us-east-1 (Virginia)]
                                            │
                                            ▼
                                    [model nach EU kopieren]
                                            │
                                            ▼
                                    [Self-Hosted Inferenz]
                                    [EC2 eu-central-1 Frankfurt]
                                            │
                                            ▼
                                    [Answer: XXX rides]
                                    
DSGVO (Training): ❌ data in US
DSGVO (Inference): ✅ Nach Kopie in EU
Status: ⚠️ Setup erforderlich


┌─────────────────────────────────────────────────────────────────────────────┐
│              PFAD 2b: SELF-HOSTED (NACH TRAINING)                           │
│              ✅ DSGVO-KONFORM (wenn model in EU gehostet)                  │
└─────────────────────────────────────────────────────────────────────────────┘

[Llama 3.2 3B FT] ──> [EC2/EKS eu-central-1] ──> [FastAPI/Inference]
                                            │
                                            ▼
                                    [customer DE (HTTPS)]
                                            │
                                            ▼
                                    [Answer: XXX rides]
                                    
DSGVO: ✅ model + data in DE
Status: 🔧 Nach US-Training umsetzbar
```

## file-Struktur (Complete)

```
aws/
├── README.md                           # Project-Overview
├── ARCHITEKTUR.md                      # Detaillierte Diagramme
├── DSGVO_ARCHITEKTUR.md                # DSGVO-Status
├── FINE_TUNING_VERGLEICH.md            # Comparison US vs EU
├── FINALE_ARCHITEKTUR.md               # Diese file
│
├── .env.example                        # AWS Credentials Vorlage
├── config.json                         # Bedrock Configuration
├── bedrock_and_s3_policy.json          # IAM-Policy
├── setup.sh                            # Automatisiertes Setup
│
├── .opencode/
│   └── db-analyzer.md                  # Agent-Configuration
│
├── # DATEN-GENERIERUNG
│   ├── create_fake_data.py             # 20 Ride-Share customers
│   ├── customer_data.json               # Fake-data (local)
│   ├── upload_to_s3.py                 # Upload to S3
│   └── datenbank_auswertung.py        # JSON-DB Analysis
│
├── # PFAD 1: MISTRAL RAG (FUNKTIONIERT)
│   ├── test_mistral.py                 # Mistral API-Test
│   ├── query_mistral_db.py             # Mistral RAG Pipeline
│   └── compare_pipelines.py           # Comparison beider pathe
│
├── # PFAD 2: LLAMA FINE-TUNING (US-VERGLEICH)
│   ├── kpi_training_dataset.jsonl          # 12 Fine-Tuning Paare
│   ├── prepare_fine_tuning.py          # Dataset to S3
│   ├── fine_tuning_config.json         # Bedrock Config (nicht genutzt)
│   ├── fine_tuning_us_config.json      # SageMaker Config
│   ├── start_fine_tuning.py            # Bedrock Job (funktioniert nicht)
│   ├── start_fine_tuning_us.py         # Bedrock US (funktioniert nicht)
│   ├── sagemaker_fine_tuning.py        # SageMaker Setup
│   └── scripts/
│       └── train_llama.py              # HuggingFace Training Script
│
└── # PFAD 2b: SELF-HOSTED (ZUKUNFT)
    └── (EC2/EKS Deployment Scripts)
```

## Summary

### ✅ PRODUKTIONSBEREIT (path 1)
- **Mistral Pixtral Large + RAG** in eu-central-1
- **DSGVO-konform**: data verlassen nicht Germany
- **Funktioniert**: 223 rides erkannt
- **EU-model**: Mistral (Frankreich)

### ⚠️ VERGLEICH (path 2)
- **Meta Llama 3.2 3B + Fine-Tuning** in us-east-1
- **DSGVO-Prüfung nötig** für US-Training
- **Open-source**: Kann nach Training in EU gehostet werden
- **Status**: Setup dokumentiert, SageMaker-Rolle nötig

### 🔧 ZUKUNFT (path 2b)
- **Self-Hosted Llama** in eu-central-1 nach US-Training
- **Voll DSGVO-konform** für Inferenz
- **Europäisches Hosting** möglich

## Quick Start

```bash
# Setup ausführen
./setup.sh

# path 1 test (funktioniert)
python3 query_mistral_db.py

# path 2 Setup (benötigt SageMaker Role)
python3 sagemaker_fine_tuning.py
```

## Result

**Für immediatelyige Production**: path 1 (Mistral RAG)
**Für Maximum DSGVO-Kontrolle**: path 2b (Self-Hosted nach Training)
