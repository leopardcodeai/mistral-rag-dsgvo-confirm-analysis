# RSI KPI Analyzer – Abschlussbericht

**Datum:** Mai 2026  
**Project:** DSGVO-konforme KI-Architecture für Ride-Sharing-KPIs  
**Ziel:** Comparison: Mistral RAG vs. Fine-Tuning (EU-datahosting)

---

## 1. PROJEKT-ÜBERSICHT

Fragestellung: "Wie viele rides haben alle customers bisher with the ride-sharing service?"

Zwei-path-Architecture:
- **path 1:** Mistral RAG (AWS Bedrock, eu-central-1)
- **path 2:** Fine-Tuning auf Colab T4 GPU, model in S3 eu-central-1

---

## 2. ARCHITEKTUR (FINAL)

```
┌──────────────────────────────────────────────────────────────┐
│                    RSI KPI ANALYZER                          │
│              data in eu-central-1 (Frankfurt)                │
└──────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │ S3: rsi-test-data   │
                    │ daten (Frankfurt) │
                    └────────┬─────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ PATH 1: Mistral RAG   │         │ PATH 2: Fine-Tuning   │
│ (Bedrock API)         │         │ (Colab T4 GPU)        │
├───────────────────────┤         ├───────────────────────┤
│ • Kein Training nötig │         │ • model: Mistral 7B  │
│ • Echtzeit-calculation │         │ • LoRA-Adapter        │
│ • 12 Q&A Paare (RAG)  │         │ • 10 epochs (12 Paare)│
│ • Answer: 223 rides│         │ • Upload to S3        │
│ • duration: ~3 seconds  │         │ • Training: ~10 Min   │
│ • Kosten: Bedrock API │         │ • Kosten: Colab Free  │
└───────────────────────┘         └───────────────────────┘
            │                                 │
            └──────────────┬──────────────────┘
                           ▼
              ┌────────────────────────┐
              │     VERGLEICH           │
              ├────────────────────────┤
              │ RAG:   223 rides ✅   │
              │ 7B FT: Unbrauchbar ❌   │
              │ (4-bit Merge verzerrt) │
              └────────────────────────┘
```

---

## 3. ALLE LERNINGS (Was wir ausprobiert haben)

### 3.1 AWS Bedrock Fine-Tuning
- **Versuch:** `mistral.pixtral-large-2502` via `CreateModelCustomizationJob`
- **Result:** ❌ Nicht supports in eu-central-1
- **Learning:** Bedrock supports kein Open-Source Fine-Tuning in EU

### 3.2 AWS SageMaker Fine-Tuning
- **Versuch:** `meta.llama3-2-3b`, `mistral.mistral-large`, GPU-Instanzen
- **Result:** ❌ Free Tier blockiert GPU-Instanzen (ResourceLimitExceeded)
- **Learning:** Ohne GPU-Quota kein SageMaker Training

### 3.3 AWS EC2 GPU-Instanz
- **Versuch:** `g5.2xlarge`, `p3.2xlarge`, `m5.xlarge` in eu-central-1
- **Result:** ❌ Free Tier blockiert alle GPU-Instanzen
- **Learning:** Nur `t3.micro` (1 vCPU, 1GB RAM) verfügbar – zu schwach

### 3.4 Google Colab Fine-Tuning (T4 GPU)
- **Versuch 1:** `distilgpt2` (82M) – ✅ Erfolgreich! (15 Min)
- **Versuch 2:** `gemma-3-1b-it` – ❌ Gated Model (HF Login nötig)
- **Versuch 3:** `mistralai/Mistral-3-3B-Instruct` – ❌ Repo existiert nicht
- **Versuch 4:** `mistralai/Mistral-7B-Instruct-v0.2` – ✅ Erfolgreich! (~10 Min)
- **Learning:** Colab T4 + LoRA + Open Weights Model = perfekte Kombi

### 3.5 model-Inferenz (Comparison)
- **Versuch 1:** Lokal laden (bfloat16) – ❌ 14GB > 12.7GB RAM → Session Crash
- **Versuch 2:** 8-bit Quantisierung – ❌ BitsAndBytes/SCB Error
- **Versuch 3:** 4-bit Quantisierung (NF4) – ✅ Lädt (~3.5GB), aber Merge verzerrt Output
- **Learning:** LoRA + 4-bit Merge = Qualitätsverlust. RAG ist für KPIs besser.

---

## 4. TECHNISCHE DETAILS

### 4.1 dataset
- **20 fiktive Ride-Share customers** mit: `kunden_id`, `adresse`, `marketing_permission`, `rides_total`, `lieblings_fahrzeugtyp`
- **12 Fine-Tuning Paare** (Q&A) für Instruction-Tuning
- **Format:** JSONL (`kpi_training_dataset.jsonl`)

### 4.2 model-Training (Colab)
```python
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # 7B, Open Weights
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
# LoRA Training: 10 epochs, batch_size=1, lr=5e-4
# Upload: s3://rsi-test-data/fine-tuned-models/rsi-7b-finetuned/
```

### 4.3 RAG-Inferenz (AWS Bedrock)
```python
# Mistral Pixtral Large via Inference Profile
modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0'
# Answer: "223 rides" (Echtzeit aus S3 customer_data.json)
```

---

## 5. ENDERGEBNIS

| Kriterium | Mistral RAG (path 1) | Mistral 7B Fine-Tuned (path 2) |
|-----------|---------------------|-------------------------------|
| **Genauigkeit** | ✅ 223 rides (präzise) | ❌ Unbrauchbar (4-bit Merge) |
| **duration** | ~3 seconds | ~10 Min Training + 5 Min Inferenz |
| **DSGVO** | ✅ eu-central-1 | ✅ eu-central-1 (nach Training) |
| **Kosten** | Bedrock API (Pay-per-Use) | Colab T4 (kostenlos) |
| **Wartung** | Keine (API) | model-Updates nötig |
| **Empfehlung** | ✅ **PRODUKTION** | ⚠️ Nur mit voller GPU |

---

## 6. PROJEKT-STRUKTUR

```
aws/
├── README.md                          # Flowchart & Doku
├── configs/                           # AWS/IAM Configuration
│   ├── bedrock_and_s3_policy.json     # IAM-Policy
│   └── rsi-keypair.pem              # SSH-Key
├── data/                              # datasets
│   ├── customer_data.json              # 20 Ride-Share customers
│   └── kpi_training_dataset.jsonl         # 12 Q&A Paare
├── scripts/                           # Alle Skripte
│   ├── query_mistral_db.py           # path 1: Mistral RAG ✅
│   ├── colab_mistral7b_vs_rag.py     # path 2: Colab FT+Comparison
│   ├── colab_fine_tuning_gemma3.py   # Training: verschiedene modele
│   ├── compare_pipelines.py          # Pipeline-Comparison
│   └── setup.sh                     # Automatisiertes Setup
├── models/                            # Fine-tuned modele (local)
│   └── model.safetensors (163MB)     # rsi-distilgpt2
├── docs/                              # Documentation
│   ├── COLAB_GUIDE.md                # Colab Guide
│   └── ABSCHLUSSBERICHT.md           # Diese file
└── s3://rsi-test-data/             # AWS S3 (Production)
    ├── customer_data.json
    ├── fine-tuning/kpi_training_dataset.jsonl
    └── fine-tuned-models/
        └── rsi-7b-finetuned/model.tar.gz  # 1.08 GB
```

---

## 7. DSGVO-COMPLIANCE

| Kriterium | Status |
|-----------|--------|
| datahosting | ✅ eu-central-1 (Frankfurt, Germany) |
| model-Inferenz (RAG) | ✅ Bedrock eu-central-1 |
| Training (Colab) | ⚠️ Temporarily US, model returned to EU |
| Fine-Tuned Model Storage | ✅ S3 eu-central-1 |
| Personenbezogene data | ✅ Keine PII in Trainingsdaten |

---

## 8. EMPFEHLUNGEN

### Für immediatelyige Production:
```bash
python3 scripts/query_mistral_db.py
# → "The sum of all rides is 223."
```
- ✅ DSGVO-konform (data + Inferenz in Frankfurt)
- ✅ Keine GPU nötig
- ✅ ~3 seconds Answerzeit

### Für Fine-Tuning (Zukunft):
- Mit **voller GPU** (A100/H100) to train, nicht LoRA+4-bit
- Oder **größeren dataset** (100+ Paare) für bessere Resultse
- Oder **RAG + Few-Shot** kombinieren

---

*Report created: Mai 2026 | RSI KPI Analyzer – DSGVO-Architecture*
