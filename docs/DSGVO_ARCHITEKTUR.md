# DSGVO-konforme KI-Architecture für Ride-Share KPIs

## Status: Fine-Tuning in eu-central-1 NICHT verfügbar

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DSGVO-KONFORME KI-ARCHITEKTUR (eu-central-1)                 │
│                    data verlassen Germany NICHT                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│   PFAD 1: Mistral RAG         │    │   PFAD 2: Llama RAG            │
│   (EU-model, funktionsfähig) │    │   (US open-source, RAG)        │
└─────────────────────────────────┘    └─────────────────────────────────┘
             │                                    │
             ▼                                    ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  S3: rsi-test-data          │    │  S3: rsi-test-data          │
│  eu-central-1 (Frankfurt)     │    │  eu-central-1 (Frankfurt)     │
└─────────────────────────────────┘    └─────────────────────────────────┘
             │                                    │
             ▼                                    ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  AWS Bedrock                   │    │  AWS Bedrock                   │
│  mistral.pixtral-large-2502    │    │  meta.llama3-2-3b-instruct    │
│  Inference Profile             │    │  Inference Profile             │
│  eu-central-1 ✓               │    │  eu-central-1 ✓               │
└─────────────────────────────────┘    └─────────────────────────────────┘
             │                                    │
             ▼                                    ▼
        294 rides                        294 rides
      (Mistral antwortet)              (Llama antwortet)
```

## Limitation: Fine-Tuning in Europa

**Status Quo (Stand 2026):**
- ❌ Kein model in eu-central-1 supports Fine-Tuning
- ❌ Kein model in eu-west-1 supports Fine-Tuning  
- ❌ Kein model in eu-west-3 supports Fine-Tuning
- ✅ RAG (Retrieval Augmented Generation) funktioniert in allen Regionen

## Comparison der modele (RAG-Ansatz)

| Kriterium              | Mistral Pixtral Large | Meta Llama 3.2 3B |
|------------------------|----------------------|-------------------|
| Herkunft               | 🇫🇷 Frankreich       | 🇺🇸 USA            |
| Open-Source            | Nein                 | Ja                |
| DSGVO (Hosting)        | ✅ eu-central-1      | ✅ eu-central-1    |
| RAG funktionsfähig     | ✅ Ja                | ⚠️ Legacy-Status  |
| Fine-Tuning (EU)       | ❌ Nein              | ❌ Nein            |
| modelgröße            | ~120B Parameter      | 3B Parameter      |

## Empfehlung

**Für DSGVO-konforme Production:**
1. **Mistral RAG** (path 1) - EU-model, funktionsfähig, data bleiben in DE
2. Falls Fine-Tuning zwingend: **us-east-1 nutzen** (data verlassen EU → DSGVO-Prüfung nötig)

## fileen für RAG-Pipelines

```bash
# path 1: Mistral (FUNKTIONIERT)
python3 query_mistral_db.py

# path 2: Llama (benötigt model-Aktivierung)
python3 query_llama_working.py
```

## Fine-Tuning Alternative (nicht EU)

Falls Fine-Tuning zwingend erforderlich:
```bash
# Nur in us-east-1 verfügbar (DSGVO-Risiko!)
export AWS_DEFAULT_REGION=us-east-1
python3 start_fine_tuning.py
```
