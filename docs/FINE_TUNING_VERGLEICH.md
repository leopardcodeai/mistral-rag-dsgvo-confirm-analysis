# Fine-Tuning Comparison: EU vs US (DSGVO-Analyse)

## Architecture-Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PFAD 1: MISTRAL RAG (EU - DSGVO-konform)                │
└─────────────────────────────────────────────────────────────────────────────┘
Quelle: S3 eu-central-1 (Frankfurt)
model: mistral.pixtral-large-2502 (Frankreich)
Status: ✅ FUNKTIONIERT
DSGVO: ✅ data verlassen nicht die EU

     [S3 Frankfurt] ──> [Mistral Bedrock] ──> [Answer: 223 rides]


┌─────────────────────────────────────────────────────────────────────────────┐
│              PFAD 2: LLAMA FINE-TUNING (US - Comparison)                    │
└─────────────────────────────────────────────────────────────────────────────┘
Quelle: S3 us-east-1 (Virginia) ← data aus EU kopiert
model: meta-llama/Llama-3.2-3B (open-source)
Training: SageMaker us-east-1
Status: ⚠️ Setup erforderlich
DSGVO: ❌ data verlassen EU (for training)

     [S3 Frankfurt] ─copy─> [S3 us-east-1] ──> [SageMaker Training]
                                                     │
                                             [Fine-tuned Llama]
                                                     │
                                              (kann in EU gehostet werden)


┌─────────────────────────────────────────────────────────────────────────────┐
│              PFAD 2b: SELF-HOSTED IN EU (Nach Training)                    │
└─────────────────────────────────────────────────────────────────────────────┘
Quelle: S3 eu-central-1 (Frankfurt)
model: Llama-3.2-3B (fine-tuned in US, dann nach EU kopiert)
Hosting: EC2/EKS in eu-central-1
Status: 🔧 Möglich nach US-Training
DSGVO: ✅ Nach Training in EU gehostet (data bleiben in EU)

     [Trained Model] ──> [EC2 eu-central-1] ──> [Inference]
```

## Aktueller Status

| Kriterium              | path 1: Mistral RAG | path 2: Llama FT (US) | path 2b: Self-Hosted |
|------------------------|----------------------|-----------------------|---------------------|
| Region (Training)      | eu-central-1         | us-east-1             | us-east-1 → eu-central-1 |
| Region (Inference)     | eu-central-1         | us-east-1             | eu-central-1 ✅      |
| Open-Source            | Nein                 | Ja ✅                 | Ja ✅               |
| Fine-Tuning möglich    | Nicht nötig (RAG)    | Ja (SageMaker)        | Ja (nach Training)   |
| DSGVO (Training)       | ✅ EU bleibt         | ❌ US (data raus)    | ❌ US (nur Training) |
| DSGVO (Inference)      | ✅ EU                | ❌ US                 | ✅ EU nach Deployment|
| data in EU            | ✅ Ja                | ❌ Nein (Training)    | ✅ Ja (nach Kopie)   |
| Funktionstüchtig       | ✅ Ja (223 rides)  | ⚠️ Setup nötig        | 🔧 Nach Training    |

## Empfehlung

**Für volle DSGVO-Konformität:**
1. ✅ **Mistral RAG** (path 1) - produktionsready, data bleiben in DE
2. 🔧 **Llama Self-Hosted** (path 2b) - nach US-Training, model in EU hosten

**Warum path 2b DSGVO-konform sein kann:**
- model-Gewichte enthalten keine PII (nur Muster)
- Nach Training: model in EU verschieben
- Inferenz läuft komplett in eu-central-1
- Keine EU-Bürgerdaten verlassen die EU für Inferenz

## fileen für path 2 (US Fine-Tuning Comparison)

```
sagemaker_fine_tuning.py    # SageMaker Job Configuration
scripts/
  └── train_llama.py        # HuggingFace Training Script
```

## Nächste Steps für path 2b

1. SageMaker Role erstellen (IAM)
2. Training Job in us-east-1 starten
3. model nach Training to S3 eu-central-1 kopieren
4. EC2/EKS in Frankfurt für Self-Hosting nutzen
5. Inferenz in EU (DSGVO-konform)
