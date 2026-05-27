# Fine-Tuning in Google Colab (Kostenlos) mit AWS S3-Anbindung

## Overview
Wir nutzen die **kostenlose T4 GPU in Google Colab**, um ein kleines model (Gemma 3 270M) zu to train.
Die data kommen von **AWS S3 (eu-central-1, Frankfurt)** und das fertige model wird dorthin zurückloaded (DSGVO-konform).

## Schritt 1: Google Colab öffnen
1. Gehe zu [Google Colab](https://colab.research.google.com/)
2. Erstelle ein **Neues Notebook**

## Schritt 2: T4 GPU aktivieren
1. Klicke oben rechts auf das **Zahnrad-Symbol** (Settings)
2. Unter "Hardware-Beschleuniger" wähle **T4 GPU**
3. Save

## Schritt 3: AWS Secrets hinterlegen
1. Klicke links auf das **Schlüssel-Symbol** (Geheimnisse/Secrets)
2. Klicke auf **"Neuen geheimen Schlüssel hinzufügen"**
3. Füge folgende zwei Schlüssel hinzu:
   - **Name:** `AWS_ACCESS_KEY_ID` | **Wert:** Dein AWS Key
   - **Name:** `AWS_SECRET_ACCESS_KEY` | **Wert:** Dein AWS Secret
4. Aktiviere den Schalter **"Zugriff auf das Notebook gewähren"** für beide.

## Schritt 4: Skript in Colab laden und ausführen
Kopiere diesen Code in die erste Zelle deines Colab Notebooks:

```python
# In Colab Zelle einfügen und ausführen
!wget -q https://rsi-test-data.s3.eu-central-1.amazonaws.com/scripts/colab_fine_tuning.py -O colab_fine_tuning.py
!pip install -q transformers datasets boto3 accelerate

# Skript ausführen
!python colab_fine_tuning.py
```

*Note: Das Skript lädt sich selbst von S3, installiert die Abhängigkeiten und startet das Training.*

## Schritt 5: Was passiert im Hintergrund?
1. **Download:** `colab_fine_tuning.py` lädt `kpi_training_dataset.jsonl` von S3 (eu-central-1) herunter.
2. **model:** Lädt `google/gemma-3-270m-it` (sehr klein, passt in T4 VRAM).
3. **Training:** Trainiert 5 epochs auf deinen 12 Ride-Share-KPI-datasetsn.
4. **Upload:** Packt das model und lädt es zu `s3://rsi-test-data/fine-tuned-models/rsi-gemma-ft/model.tar.gz`.

## Schritt 6: Result prüfen (In AWS Console oder CLI)
```bash
# Prüfe, ob das model in Frankfurt (eu-central-1) gelandet ist
aws s3 ls s3://rsi-test-data/fine-tuned-models/rsi-gemma-ft/ --region eu-central-1
```

## DSGVO-Vorteil
- **Training:** In den USA (Google Server) - *Temporarily, keine dauerhafte Speicherung der PII.*
- **dataset:** Wird bei Bedarf loaded, nach Training vom Colab-Speicher gelöscht.
- **model:** Wird **back to Frankfurt (eu-central-1)** loaded.
- **Inferenz:** Findet danach wieder in Europa statt (via Mistral RAG oder Self-Hosted Llama).

## Errorbehebung
- **"No space left on device":** In Colab: `!rm -rf /content/sample_data` ausführen.
- **AWS Error:** Prüfe, ob die Secrets in Colab korrekt hinterlegt sind.
- **CUDA Out of Memory:** Das Skript nutzt `bfloat16` und kleine Batch-Größen. Bei Problemen `per_device_train_batch_size=1` lassen.
