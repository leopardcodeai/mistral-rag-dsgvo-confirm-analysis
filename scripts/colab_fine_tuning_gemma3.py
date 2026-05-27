# Google Colab Fine-Tuning Script für Gemma 3 1B
# data: AWS S3 eu-central-1 (DSGVO)
# Training: Google Colab (kostenlos T4 GPU)
# model: google/gemma-3-1b-it (1B Parameter, offen)

import os
import json
import boto3
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import torch

# ==========================================
# 1. AWS KONFIGURATION (Keys direkt eintragen!)
# ==========================================
# HIER DEINE AWS KEYS EINTRAGEN (keine Secrets)
os.environ['AWS_ACCESS_KEY_ID'] = 'DEIN_AWS_KEY_HIER'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'DEIN_AWS_SECRET_HIER'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 2. DATEN VON AWS S3 LADEN (eu-central-1)
# ==========================================
print("Load dataset von AWS S3 (eu-central-1)...")
s3 = boto3.client('s3', region_name='eu-central-1')
try:
    response = s3.get_object(Bucket='rsi-test-data', Key='fine-tuning/kpi_training_dataset.jsonl')
    raw_data = response['Body'].read().decode('utf-8').splitlines()
    train_data = [json.loads(line) for line in raw_data if line.strip()]
    print(f"✓ {len(train_data)} datasets loaded.")
except Exception as e:
    print(f"Error beim Loadn von S3: {e}")
    raise

# ==========================================
# 3. LOAD MODEL (Gemma 3 1B - Klein und T4-kompatibel)
# ==========================================
model_name = "google/gemma-3-1b-it"  # 1B Parameter, offen
print(f"Load model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Efficient for T4
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. DATEN FORMATIEREN (Gemma 3 Chat Template)
# ==========================================
print("Formatting data for Gemma 3 Chat...")
def format_instruction(example):
    # Gemma 3 nutzt <start_of_turn>, <end_of_turn> Tags
    prompt = f"<start_of_turn>user\n{example['prompt']}<end_of_turn>\n<start_of_turn>model\n{example['completion']}<end_of_turn>"
    return {"text": prompt}

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_instruction)

# Tokenisierung
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
print("✓ data tokenisiert.")

# ==========================================
# 5. TRAINING KONFIGURATION (T4-optimiert)
# ==========================================
print("\nStart Fine-Tuning auf Colab T4 GPU...")
training_args = TrainingArguments(
    output_dir="./rsi-gemma-3b-finetuned",
    num_train_epochs=10,              # 10 epochs für 12 datasets
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    bf16=True,                       # T4 supports bf16
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"                  # Kein WandB in Colab
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator
)

# ==========================================
# 6. TRAINING STARTEN
# ==========================================
trainer.train()
print("\n✓ Training abgeschlossen!")

# ==========================================
# 7. UPLOAD MODEL TO AWS S3 (back to Frankfurt)
# ==========================================
print("\nSave model und lade to S3 (eu-central-1)...")
trainer.save_model("./rsi-gemma-3b-finetuned")
tokenizer.save_pretrained("./rsi-gemma-3b-finetuned")

# model packen
import tarfile
model_tar = "/tmp/rsi-gemma-3b.tar.gz"
with tarfile.open(model_tar, "w:gz") as tar:
    tar.add("./rsi-gemma-3b-finetuned", arcname=".")

print(f"✓ model gepackt: {model_tar}")

# Hochladen to S3 (eu-central-1)
print("Load zu s3://rsi-test-data/fine-tuned-models/rsi-gemma-3b/...")
s3.upload_file(
    model_tar,
    'rsi-test-data',
    'fine-tuned-models/rsi-gemma-3b/model.tar.gz'
)

print("\n" + "="*50)
print("ERFOLGREICH ABGESCHLOSSEN!")
print("="*50)
print("model stored in:")
print("s3://rsi-test-data/fine-tuned-models/rsi-gemma-3b/model.tar.gz")
print("\nDSGVO-Status: model befindet sich wieder in eu-central-1 (Frankfurt)")
