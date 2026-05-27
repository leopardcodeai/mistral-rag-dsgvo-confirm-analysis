# Complete Colab Fine-Tuning Script - Gemma 3 1B
# data: AWS S3 eu-central-1 (DSGVO)
# Training: Google Colab T4 GPU (kostenlos)
# model: google/gemma-3-1b-it (1B Parameter, offen)

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
print("Installing dependencies...")
!pip install -q transformers datasets boto3 accelerate torch

# ==========================================
# 2. AWS CREDENTIALS SETZEN (HIER EINTRAGEN!)
# ==========================================
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'DEIN_AWS_KEY_HIER'  # z.B. 'AKIAIOSFODNN7EXAMPLE'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'DEIN_AWS_SECRET_HIER'  # z.B. 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 3. DATEN VON S3 LADEN (eu-central-1, Frankfurt)
# ==========================================
print("\nLoad dataset von AWS S3 (eu-central-1)...")
import boto3
import json

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
# 4. LOAD MODEL (Gemma 3 1B - NO LOGIN NEEDED!)
# ==========================================
print("\nLoad model: google/gemma-3-1b-it...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

model_name = "google/gemma-3-1b-it"  # 1B Parameter, offen!
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # T4 supports bf16
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 5. DATEN FORMATIEREN (Gemma 3 Format)
# ==========================================
print("Formatting data for Gemma 3...")
def format_instruction(example):
    # Gemma 3 Format: <start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n...\n<end_of_turn>
    return {"text": f"<start_of_turn>user\n{example['prompt']}<end_of_turn>\n<start_of_turn>model\n{example['completion']}<end_of_turn>"}

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_instruction)

# Tokenisierung
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
print("✓ data tokenisiert.")

# ==========================================
# 6. TRAINING KONFIGURATION (T4-optimiert)
# ==========================================
print("\nStart Fine-Tuning auf Colab T4 GPU...")
training_args = TrainingArguments(
    output_dir="./rsi-gemma-3b-finetuned",
    num_train_epochs=10,              # 10 epochs für 12 datasets
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    bf16=True,                       # T4: bfloat16
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"                  # Kein WandB in Colab
)

from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator
)

# ==========================================
# 7. TRAINING STARTEN
# ==========================================
trainer.train()
print("\n✓ Training abgeschlossen!")

# ==========================================
# 8. UPLOAD MODEL TO AWS S3 (back to Frankfurt)
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
