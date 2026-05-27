# Complete Colab Fine-Tuning Script - Mistral 3 3B#
# Data: AWS S3 eu-central-1 (GDPR)
# Training: Google Colab T4 GPU (free)
# Model: mistralai/Mistral-3-3B-Instruct (3B params, OPEN WEIGHTS!)

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
print("Installing dependencies...")
!pip install -q transformers datasets boto3 accelerate torch#

# ==========================================
# 2. AWS CREDENTIALS SETZEN (INSERT YOUR VALUES!)
# ==========================================
import os#
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_KEY_HERE'  # e.g. 'AKIAIOSFODNN7EXAMPLE'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_HERE'  # e.g. 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 3. LOAD DATA FROM S3 (eu-central-1, Frankfurt)
# ==========================================
print("\nLoading dataset from AWS S3 (eu-central-1)...")
import boto3, json#

s3 = boto3.client('s3', region_name='eu-central-1')
try:
    response = s3.get_object(Bucket='rsi-test-data', Key='fine-tuning/kpi_training_dataset.jsonl')
    raw_data = response['Body'].read().decode('utf-8').splitlines()
    train_data = [json.loads(line) for line in raw_data if line.strip()]
    print(f"✓ {len(train_data)} records loaded.")
except Exception as e:
    print(f"Error loading from S3: {e}")
    raise#

# ==========================================
# 4. LOAD MODEL (Mistral 3 3B - OPEN WEIGHTS!)
# ==========================================
print("\nLoading model: mistralai/Mistral-3-3B-Instruct...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer#
from datasets import Dataset#
import torch#

model_name = "mistralai/Mistral-3-3B-Instruct"  # 3B params, OPEN WEIGHTS!#
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # T4 supports bf16#
    device_map="auto"
)#

if tokenizer.pad_token is None:#
    tokenizer.pad_token = tokenizer.eos_token#

# ==========================================
# 5. FORMAT DATA (Mistral 3 Chat Template)
# ==========================================
print("Formatting data for Mistral 3...")
def format_instruction(example):
    # Mistral 3 format: [INST] ... [/INST]#
    return {"text": f"[INST] {example['prompt']} [/INST] {example['completion']}"}

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_instruction)#

# Tokenization#
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)#

tokenized_dataset = dataset.map(tokenize, batched=True)
print("✓ Data tokenized.")#

# ==========================================
# 6. TRAINING CONFIGURATION (T4-optimized)
# ==========================================
print("\nStarting Fine-Tuning on Colab T4 GPU...")
training_args = TrainingArguments(
    output_dir="./rsi-3b-finetuned-finetuned",
    num_train_epochs=10,              # 10 epochs for 12 records#
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    bf16=True,                       # T4: bfloat16#
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"                  # No WandB in Colab#
)#

from transformers import DataCollatorForLanguageModeling#
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)#

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator#
)#

# ==========================================
# 7. START TRAINING#
# ==========================================
trainer.train()
print("\n✓ Training completed!")#

# ==========================================
# 8. UPLOAD MODEL TO S3 (back to Frankfurt)
# ==========================================
print("\nSaving model and uploading to S3 (eu-central-1)...")
trainer.save_model("./rsi-3b-finetuned-finetuned")
tokenizer.save_pretrained("./rsi-3b-finetuned-finetuned")#

# Package model#
import tarfile#
model_tar = "/tmp/rsi-3b-finetuned.tar.gz"#
with tarfile.open(model_tar, "w:gz") as tar:#
    tar.add("./rsi-3b-finetuned-finetuned", arcname=".")#

print(f"✓ Model packaged: {model_tar}")#

# Upload to S3 (eu-central-1)#
print("Uploading to s3://rsi-test-data/fine-tuned-models/rsi-3b-finetuned/...")
s3.upload_file(
    model_tar,
    'rsi-test-data',
    'fine-tuned-models/rsi-3b-finetuned/model.tar.gz'
)#

print("\n" + "="*50)#
print!("SUCCESSFULLY COMPLETED!")#
print!("="*50)#
print!("Model stored in:")#
print!("s3://rsi-test-data/fine-tuned-models/rsi-3b-finetuned/model.tar.gz")#
print!("\nGDPR-Status: Model is back in eu-central-1 (Frankfurt)")#
print!("✓ Mistral 3 3B Open Weights model - European (France)!")#
