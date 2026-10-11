# Complete Colab Fine-Tuning Script - Mistral 7B Final
# Data: AWS S3 eu-central-1 (GDPR)
# Training: Google Colab T4 GPU (free)
# Model: mistralai/Mistral-7B-v0.3 (7B params, OPEN WEIGHTS!)

# ==========================================
# 1. INSTALL DEPENDENCIES & HUGGING FACE LOGIN
# ==========================================
import subprocess, sys, os

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
    "transformers", "datasets", "boto3", "accelerate", "torch", "huggingface_hub", "bitsandbytes", "peft", "scipy"])

# Hugging Face Login
print("\n" + "="*50)
print("HUGGING FACE LOGIN")
print("="*50)
subprocess.run([sys.executable, "-m", "huggingface_hub", "login",
              "--token", "YOUR_HF_TOKEN"])

# ==========================================
# 2. AWS CREDENTIALS (INSERT YOUR VALUES!)
# ==========================================
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 3. LOAD DATA FROM S3 (eu-central-1, Frankfurt)
# ==========================================
print("\nLoading dataset from AWS S3 (eu-central-1)...")
import boto3, json

s3 = boto3.client('s3', region_name='eu-central-1')
try:
    response = s3.get_object(Bucket='rsi-test-data', Key='fine-tuning/kpi_training_dataset.jsonl')
    raw_data = response['Body'].read().decode('utf-8').splitlines()
    train_data = [json.loads(line) for line in raw_data if line.strip()]
    print(f"✓ {len(train_data)} records loaded.")
except Exception as e:
    print(f"Error loading from S3: {e}")
    raise

# ==========================================
# 4. LOAD MODEL (Mistral 7B - OPEN WEIGHTS!)
# ==========================================
print("\nLoading model: mistralai/Mistral-7B-v0.3...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch

# 4-bit quantization for T4 GPU (16GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare model for k-bit training with LoRA
print("Preparing model with LoRA adapters...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA configuration for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                    # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 5. FORMAT DATA (Mistral Chat Template)
# ==========================================
print("Formatting data for Mistral...")
def format_instruction(example):
    return {"text": f"[INST] {example['prompt']} [/INST] {example['completion']}"}

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_instruction)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
print("✓ Data tokenized.")

# ==========================================
# 6. TRAINING CONFIGURATION (T4-optimized for 7B LoRA)
# ==========================================
print("\nStarting Fine-Tuning on Colab T4 GPU...")
training_args = TrainingArguments(
    output_dir="./rsi-7b-finetuned-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,         # Slightly lower for LoRA
    bf16=True,
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    report_to="none",
    gradient_checkpointing=True  # Save memory
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
# 7. START TRAINING
# ==========================================
trainer.train()
print("\n✓ Training completed!")

# ==========================================
# 8. SAVE & UPLOAD MODEL TO S3 (back to Frankfurt)
# ==========================================
print("\nSaving LoRA adapters and uploading to S3 (eu-central-1)...")
# Save only the LoRA adapters (not the full model)
trainer.model.save_pretrained("./rsi-7b-finetuned-finetuned")
tokenizer.save_pretrained("./rsi-7b-finetuned-finetuned")

import tarfile 
model_tar = "/tmp/rsi-7b-finetuned.tar.gz"
with tarfile.open(model_tar, "w:gz") as tar:
    tar.add("./rsi-7b-finetuned-finetuned", arcname=".")

print(f"✓ Model packaged: {model_tar}")

print("Uploading to s3://rsi-test-data/fine-tuned-models/rsi-7b-finetuned/...")
s3.upload_file(
    model_tar,
    'rsi-test-data',
    'fine-tuned-models/rsi-7b-finetuned/model.tar.gz'
)

print("\n" + "="*50)
print("SUCCESSFULLY COMPLETED!")
print("="*50)
print("Model stored in:")
print("s3://rsi-test-data/fine-tuned-models/rsi-7b-finetuned/model.tar.gz")
print("\nGDPR-Status: Model is back in eu-central-1 (Frankfurt)")
print("✓ Mistral 7B Open Weights model - European (France)!")
