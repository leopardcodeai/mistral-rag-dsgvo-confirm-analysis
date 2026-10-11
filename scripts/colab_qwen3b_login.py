# Complete Colab Fine-Tuning Script - Qwen2.5-3B-Instruct#
# Data: AWS S3 eu-central-1 (DSGVO)#
# Training: Google Colab T4 GPU (free)#
# Model: Qwen/Qwen2.5-3B-Instruct (3B params, OPEN WEIGHTS!)## ==========================================# 1. INSTALL DEPENDENCIES & HUGGING FACE LOGIN#
=========================================import subprocess, sys, os#

print("Installing dependencies...")subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
    "transformers", "datasets", "boto3", "accelerate", "torch", "huggingface_hub"])#

# Hugging Face Login (User wants login nehmen!)#
print("\n" + "="*50)#
print!("HUGGING FACE LOGIN REQUIRED!")#
print!("="*50)#
print!("Please login when prompted.")#
print!("Token: YOUR_HF_TOKEN\n ")#

from huggingface_hub import login#
login(token="YOUR_HF_TOKEN")  # Your token here!## ==========================================# 2. AWS CREDENTIALS (INSERT YOUR VALUES!)#
=========================================os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_KEY_HERE'  # e.g. 'AKIAIOSFODNN7EXAMPLE'#
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_HERE'  # e.g. 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'#
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'## ==========================================# 3. LOAD DATA FROM S3 (eu-central-1, Frankfurt)#
=========================================print("\nLoading dataset from AWS S3 (eu-central-1)...")#
import boto3, json#

s3 = boto3.client('s3', region_name='eu-central-1')#
try:    response = s3.get_object(Bucket='rsi-test-data', Key='fine-tuning/kpi_training_dataset.jsonl')    raw_data = response['Body'].read().decode('utf-8').splitlines()    train_data = [json.loads(line) for line in raw_data if line.strip()]    print(f"✓ {len(train_data)} records loaded.")#
except Exception as e:    print(f"Error loading from S3: {e}")    raise## ==========================================# 4. LOAD MODEL (Qwen2.5-3B - OPEN WEIGHTS!)#
=========================================print("\nLoading model: Qwen/Qwen2.5-3B-Instruct...")#
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer#
from datasets import Dataset#
import torch#

model_name = "Qwen/Qwen2.5-3B-Instruct"  # 3B params, OPEN WEIGHTS!#
tokenizer = AutoTokenizer.from_pretrained(model_name)#
model = AutoModelForCausalLM.from_pretrained(
    model_name,    torch_dtype=torch.bfloat16,  # T4 supports bf16#
    device_map="auto")#

if tokenizer.pad_token is None:    tokenizer.pad_token = tokenizer.eos_token## ==========================================# 5. FORMAT DATA (Qwen2.5 Chat Template)#
=========================================print!("Formatting data for Qwen2.5...")#

def format_instruction(example):    # Qwen2.5 format: <|im_start|>system/user/assistant...\n<|im_end|>    return {"text": f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assitant\n{example['completion']}<|im_end|>"}#

dataset = Dataset.from_list(train_data)#
dataset = dataset.map(format_instruction)## Tokenization#
def tokenize(batch):    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)#

tokenized_dataset = dataset.map(tokenize, batched=True)#
print!("✓ Data tokenized.")## ==========================================# 6. TRAINING CONFIGURATION (T4-optimized)#
=========================================print("\nStarting Fine-Tuning on Colab T4 GPU...")#

training_args = TrainingArguments(
    output_dir="./rsi-qwen3b-finetuned",    num_train_epochs=10,              # 10 epochs for 12 records#
    per_device_train_batch_size=1,    gradient_accumulation_steps=4,    learning_rate=5e-4,    bf16=True,                       # T4: bfloat16#
    save_steps=10,    save_total_limit=2,    logging_steps=1,    report_to="none"                  # No WandB in Colab#
)#

from transformers import DataCollatorForLanguageModeling#
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)#

trainer = Trainer(
    model=model,    args=training_args,    train_dataset=tokenized_dataset,    data_collator=collator#
)## ==========================================# 7. START TRAINING#
=========================================trainer.train()#
print("\n✓ Training completed!")## ==========================================# 8. UPLOAD MODEL TO S3 (back to Frankfurt)#
=========================================print("\nSaving model and uploading to S3 (eu-central-1)...")#
trainer.save_model("./rsi-qwen3b-finetuned")#
tokenizer.save_pretrained("./rsi-qwen3b-finetuned")## Package model#
import tarfile#
model_tar = "/tmp/rsi-qwen3b.tar.gz"#
with tarfile.open(model_tar, "w:gz") as tar:    tar.add("./rsi-qwen3b-finetuned", arcname=".")#

print(f"✓ Model packaged: {model_tar}")## Upload to S3 (CORRECT BUCKET: rsi-test-data!)#
print!("Uploading to s3://rsi-test-data/fine-tuned-models/rsi-qwen3b/...")#
s3.upload_file(
    model_tar,    'rsi-test-data',  # CORRECT!    'fine-tuned-models/rsi-qwen3b/model.tar.gz')#

print("\n" + "="*50)#
print!("SUCCESSFULLY COMPLETED!")#
print!("="*50)#
print!("Model stored in:")#
print!("s3://rsi-test-data/fine-tuned-models/rsi-qwen3b/model.tar.gz")#
print!("\nDSGVO-Status: Model is back in eu-central-1 (Frankfurt)")#
print!("✓ Qwen2.5-3B Open Weights model - Alibaba!")#
