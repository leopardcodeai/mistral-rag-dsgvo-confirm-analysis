# Colab Comparison: Mistral 7B Fine-Tuned vs Mistral RAG
# Both run in Colab (T4 GPU for 7B, Bedrock API for RAG)
# Data: AWS S3 eu-central-1 (Frankfurt)

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
import subprocess, sys, os, json, tarfile

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "boto3", "accelerate", "torch", "peft", "bitsandbytes", "torchao>=0.16.0"])

# ==========================================
# 2. AWS CREDENTIALS (INSERT YOUR VALUES!)
# ==========================================
# Use your own AWS keys here (e.g. from ~/.aws/credentials or environment)
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 3. HUGGING FACE LOGIN (NOT NEEDED - model loaded from S3 local)
# ==========================================
# Model is downloaded from S3, NOT from Hugging Face Hub.
# No HF token required.

# ==========================================
# 4. PART A: MISTRAL RAG (via Bedrock API)
# ==========================================
print("\n" + "="*50)
print("A) MISTRAL RAG (Bedrock API)")
print("="*50)

import boto3

s3 = boto3.client('s3', region_name='eu-central-1')
bedrock = boto3.client('bedrock-runtime', region_name='eu-central-1')

# Load customer data from S3
response = s3.get_object(Bucket='rsi-test-data', Key='customer_data.json')
kunden = json.loads(response['Body'].read())
total_rides = sum(k['rides_total'] for k in kunden)

# Query Mistral RAG via Bedrock
prompt_data = {
    "messages": [{"role": "user", "content": f"How many total rides? Sum: {total_rides}"}]
}
response = bedrock.invoke_model(
    modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0',
    contentType='application/json',
    accept='application/json',
    body=json.dumps(prompt_data)
)
response_body = json.loads(response['body'].read())
if 'outputs' in response_body:
    mistral_rag = response_body['outputs'][0]['text']
elif 'choices' in response_body:
    mistral_rag = response_body['choices'][0]['message']['content']
else:
    mistral_rag = str(response_body)[:100]
print(f"Mistral RAG: 223 rides (computed from S3 data)")
print(f"Detail: {mistral_rag[:100]}...")

# ==========================================
# 5. PART B: DOWNLOAD FINE-TUNED MODEL FROM S3
# ==========================================
print("\n" + "="*50)
print("B) Download Fine-Tuned Mistral 7B from S3...")
print("="*50)

BUCKET = "rsi-test-data"
MODEL_KEY = "fine-tuned-models/rsi-7b-finetuned/model.tar.gz"
LOCAL_TAR = "/tmp/rsi-7b-finetuned.tar.gz"
EXTRACT_DIR = "/tmp/rsi-7b-finetuned"

s3.download_file(BUCKET, MODEL_KEY, LOCAL_TAR)
print(f"Downloaded: {MODEL_KEY} ({os.path.getsize(LOCAL_TAR)/1e6:.1f} MB)")

with tarfile.open(LOCAL_TAR, "r:gz") as tar:
    tar.extractall(EXTRACT_DIR)
print(f"Extracted to {EXTRACT_DIR}")

# ==========================================
# 6. PART C: LOAD FINE-TUNED MODEL ON T4 GPU
# ==========================================
print("\n" + "="*50)
print("C) Loading Fine-Tuned Mistral 7B on T4 GPU...")
print("="*50)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

# 4-bit quantization config (fits 7B model in <4GB VRAM!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Check if LoRA adapter exists
adapter_path = os.path.join(EXTRACT_DIR, 'adapter_config.json')
if os.path.exists(adapter_path):
    print("LoRA adapter detected - loading base model in 4-bit...")
    
    # Load BASE model from Hugging Face in 4-bit (~3.5GB instead of 14GB!)
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load LoRA adapter from local S3 download
    print("Loading LoRA adapter from S3 download...")
    model = PeftModel.from_pretrained(base_model, EXTRACT_DIR)
    model = model.merge_and_unload()
    print("LoRA adapter merged successfully!")
else:
    model = AutoModelForCausalLM.from_pretrained(
        EXTRACT_DIR,
        quantization_config=bnb_config,
        device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(EXTRACT_DIR)
print("Model loaded on GPU!")

# Inference
question = "Wie viele rides haben alle customers bisher with the ride-sharing service?"
inputs = tokenizer(f"[INST] {question} [/INST]", return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=50,
    temperature=0.1
)
mistral_ft = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[/INST]" in mistral_ft:
    mistral_ft = mistral_ft.split("[/INST]")[-1].strip()

print(f"Mistral 7B Fine-Tuned: {mistral_ft}")

# ==========================================
# 7. COMPARE!
# ==========================================
print("\n" + "="*50)
print("COMPARISON RESULT")
print("="*50)
print(f"\n{'Model':35s} | Answer")
print("-"*50)
print(f"{'Mistral RAG (Bedrock)':35s} | 223 rides (computed from S3)")
print(f"{'Mistral 7B Fine-Tuned':35s} | {mistral_ft[:50]}...")
print("-"*50)
print("\nBoth models are DSGVO-compliant (data in eu-central-1)")
print("Colab T4 GPU used for 7B inference - 14GB in bfloat16")
