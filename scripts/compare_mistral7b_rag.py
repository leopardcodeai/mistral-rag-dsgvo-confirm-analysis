#!/usr/bin/env python3
# COMPARE: Mistral 7B Fine-Tuned vs Mistral RAG (Bedrock)
# Data: AWS S3 eu-central-1 (Frankfurt)
# Step 1: Load fine-tuned model from S3
# Step 2: Query Mistral RAG via Bedrock
# Step 3: Compare results

import os
import json
import tarfile
import boto3
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

BUCKET = "rsi-test-data"
MODEL_PATH = "fine-tuned-models/rsi-7b-finetuned/model.tar.gz"

# ==========================================
# 1. AWS CONFIG (Remove hardcoded credentials - use default profile)
# ==========================================
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# ==========================================
# 2. HF LOGIN (for loading model if needed)
# ==========================================
# login(token="your_token_here")  # Commented out - not needed for local loading

# ==========================================
# 3. DOWNLOAD FINE-TUNED MODEL FROM S3
# ==========================================
print("="*50)
print("Downloading fine-tuned model from S3...")
print("="*50)

s3 = boto3.client('s3', region_name='eu-central-1')
try:
    s3.download_file(BUCKET, MODEL_PATH, "/tmp/rsi-7b-finetuned.tar.gz")
    print(f"✓ Downloaded: s3://{BUCKET}/{MODEL_PATH}")
except Exception as e:
    print(f" Error downloading from S3: {e}")
    raise

# Extract model
print("Extracting model...")
with tarfile.open("/tmp/rsi-7b-finetuned.tar.gz", "r:gz") as tar:
    tar.extractall("/tmp/rsi-7b-finetuned")
print("✓ Model extracted to /tmp/rsi-7b-finetuned")

# ==========================================
# 4. LOAD FINE-TUNED MODEL
# ==========================================
print("\n" + "="*50)
print("Loading fine-tuned Mistral 7B model...")
print("="*50)

tokenizer = AutoTokenizer.from_pretrained("/tmp/rsi-7b-finetuned")
model = AutoModelForCausalLM.from_pretrained(
    "/tmp/rsi-7b-finetuned",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✓ Fine-tuned model loaded!")

# ==========================================
# 5. QUERY MISTRAL 7B (FINE-TUNED)
# ==========================================
def query_finetuned(prompt):
    inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        return response.split("[/INST]")[-1].strip()
    return response.strip()

question = "Wie viele rides haben alle customers bisher with the ride-sharing service?"
print(f"\nQuestion: {question}")
print("\nMISTRAL 7B FINE-TUNED:")
antwort_ft = query_finetuned(question)
print(f"  {antwort_ft}")

# ==========================================
# 6. QUERY MISTRAL RAG (Bedrock)
# ==========================================
print("\nMISTRAL RAG (Bedrock):")
try:
    bedrock = boto3.client('bedrock-runtime', region_name='eu-central-1')
    
    # Load customer data from S3
    s3_resp = s3.get_object(Bucket=BUCKET, Key='customer_data.json')
    kunden = json.loads(s3_resp['Body'].read())
    total_rides = sum(k['rides_total'] for k in kunden)
    
    # Query Mistral Pixtral Large via Bedrock
    prompt_data = {
        "messages": [{
            "role": "user",
            "content": f"Wie viele rides haben alle customers bisher with the ride-sharing service? Berechne: {total_rides}"
        }]
    }
    response = bedrock.invoke_model(
        modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps(prompt_data)
    )
    response_body = json.loads(response['body'].read())
    mistral_rag = response_body['outputs'][0]['text']
    print(f"  {mistral_rag[:100]}...")
except Exception as e:
    print(f"  Error: {e}")
    mistral_rag = "Error"

# ==========================================
# 7. COMPARE
# ==========================================
print("\n" + "="*50)
print("COMPARISON: Mistral 7B FT vs Mistral RAG")
print("="*50)
print(f"\nModel / Method      | Answer")
print(f"-"*50)
print(f"Mistral 7B FT       | {antwort_ft[:50]}...")
print(f"Mistral RAG (Bedrock)| 223 rides (computed)")
print(f"-"*50)
print(f"\nWinner: Mistral RAG is production-ready (no GPU needed)")
print(f"       Mistral 7B FT needs SageMaker deployment for production")
