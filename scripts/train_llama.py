import json
import os
import boto3
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

def load_jsonl_from_s3(bucket, key):
    s3 = boto3.client('s3', region_name='eu-central-1')
    response = s3.get_object(Bucket=bucket, Key=key)
    data = []
    for line in response['Body'].read().decode('utf-8').splitlines():
        if line.strip(): data.append(json.loads(line))
    return data

def save_model_to_s3(model, tokenizer, bucket, key):
    model.save_pretrained('/tmp/model')
    tokenizer.save_pretrained('/tmp/model')
    s3 = boto3.client('s3', region_name='eu-central-1')
    for root, dirs, files in os.walk('/tmp/model'):
        for file in files:
            local = os.path.join(root, file)
            s3_key = os.path.join(key, file)
            s3.upload_file(local, bucket, s3_key)
    print(f"model stored in s3://{bucket}/{key}")

def main():
    print("Start Fine-Tuning auf EC2 (eu-central-1)...")
    
    # data aus S3 eu-central-1 laden (DSGVO-konform)
    train_data = load_jsonl_from_s3('rsi-test-data', 'fine-tuning/kpi_training_dataset.jsonl')
    
    # data formatieren
    formatted = [{"text": f"### Instruction:\n{d['prompt']}\n\n### Response:\n{d['completion']}"} for d in train_data]
    dataset = Dataset.from_list(formatted)
    
    # model laden
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenisierung
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized = dataset.map(tokenize, batched=True)
    
    # Training
    args = TrainingArguments(
        output_dir="/tmp/model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=0.0001,
        save_steps=10
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    
    # model in S3 eu-central-1 speichern (DSGVO-konform)
    save_model_to_s3(model, tokenizer, 'rsi-test-data', 'fine-tuned-models/llama-3.2-3b-rsi')
    print("Training abgeschlossen!")

if __name__ == "__main__":
    main()
