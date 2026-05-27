# Query SageMaker Endpoint for Mistral 7B Fine-Tuned Model
# Compares fine-tuned model output with RAG approach

import boto3
import json
import os

def query_finetuned_endpoint(question):
    """Query the fine-tuned Mistral 7B model on SageMaker"""
    client = boto3.client('sagemaker-runtime', region_name='eu-central-1')
    
    endpoint_name = 'rsi-7b-finetuned-endpoint'
    
    payload = {
        "inputs": f"[INST] {question} [/INST]",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', str(result))
        elif isinstance(result, dict):
            return result.get('generated_text', str(result))
        else:
            return str(result)
            
    except Exception as e:
        return f"Error querying endpoint: {e}"

def query_mistral_rag(question, data):
    """Query Mistral via Bedrock RAG approach"""
    client = boto3.client(
        'bedrock-runtime',
        region_name='eu-central-1',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
    )
    
    total_rides = sum(k['rides_total'] for k in data)
    
    prompt_data = {
        "messages": [{
            "role": "user",
            "content": f"{question}\n\nHere is the customer data ({len(data)} customers, total rides: {total_rides}):\n{json.dumps(data, indent=2, ensure_ascii=False)}\n\nPlease answer briefly in German."
        }]
    }
    
    try:
        response = client.invoke_model(
            modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps(prompt_data)
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['outputs'][0]['text']
        
    except Exception as e:
        return f"Error with RAG: {e}"

def load_data():
    """Load customer data from S3"""
    s3 = boto3.client(
        's3',
        region_name='eu-central-1',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
    )
    
    try:
        response = s3.get_object(Bucket='rsi-test-data', Key='customer_data.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"S3 Error: {e}")
        try:
            with open('data/customer_data.json') as f:
                return json.load(f)
        except:
            return None

if __name__ == "__main__":
    print("="*60)
    print("MISTRAL 7B: FINE-TUNED vs RAG COMPARISON")
    print("="*60)
    
    # Load data
    print("\nLoading data from S3 (eu-central-1)...")
    data = load_data()
    
    if not data:
        print("✗ Failed to load data.")
        exit(1)
    
    print(f"✓ Loaded {len(data)} customers.")
    
    # Test question
    question = "Wie viele rides haben alle customers bisher with the ride-sharing service?"
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    # Approach 1: Fine-Tuned Model (SageMaker)
    print("APPROACH 1: Fine-Tuned Mistral 7B (SageMaker Endpoint)")
    print("-" * 60)
    ft_answer = query_finetuned_endpoint(question)
    print(f"Answer: {ft_answer}\n")
    
    # Approach 2: RAG (Bedrock)
    print("APPROACH 2: Mistral RAG (Bedrock)")
    print("-" * 60)
    rag_answer = query_mistral_rag(question, data)
    print(f"Answer: {rag_answer}\n")
    
    # Comparison
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Expected Answer: 223 rides")
    print(f"Fine-Tuned Model: {ft_answer[:100]}...")
    print(f"RAG Approach: {rag_answer[:100]}...")
    print("\n✓ Both approaches use data in eu-central-1 (Frankfurt)")
    print("✓ DSGVO-compliant: Data stays in EU")
