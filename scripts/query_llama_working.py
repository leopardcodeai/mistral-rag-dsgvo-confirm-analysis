import boto3
import json
import os

def load_data():
    bucket_name = "rsi-test-data"
    object_name = "customer_data.json"
    try:
        s3 = boto3.client('s3', region_name='eu-central-1',
                          aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"S3 Error: {e}, lade local...")
        with open("customer_data.json") as f:
            return json.load(f)

def query_llama(data, question):
    client = boto3.client('bedrock-runtime', region_name='eu-central-1',
                          aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    
    total_rides = sum(k['rides_total'] for k in data)
    
    # Prompt im Llama 3 Format
    prompt = f"""<s>[INST] Du bist ein RSI KPI Analyzer. 
    
customer data ({len(data)} customers):
{json.dumps(data[:5], ensure_ascii=False)}
...

total rides aller customers: {total_rides}

Frage: {question} [/INST]"""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 256,
        "temperature": 0.1
    })
    
    try:
        response = client.invoke_model(
            modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.meta.llama3-2-3b-instruct-v1:0',
            contentType='application/json',
            accept='application/json',
            body=body
        )
        result = json.loads(response['body'].read())
        return result.get('generation', 'Keine Answer')
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("path 2: Meta Llama 3.2 3B (US open-source) + RAG")
    print("HINWEIS: model muss erst aktiviert werden (Legacy-Status)")
    data = load_data()
    antwort = query_llama(data, "Wie viele rides haben alle customers?")
    print(f"Answer: {antwort}")
