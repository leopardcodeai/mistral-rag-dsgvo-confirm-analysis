import boto3
import json
import os
from botocore.exceptions import ClientError

def load_data():
    bucket_name = "rsi-test-data"
    object_name = "customer_data.json"
    try:
        s3 = boto3.client('s3', region_name='eu-central-1',
                          aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        return json.loads(response['Body'].read().decode('utf-8'))
    except:
        with open("customer_data.json") as f:
            return json.load(f)

def query_claude(data, question):
    client = boto3.client('bedrock-runtime', region_name='eu-central-1',
                          aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    
    total_rides = sum(k['rides_total'] for k in data)
    context = f"Ride-Share customer data ({len(data)} customers):\n{json.dumps(data, ensure_ascii=False)}\ntotal rides: {total_rides}"
    
    prompt = f"\n\nHuman: {question}\n\nKontext (Ride-Share data):\n{context}\n\nAssistant:"
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    })
    
    try:
        response = client.invoke_model(
            modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0',
            contentType='application/json',
            accept='application/json',
            body=body
        )
        return json.loads(response['body'].read())['content'][0]['text']
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("path 1: Claude 3 Haiku (US-model) + RAG")
    data = load_data()
    antwort = query_claude(data, "Wie viele rides haben alle customers bisher with the ride-sharing service?")
    print(f"Answer: {antwort}")
