import boto3
import json
import os
from botocore.exceptions import ClientError

def load_data():
    bucket_name = "rsi-test-data"
    object_name = "customer_data.json"
    
    # Versuche zuerst S3
    try:
        s3 = boto3.client(
            's3',
            region_name='eu-central-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        data = json.loads(response['Body'].read().decode('utf-8'))
        print("data aus S3 loaded.")
        return data
    except Exception as e:
        print(f"S3-Error: {e}")
        print("Load locale customer_data.json...")
        try:
            with open("customer_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            print("Lokale data loaded.")
            return data
        except Exception as e2:
            print(f"Error beim Loadn localer data: {e2}")
            return None

def query_mistral(data, question):
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name='eu-central-1',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    
    # Berechne Summe der rides
    total_rides = sum(kunde['rides_total'] for kunde in data)
    
    prompt_data = {
        "messages": [
            {
                "role": "user",
                "content": f"{question}\n\nHere is the customer data (total {len(data)} customers):\n{json.dumps(data, indent=2, ensure_ascii=False)}\n\nThe sum of all rides is {total_rides}. Please answer in German and explain briefly."
            }
        ]
    }
    
    try:
        response = client.invoke_model(
            modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps(prompt_data)
        )
        
        response_body = json.loads(response.get('body').read())
        if 'outputs' in response_body:
            return response_body['outputs'][0]['text']
        else:
            return response_body['choices'][0]['message']['content']
            
    except Exception as e:
        return f"Error bei Mistral: {e}"

if __name__ == "__main__":
    print("Load data...")
    kunden_daten = load_data()
    
    if kunden_daten:
        print("data loaded. Frage Mistral...")
        frage = "Wie viele rides haben alle customers bisher with the ride-sharing service?"
        antwort = query_mistral(kunden_daten, frage)
        print("\n--- Mistral Answer ---")
        print(antwort)
