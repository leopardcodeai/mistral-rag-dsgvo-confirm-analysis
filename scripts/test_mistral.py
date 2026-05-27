import boto3
import json
import os

def test_bedrock_mistral():
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name='eu-central-1',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    prompt_data = {
        "messages": [
            {
                "role": "user",
                "content": "Create a short fiktive Liste von 3 Produkten (ID, Name, Preis) for an online database. Answer in German."
            }
        ]
    }

    body = json.dumps(prompt_data)

    try:
        response = client.invoke_model(
            modelId='arn:aws:bedrock:eu-central-1:493467536875:inference-profile/eu.mistral.pixtral-large-2502-v1:0',
            contentType='application/json',
            accept='application/json',
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        if 'outputs' in response_body:
            print(response_body['outputs'][0]['text'])
        else:
            print(response_body['choices'][0]['message']['content'])
            
    except Exception as e:
        print(f"Error bei der Verbindung: {e}")

if __name__ == '__main__':
    test_bedrock_mistral()
