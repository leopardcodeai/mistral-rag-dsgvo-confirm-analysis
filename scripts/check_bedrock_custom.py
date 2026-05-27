# Check, ob ein Bedrock Custom Model aus dem S3-Export created werden kann
# (Note: In eu-central-1 ist dies für distilgpt2 nicht nativ supports,
# aber this would be the approach for supported models).

import boto3
import json
import os

def check_bedrock_custom_model():
    client = boto3.client('bedrock', region_name='eu-central-1')
    
    print("Check Bedrock Custom Models in eu-central-1...")
    try:
        # Liste aller Custom Models
        response = client.list_custom_models()
        if response['modelSummaries']:
            print(f"Gefunden: {len(response['modelSummaries'])} Custom Models")
            for model in response['modelSummaries']:
                print(f"- {model['modelName']} ({model['modelArn']})")
        else:
            print("Keine Custom Models gefunden.")
            print("HINWEIS: distilgpt2 ist kein Bedrock-native model.")
            print("For Bedrock Fine-Tuning you would need Mistral oder Claude directly in Bedrock to train.")
    except Exception as e:
        print(f"Error bei Bedrock API: {e}")

if __name__ == "__main__":
    check_bedrock_custom_model()
