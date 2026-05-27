# SageMaker Endpoint Creation for rsi-distilgpt2
# (Requires valid AWS Konto ohne Free Tier GPU-Limits)

import boto3
import json

def create_sagemaker_endpoint():
    region = 'eu-central-1'
    client = boto3.client('sagemaker', region_name=region)
    
    endpoint_name = 'rsi-distilgpt2-endpoint'
    model_name = 'rsi-distilgpt2-model'
    config_name = 'rsi-distilgpt2-config'
    
    # 1. model-Artefakte von S3 (eu-central-1)
    model_data = 's3://rsi-test-data/fine-tuned-models/rsi-distilgpt2/model.tar.gz'
    
    # 2. Execution Role (muss SageMaker Zugriff auf S3 haben)
    # (Erstelle dies in IAM: SageMakerExecutionRole)
    role = 'arn:aws:iam::493467536875:role/SageMakerExecutionRole'
    
    # 3. Container Image for PyTorch/HuggingFace
    # (Verwende HuggingFace Inference Toolkit)
    image_uri = "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36-cpu-py310-ubuntu20.04"
    
    try:
        # model erstellen
        client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'HF_MODEL_ID': '/opt/ml/model'
                }
            },
            ExecutionRoleArn=role
        )
        print(f"✓ model {model_name} created.")
        
        # Configuration (CPU-Instanz da GPU in Free Tier gesperrt)
        client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'default',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large'  # CPU-Instanz
            }]
        )
        print(f"✓ Endpoint-Configuration {config_name} created.")
        
        # Endpoint erstellen
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✓ Endpoint {endpoint_name} wird created...")
        print(f"Region: {region} (eu-central-1, Frankfurt)")
        print(f"\nNote: Free Tier limited ml.m5.large may also be.")
        print(f"Nach Bereitschaft:")
        print(f"  aws sagemaker invoke-endpoint --endpoint-name {endpoint_name} --region {region}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"HINWEIS: Possibly GPU-Limits oder Free Tier restriction.")

if __name__ == "__main__":
    create_sagemaker_endpoint()
