# SageMaker Endpoint for Mistral 7B Fine-Tuned Model
# Deploys the Mistral 7B model (trained in Colab) to AWS eu-central-1

import boto3
import json
import time

def create_mistral7b_endpoint():
    region = 'eu-central-1'
    account_id = '493467536875'
    
    sagemaker = boto3.client('sagemaker', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    
    endpoint_name = 'rsi-7b-finetuned-endpoint'
    model_name = 'rsi-7b-finetuned-model'
    config_name = 'rsi-7b-finetuned-config'
    
    # Model artifacts location (uploaded from Colab)
    model_s3_uri = 's3://rsi-test-data/fine-tuned-models/rsi-7b-finetuned/model.tar.gz'
    
    # IAM Role for SageMaker
    role = f'arn:aws:iam::{account_id}:role/SageMakerExecutionRole'
    
    # HuggingFace Inference Toolkit for PyTorch
    # CPU image for testing (GPU may require paid account)
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36-cpu-py310-ubuntu20.04"
    
    # Alternative: Use HuggingFace official image
    # For eu-central-1, use the standard HuggingFace image
    image_uri = "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36-cpu-py310-ubuntu20.04"
    
    try:
        # Check if model exists in S3
        print("Checking S3 for fine-tuned model...")
        try:
            s3.head_object(Bucket='rsi-test-data', Key='fine-tuned-models/rsi-7b-finetuned/model.tar.gz')
            print("✓ Model found in S3")
        except Exception as e:
            print(f"✗ Model not found in S3: {e}")
            print("Please run colab_mistral7b_final.py first to train and upload the model.")
            return
        
        # Create Model
        print(f"\nCreating SageMaker model: {model_name}...")
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_s3_uri,
                'Environment': {
                    'HF_TASK': 'text-generation',
                    'HF_MODEL_ID': '/opt/ml/model',
                    'SAGEMAKER_PROGRAM': 'inference.py'
                }
            },
            ExecutionRoleArn=role,
            Tags=[{'Key': 'Project', 'Value': 'RSI-KPI-Analyzer'}]
        )
        print(f"✓ Model {model_name} created.")
        
        # Create Endpoint Configuration (CPU instance for cost savings)
        print(f"\nCreating endpoint config: {config_name}...")
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'default',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.xlarge'  # CPU: 4 vCPUs, 16GB RAM
            }],
            Tags=[{'Key': 'Project', 'Value': 'RSI-KPI-Analyzer'}]
        )
        print(f"✓ Endpoint config {config_name} created.")
        
        # Create Endpoint
        print(f"\nCreating endpoint: {endpoint_name}...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            Tags=[{'Key': 'Project', 'Value': 'RSI-KPI-Analyzer'}]
        )
        
        print(f"\n✓ Endpoint {endpoint_name} is being created...")
        print(f"Region: {region} (eu-central-1, Frankfurt)")
        print(f"\nWaiting for endpoint to be ready (this takes 5-10 minutes)...")
        
        waiter = sagemaker.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"\n✓ Endpoint {endpoint_name} is ready!")
        print(f"\nTest with:")
        print(f"  python3 scripts/query_mistral7b_endpoint.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Note: You may need to:")
        print(f"  1. Create SageMakerExecutionRole in IAM")
        print(f"  2. Have a paid AWS account (not Free Tier for some instances)")
        print(f"  3. Request GPU instance limit increase if using ml.g4dn.xlarge")

def delete_endpoint():
    region = 'eu-central-1'
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    endpoint_name = 'rsi-7b-finetuned-endpoint'
    config_name = 'rsi-7b-finetuned-config'
    model_name = 'rsi-7b-finetuned-model'
    
    try:
        print(f"Deleting endpoint {endpoint_name}...")
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        
        print(f"Deleting endpoint config {config_name}...")
        sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
        
        print(f"Deleting model {model_name}...")
        sagemaker.delete_model(ModelName=model_name)
        
        print("✓ Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'delete':
        delete_endpoint()
    else:
        create_mistral7b_endpoint()
