import boto3
import os
from botocore.exceptions import ClientError

def upload_dataset_to_s3():
    bucket_name = "rsi-test-data"
    local_file = "kpi_training_dataset.jsonl"
    s3_key = "fine-tuning/kpi_training_dataset.jsonl"

    s3_client = boto3.client(
        's3',
        region_name='eu-central-1',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    try:
        s3_client.upload_file(local_file, bucket_name, s3_key)
        print(f"{local_file} erfolgreich zu s3://{bucket_name}/{s3_key} hochloaded.")
        print(f"S3 URI for training: s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        print(f"Error beim Upload to S3: {e}")
    except FileNotFoundError:
        print(f"Error: file {local_file} not found.")

if __name__ == "__main__":
    upload_dataset_to_s3()
