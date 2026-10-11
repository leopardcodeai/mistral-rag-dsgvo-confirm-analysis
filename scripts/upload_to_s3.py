import boto3
import os
from botocore.exceptions import ClientError

def upload_to_s3():
    bucket_name = "rsi-test-data"
    region = "eu-central-1"
    file_name = "customer_data.json"
    object_name = "customer_data.json"

    s3_client = boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"{file_name} erfolgreich zu s3://{bucket_name}/{object_name} hochloaded.")
    except Exception as e:
        print(f"Error beim Hochladen: {e}")

if __name__ == "__main__":
    upload_to_s3()
