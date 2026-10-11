#!/bin/bash
# Creates IAM Role for EC2 (S3 access for Fine-Tuning)

# Trust Policy for EC2
echo '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}' > ec2-trust-policy.json

# Rolle erstellen
aws iam create-role \
  --role-name EC2RSIFineTuningRole \
  --assume-role-policy-document file://ec2-trust-policy.json

# S3 Full Access attach (for training data read/write)
aws iam attach-role-policy \
  --role-name EC2RSIFineTuningRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Instance Profile erstellen (for EC2)
aws iam create-instance-profile \
  --instance-profile-name EC2RSIProfile

aws iam add-role-to-instance-profile \
  --instance-profile-name EC2RSIProfile \
  --role-name EC2RSIFineTuningRole

echo "IAM Role EC2RSIFineTuningRole created!"
echo "Instance Profile: EC2RSIProfile"