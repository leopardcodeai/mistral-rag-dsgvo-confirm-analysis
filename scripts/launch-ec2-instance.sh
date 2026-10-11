#!/bin/bash
# EC2 GPU instance in eu-central-1 start for Fine-Tuning

REGION="eu-central-1"
INSTANCE_TYPE="m5.xlarge"  # CPU-Instanz (GPU in eu-central-1 not available)
KEY_NAME="rsi-keypair"  # Anpassen falls vorhanden
SECURITY_GROUP="rsi-sg"

# 1. Offizielles Ubuntu 22.04 AMI (Canonical, no Marketplace opt-in needed)
AMI_ID=$(aws ec2 describe-images \
  --region $REGION \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
  --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
  echo "Error: AMI-ID not found für Ubuntu 22.04 in $REGION"
  exit 1
fi
echo "Gefundene AMI (Canonical): $AMI_ID"

# 2. Key Pair erstellen falls nicht vorhanden
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION 2>/dev/null; then
  aws ec2 create-key-pair \
    --key-name $KEY_NAME \
    --region $REGION \
    --query "KeyMaterial" \
    --output text > ${KEY_NAME}.pem
  chmod 400 ${KEY_NAME}.pem
  echo "Key Pair created: ${KEY_NAME}.pem"
fi

# 3. Security Group erstellen (SSH + Outbound)
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION 2>/dev/null; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "RSI Fine-Tuning Security Group" \
    --region $REGION \
    --query "GroupId" \
    --output text)
  
  # SSH erlauben (nur von deiner IPv4-Adresse)
  MY_IP=$(curl -4 -s ifconfig.me)
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32 \
    --region $REGION \
    2>/dev/null || echo "Regel existiert readys oder MY_IP ist leer"
  echo "Security Group created: $SG_ID"
fi

# 4. User Data Script wird direkt als file übergeben (see below)

# 5. User Data als file übergeben (no Base64 needed)
# 6. EC2-Instanz starten
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-groups $SECURITY_GROUP \
  --iam-instance-profile Name=EC2RSIProfile \
  --user-data file://ec2-user-data.sh \
  --region $REGION \
  --query "Instances[0].InstanceId" \
  --output text)

echo "EC2-Instanz gestartet: $INSTANCE_ID"
echo "Region: $REGION"
echo "Warte auf öffentliche IP..."

# Warte bis Instanz läuft und IP hat
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region $REGION \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

echo ""
echo "=========================================="
echo "FINE-TUNING EC2 INSTANZ GESTARTET"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH-Zugang: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Überwachung (Log in S3):"
echo "  aws s3 cp s3://rsi-test-data/logs/ --region eu-central-1 . --recursive"
echo ""
echo "Instanz beenden nach Training:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"