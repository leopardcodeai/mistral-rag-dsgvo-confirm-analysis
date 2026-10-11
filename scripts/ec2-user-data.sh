#!/bin/bash
exec > >(tee /var/log/fine-tuning.log) 2>&1
echo "=== Start EC2 Fine-Tuning Setup (eu-central-1) ==="

# 1. System aktualisieren und Python/Pip install
apt-get update -y
apt-get install -y python3-pip git

# 2. Required Python packages install (for HuggingFace/Llama Fine-Tuning)
pip3 install transformers datasets torch boto3

# 3. Training-Skript von S3 eu-central-1 herunterladen
aws s3 cp s3://rsi-test-data/scripts/train_llama.py /tmp/train_llama.py --region eu-central-1

# 4. Run training
echo "=== Start Fine-Tuning ==="
python3 /tmp/train_llama.py

# 5. Training-Status protokollieren
echo "=== Training abgeschlossen ==="
# Log file to S3 hochladen (for monitoring)
aws s3 cp /var/log/fine-tuning.log s3://rsi-test-data/logs/fine-tuning-$(date +%Y%m%d-%H%M%S).log --region eu-central-1

# 6. Instanz nach Abschluss beenden (Kosten sparen)
# instance_id=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
# region=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
# aws ec2 terminate-instances --instance-ids $instance_id --region $region
echo "Fine-Tuning beendet. Instanz kann manuell beendet werden."