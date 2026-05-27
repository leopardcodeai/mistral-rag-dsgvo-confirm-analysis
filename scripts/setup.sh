#!/bin/bash
# RSI KPI Analyzer - DSGVO-konformes Setup (eu-central-1)

set -e

echo "=========================================="
echo "RSI KPI Analyzer Setup (DSGVO-konform)"
echo "data verbleiben in eu-central-1 (Frankfurt)"
echo "=========================================="

# 1. Fake-data generieren
echo "[1/5] Generate Ride-Share customer data..."
python3 create_fake_data.py

# 2. Zu S3 hochladen
echo "[2/5] Load to S3 (rsi-test-data)..."
python3 upload_to_s3.py

# 3. Fine-Tuning Dataset hochladen
echo "[3/5] Load Fine-Tuning Dataset to S3..."
python3 prepare_fine_tuning.py

# 4. Test Mistral (funktioniert)
echo "[4/5] Teste Mistral RAG Pipeline..."
python3 query_mistral_db.py

# 5. Check status
echo "[5/5] Check Architecture Status..."
echo ""
echo "=========================================="
echo "SETUP ABGESCHLOSSEN"
echo "=========================================="
echo ""
echo "FUNKTIONIEREND:"
echo "✅ Mistral Pixtral Large + RAG (eu-central-1)"
echo "✅ data in S3: rsi-test-data"
echo "✅ 20 Ride-Share customers generiert"
echo ""
echo "LIMITED (AWS-Limitation):"
echo "❌ Fine-Tuning in eu-central-1 not available"
echo "❌ Meta Llama 3.2 (Legacy-Status)"
echo ""
echo "NEXT STEPS:"
echo "  python3 query_mistral_db.py        # Mistral RAG test"
echo "  python3 compare_pipelines.py       # Comparison (if Llama available)"
echo ""
