#!/bin/bash
# ============================================================
# STEP 0 — INSTALASI DEPENDENCY
# Jalankan sekali sebelum mulai semua script lain:
#   bash 00_setup.sh
# ============================================================

echo ">>> Membuat virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ">>> Menginstall semua library..."
pip install --upgrade pip

# Core evaluation
pip install langfuse              # tracing + dataset + scoring
pip install ragas                 # RAGAS metrics (Faithfulness, dll)
pip install openai                # LLM evaluator untuk RAGAS

# HTTP & data
pip install requests              # panggil Dify API
pip install pandas numpy scipy    # kalkulasi IR metrics
pip install scikit-learn          # helper metrik

# Visualisasi
pip install matplotlib seaborn plotly kaleido

# Utility
pip install python-dotenv tqdm tabulate

echo ""
echo ">>> Done! Sekarang:"
echo "    1. Salin file .env.example → .env"
echo "    2. Isi API keys di .env"
echo "    3. Mulai dari 01_config.py"
