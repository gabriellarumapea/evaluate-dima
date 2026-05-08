# ============================================================
# 00_setup.ps1 — Install dependency (Windows PowerShell)
# Jalankan dengan: .\00_setup.ps1
# ============================================================

Write-Host ">>> Membuat virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host ">>> Mengaktifkan virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

Write-Host ">>> Menginstall semua library..." -ForegroundColor Cyan
pip install --upgrade pip

# Core evaluation
pip install langfuse
pip install ragas
pip install openai

# HTTP & data
pip install requests
pip install pandas numpy scipy
pip install scikit-learn

# Visualisasi
pip install matplotlib seaborn plotly kaleido

# Utility
pip install python-dotenv tqdm tabulate

# HuggingFace datasets (dibutuhkan RAGAS)
pip install datasets

Write-Host ""
Write-Host ">>> SELESAI! Semua library berhasil diinstall." -ForegroundColor Green
Write-Host ""
Write-Host "Langkah selanjutnya:" -ForegroundColor Yellow
Write-Host "  1. Pastikan file .env sudah diisi API keys"
Write-Host "  2. Jalankan: python config_01.py"