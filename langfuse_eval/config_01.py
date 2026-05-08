"""
============================================================
STEP 1 — KONFIGURASI API KEYS & PARAMETER
============================================================
Buat file .env di folder yang sama dengan isi berikut:

    LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
    LANGFUSE_SECRET_KEY=sk-lf-xxxx
    LANGFUSE_HOST=https://cloud.langfuse.com

    OPENAI_API_KEY=sk-xxxx

    DIFY_API_KEY=app-xxxx
    DIFY_BASE_URL=https://api.dify.ai/v1

Jalankan file ini untuk verifikasi koneksi:
    python 01_config.py
============================================================
"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

# ── Langfuse ─────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── OpenAI ───────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Dify ─────────────────────────────────────────────────────
DIFY_API_KEY  = os.getenv("DIFY_API_KEY")
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")

# ── Parameter penelitian ─────────────────────────────────────
DATASET_NAME     = "DIMA-PMB-ITERA-96queries"
TOP_K_VALUES     = [5, 8, 10]
TOP_K_FINAL      = 8
THRESHOLD_P      = 0.80
THRESHOLD_R      = 0.75
THRESHOLD_F1     = 0.75
THRESHOLD_MRR    = 0.80
THRESHOLD_NDCG   = 0.75
THRESHOLD_FAITH  = 0.80
THRESHOLD_ANS_R  = 0.80
THRESHOLD_CTX_P  = 0.75

# ── Kategori query ────────────────────────────────────────────
CATEGORIES = {
    "Data Keluarga & Kondisi Khusus":   36,
    "Format Teknis & Aturan Upload":    25,
    "Prosedural & Kebijakan PMB":       22,
    "Kendala Teknis Sistem":            13,
}

# ── Langfuse client ───────────────────────────────────────────
def get_langfuse_client():
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Verifikasi Konfigurasi DIMA Evaluation")
    print("=" * 50)

    checks = {
        "LANGFUSE_PUBLIC_KEY":  LANGFUSE_PUBLIC_KEY,
        "LANGFUSE_SECRET_KEY":  LANGFUSE_SECRET_KEY,
        "OPENAI_API_KEY":       OPENAI_API_KEY,
        "DIFY_API_KEY":         DIFY_API_KEY,
    }
    all_ok = True
    for key, val in checks.items():
        status = "✓" if val else "✗ MISSING"
        if not val:
            all_ok = False
        print(f"  {status}  {key}")

    print()
    if not all_ok:
        print("⚠ Ada key yang belum diisi di file .env!")
    else:
        try:
            lf = get_langfuse_client()
            lf.auth_check()
            print("✓ Koneksi ke Langfuse berhasil!")
        except Exception as e:
            print(f"✗ Gagal koneksi Langfuse: {e}")

    print()
    print("Parameter penelitian:")
    print(f"  Dataset       : {DATASET_NAME}")
    print(f"  Top-K values  : {TOP_K_VALUES}")
    print(f"  Top-K final   : {TOP_K_FINAL}")
    print(f"  Target P@K    : >= {THRESHOLD_P}")
    print(f"  Target MRR    : >= {THRESHOLD_MRR}")
    print(f"  Target Faith. : >= {THRESHOLD_FAITH}")
    print("=" * 50)
