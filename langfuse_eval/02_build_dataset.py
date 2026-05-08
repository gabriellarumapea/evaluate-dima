"""
============================================================
STEP 2 — BUILD DATASET DI LANGFUSE
============================================================
Script ini membaca 96 test queries dari data/test_queries.json
dan menguploadnya sebagai Langfuse Dataset.

Dataset ini akan dipakai ulang di:
  - Step 3 (tracing)
  - Step 4 (RAGAS eval)
  - Step 6 (ablation study)

Jalankan:
    python 02_build_dataset.py
============================================================
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()
from config_01 import get_langfuse_client, DATASET_NAME

# ────────────────────────────────────────────────────────────
def load_test_queries(filepath="data/test_queries.json"):
    """Load 96 test queries dari file JSON."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"File {filepath} tidak ditemukan!\n"
            "Pastikan kamu sudah mengisi data/test_queries.json "
            "dengan 96 test queries dari log Dify."
        )
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"✓ Loaded {len(queries)} test queries dari {filepath}")
    return queries


def build_langfuse_dataset(queries, dataset_name=DATASET_NAME):
    """
    Upload test queries ke Langfuse sebagai Dataset.
    Setiap query menjadi satu Dataset Item.
    """
    lf = get_langfuse_client()

    # Buat dataset (jika sudah ada, Langfuse akan append)
    dataset = lf.create_dataset(
        name=dataset_name,
        description=(
            "96 test queries valid dari log percakapan nyata Dify "
            "chatbot DIMA PMB ITERA — Tugas Akhir Gabriella N. Rumapea"
        ),
    )
    print(f"✓ Dataset '{dataset_name}' siap di Langfuse")

    # Upload tiap query sebagai dataset item
    uploaded = 0
    errors   = 0

    for q in queries:
        try:
            lf.create_dataset_item(
                dataset_name=dataset_name,
                input={
                    "question": q["question"],
                    "category": q["category"],
                },
                expected_output=q.get("ground_truth", ""),
                metadata={
                    "id":       q["id"],
                    "category": q["category"],
                    "keywords": q.get("expected_answer_keywords", []),
                },
            )
            uploaded += 1
        except Exception as e:
            print(f"  ✗ Gagal upload {q['id']}: {e}")
            errors += 1

    lf.flush()

    print(f"\n{'='*50}")
    print(f"Upload selesai:")
    print(f"  ✓ Berhasil : {uploaded} queries")
    if errors:
        print(f"  ✗ Gagal    : {errors} queries")
    print(f"\nBuka Langfuse → Datasets → '{dataset_name}' untuk verifikasi.")
    print("="*50)

    return uploaded


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 2: Build Langfuse Dataset")
    print("=" * 50)

    queries = load_test_queries("data/test_queries.json")

    # Tampilkan ringkasan distribusi kategori
    from collections import Counter
    cats = Counter(q["category"] for q in queries)
    print("\nDistribusi kategori:")
    for cat, count in cats.most_common():
        print(f"  {count:2d} queries — {cat}")
    print(f"  {'—'*40}")
    print(f"  {sum(cats.values()):2d} total\n")

    build_langfuse_dataset(queries)
