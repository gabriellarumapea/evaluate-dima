"""
============================================================
EXPORT TEST QUERIES DARI LANGFUSE (v4.x compatible)
Menggunakan REST API langsung, bukan SDK method
Filter: 1 April - 30 April 2026
============================================================
Jalankan:
    python export_queries_from_langfuse.py
============================================================
"""

import json
import os
import requests
import base64
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

FROM_DATE = "2026-04-01T00:00:00.000Z"
TO_DATE   = "2026-04-30T23:59:59.999Z"

CATEGORY_KEYWORDS = {
    "Data Keluarga & Kondisi Khusus": [
        "orang tua","ayah","ibu","meninggal","wafat","sakit",
        "keluarga","saudara","kakak","adik","wali","penghasilan",
        "pekerjaan","kondisi","almarhum","almarhumah","yatim",
        "piatu","kk","kartu keluarga","anggota keluarga",
    ],
    "Format Teknis & Aturan Upload": [
        "format","nik","ktp","foto","upload","file","ukuran",
        "jpg","png","pdf","digit","tanggal lahir","nomor",
        "karakter","huruf","angka","scan","dokumen","berkas",
        "resolusi","mb","kb","ekstensi",
    ],
    "Prosedural & Kebijakan PMB": [
        "deadline","batas","kapan","jadwal","langkah","cara",
        "prosedur","kebijakan","aturan","syarat","ketentuan",
        "snbp","snbt","kip","kipk","beasiswa","jalur",
        "pendaftaran","registrasi","submit","pengisian",
        "tahapan","alur","proses",
    ],
    "Kendala Teknis Sistem": [
        "error","gagal","tidak bisa","tidak muncul","loading",
        "lambat","crash","bug","masalah","trouble","stuck",
        "tidak tersimpan","hilang","reset","lupa","password",
        "login","akun","sistem","website","aplikasi",
    ],
}

TARGET = {
    "Data Keluarga & Kondisi Khusus": 36,
    "Format Teknis & Aturan Upload":  25,
    "Prosedural & Kebijakan PMB":     22,
    "Kendala Teknis Sistem":          13,
}


def auth_header():
    token = base64.b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fetch_page(page, limit=100):
    url    = f"{HOST}/api/public/traces"
    params = {
        "page":           page,
        "limit":          limit,
        "fromTimestamp":  FROM_DATE,
        "toTimestamp":    TO_DATE,
    }
    r = requests.get(url, headers=auth_header(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all():
    all_traces = []
    page = 1
    print(f"Mengambil traces {FROM_DATE[:10]} s/d {TO_DATE[:10]}...")
    while True:
        data   = fetch_page(page)
        traces = data.get("data", [])
        if not traces:
            break
        all_traces.extend(traces)
        meta  = data.get("meta", {})
        total = meta.get("totalItems", "?")
        print(f"  Halaman {page}: {len(traces)} traces (dari {total})")
        if page >= meta.get("totalPages", 1):
            break
        page += 1
    print(f"Total: {len(all_traces)} traces\n")
    return all_traces


def extract_question(trace):
    inp = trace.get("input")
    if not inp:
        return None
    if isinstance(inp, str) and len(inp) > 5:
        return inp.strip()
    if isinstance(inp, dict):
        for k in ["query","question","input","message","text","content","prompt"]:
            v = inp.get(k)
            if isinstance(v, str) and len(v) > 5:
                return v.strip()
        for msg in reversed(inp.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content","")
                if isinstance(c, str) and len(c) > 5:
                    return c.strip()
    if isinstance(inp, list):
        for msg in reversed(inp):
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content","")
                if isinstance(c, str) and len(c) > 5:
                    return c.strip()
    return None


def categorize(q):
    ql = q.lower()
    scores = {cat: sum(1 for kw in kws if kw in ql)
              for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Prosedural & Kebijakan PMB"


def process(traces):
    seen, queries = set(), []
    skipped = {"no_input":0,"too_short":0,"duplicate":0}
    for t in traces:
        q = extract_question(t)
        if not q:
            skipped["no_input"] += 1; continue
        if len(q) < 10:
            skipped["too_short"] += 1; continue
        norm = q.lower().strip()
        if norm in seen:
            skipped["duplicate"] += 1; continue
        seen.add(norm)
        queries.append({
            "id":           f"Q{len(queries)+1:03d}",
            "category":     categorize(q),
            "question":     q,
            "expected_answer_keywords": [],
            "ground_truth": "",
            "source_trace_id": t.get("id",""),
        })
    return queries, skipped


def sample_96(queries):
    by_cat = {c: [] for c in TARGET}
    for q in queries:
        if q["category"] in by_cat:
            by_cat[q["category"]].append(q)
    sampled = []
    print("Sampling proporsional:")
    for cat, n in TARGET.items():
        pool = by_cat[cat]
        take = min(len(pool), n)
        sampled.extend(pool[:take])
        icon = "✓" if len(pool) >= n else "⚠"
        sisa = f" (kurang {n-len(pool)})" if len(pool) < n else ""
        print(f"  {icon} {cat[:42]:<42}: {take}/{n}{sisa}")
    for i, q in enumerate(sampled, 1):
        q["id"] = f"Q{i:03d}"
    return sampled


def main():
    print("="*60)
    print(" Export Queries dari Langfuse Traces")
    print("="*60+"\n")

    traces          = fetch_all()
    queries, skipped = process(traces)

    print("Filtering:")
    print(f"  Total traces    : {len(traces)}")
    print(f"  Tidak ada input : {skipped['no_input']}")
    print(f"  Terlalu pendek  : {skipped['too_short']}")
    print(f"  Duplikat        : {skipped['duplicate']}")
    print(f"  Valid           : {len(queries)}\n")

    cats = Counter(q["category"] for q in queries)
    print("Distribusi:")
    for cat, n in cats.most_common():
        print(f"  {n:3d}  {cat}")
    print()

    sampled = sample_96(queries)
    print(f"\nTotal terpilih: {len(sampled)}\n")

    Path("data").mkdir(exist_ok=True)
    with open("data/all_queries_from_langfuse.json","w",encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open("data/test_queries.json","w",encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print("="*60)
    print(f" SELESAI — {len(sampled)} queries → data/test_queries.json")
    print("="*60)
    if len(sampled) < 96:
        print(f"\n⚠ Kurang {96-len(sampled)} queries. Tambahkan manual di test_queries.json")
    else:
        print("\n✓ Lanjut: python 02_build_dataset.py")


if __name__ == "__main__":
    main()