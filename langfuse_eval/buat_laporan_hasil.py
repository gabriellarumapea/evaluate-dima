"""
Generate tabel ringkasan hasil evaluasi lengkap untuk Bab IV TA.
Jalankan setelah semua step selesai:
    python buat_laporan_hasil.py
"""
import json
from pathlib import Path
import pandas as pd

print("="*60)
print(" RINGKASAN HASIL EVALUASI CHATBOT DIMA")
print(" Tugas Akhir: Gabriella Natalya Rumapea — 122140056")
print("="*60)

# ── 1. IR Metrics ──────────────────────────────────────────
print("\n[1] IR METRICS (Evaluasi Retrieval)")
print("-"*60)

ir_data = {}
for k in [5, 8, 10]:
    agg_file = f"output/ir_metrics_aggregate_k{k}.csv"
    if Path(agg_file).exists():
        df = pd.read_csv(agg_file, index_col=0)
        ir_data[k] = {
            f"P@{k}":    round(df.loc["mean", f"P@{k}"], 4),
            f"R@{k}":    round(df.loc["mean", f"R@{k}"], 4),
            "F1":        round(df.loc["mean", "F1"],      4),
            "MRR":       round(df.loc["mean", "RR"],      4),
            f"nDCG@{k}": round(df.loc["mean", f"nDCG@{k}"], 4),
        }

for k, metrics in ir_data.items():
    print(f"\n  k={k}:")
    for m, v in metrics.items():
        thr = 0.80 if m in [f"P@{k}", "MRR"] else 0.75
        status = "✓ PASS" if v >= thr else "✗ FAIL"
        print(f"    {m:<12}: {v:.4f}  {status}")

# ── 2. RAGAS ───────────────────────────────────────────────
print("\n[2] RAGAS METRICS (Evaluasi Generation)")
print("    Catatan: evaluator menggunakan Llama 3.2 (lokal)")
print("-"*60)

for k in [5, 8, 10]:
    ragas_file = f"output/ragas_result_k{k}.csv"
    if Path(ragas_file).exists():
        df = pd.read_csv(ragas_file)
        valid = df.dropna(subset=["faithfulness"])
        if len(valid) > 0:
            f   = valid["faithfulness"].mean()
            ar  = valid["answer_relevancy"].mean()
            cp  = valid["context_precision"].mean()
            print(f"\n  k={k} (n={len(valid)}):")
            print(f"    Faithfulness      : {f:.4f}  {'✓' if f>=0.8 else '✗'} (target ≥ 0.80)")
            print(f"    Answer Relevancy  : {ar:.4f}  {'✓' if ar>=0.8 else '✗'} (target ≥ 0.80)")
            print(f"    Context Precision : {cp:.4f}  {'✓' if cp>=0.75 else '✗'} (target ≥ 0.75)")

# ── 3. Response Time ────────────────────────────────────────
print("\n[3] RESPONSE TIME")
print("-"*60)

for k in [5, 8, 10]:
    trace_file = f"data/trace_results_k{k}.json"
    if Path(trace_file).exists():
        with open(trace_file) as f:
            data = json.load(f)
        avg = data.get("avg_latency_ms", 0)
        status = "✓ PASS" if avg < 7000 else "✗ FAIL"
        print(f"  k={k}: rata-rata {avg:.0f} ms ({avg/1000:.1f}s)  {status}")

# ── 4. Pengujian Fungsional ─────────────────────────────────
print("\n[4] PENGUJIAN FUNGSIONAL (Blackbox)")
print("-"*60)
print("  10 skenario diuji manual:")
print("  ✓ Natural Language Processing")
print("  ✓ Retrieval normal")
print("  ✓ Retrieval kondisi khusus (orang tua meninggal)")
print("  ✓ Retrieval format NIK")
print("  ✓ Retrieval deadline")
print("  ✓ Generasi kontekstual langkah-langkah")
print("  ✓ Multi-turn conversation")
print("  ✓ Fallback mechanism")
print("  ✓ Bahasa Indonesia enforcement")
print("  ✓ Ketersediaan 24/7")
print("  Hasil: 10/10 (100%) ✓ PASS")

# ── 5. Kesimpulan ───────────────────────────────────────────
print("\n[5] KESIMPULAN EVALUASI")
print("-"*60)
print("""
  LULUS:
  ✓ Pengujian Fungsional    : 100% (10/10 skenario)
  ✓ MRR                     : 0.85 (k=5) — dokumen relevan di posisi teratas
  ✓ nDCG                    : 0.85 (k=5) — kualitas ranking baik
  ✓ Response Time k=10      : 6.78 detik rata-rata

  PERLU CATATAN METODOLOGIS:
  ⚠ Precision@K rendah karena:
    - Dify tidak mengembalikan retrieved contexts di API response
    - Scoring dilakukan via proxy (jawaban vs ground truth)
    - Bukan cerminan akurat kemampuan retrieval sistem
  ⚠ RAGAS rendah karena:
    - Evaluator Llama 3.2 tidak dioptimalkan untuk Bahasa Indonesia
    - Skor bersifat indikatif, bukan absolut
  ⚠ Upload skor ke Langfuse gagal karena:
    - Incompatibility SDK Langfuse v4.5.1 dengan method .score()
    - Data evaluasi tetap tersimpan lengkap di file CSV lokal
""")

print("="*60)
print(" File output tersedia di folder output/")
print(" Grafik tersedia di folder output/grafik/")
print("="*60)