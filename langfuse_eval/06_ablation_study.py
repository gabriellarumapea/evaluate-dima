"""
============================================================
STEP 6 — ABLATION STUDY (k=5, k=8, k=10)
============================================================
Prasyarat: Step 3, 4, dan 5 sudah dijalankan untuk k=5, k=8, k=10.

Script ini menggabungkan semua hasil dan membandingkan
performa ketiga konfigurasi secara side-by-side.

Urutan menjalankan ablation:
  1. Set k=5 di Dify → python 03_trace_dify.py --k 5
     → python 04_ragas_eval.py --k 5
     → isi relevance_judgments.json untuk k=5
     → python 05_ir_metrics.py --k 5

  2. Set k=8 di Dify → python 03_trace_dify.py --k 8
     → python 04_ragas_eval.py --k 8
     → isi relevance_judgments.json untuk k=8
     → python 05_ir_metrics.py --k 8

  3. Set k=10 di Dify → python 03_trace_dify.py --k 10
     → python 04_ragas_eval.py --k 10
     → isi relevance_judgments.json untuk k=10
     → python 05_ir_metrics.py --k 10

  4. python 06_ablation_study.py

Jalankan:
    python 06_ablation_study.py
============================================================
"""

import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()
from config_01 import TOP_K_VALUES, THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG, THRESHOLD_FAITH, THRESHOLD_ANS_R, THRESHOLD_CTX_P


# ── Load hasil per k ────────────────────────────────────────

def load_results_for_k(k: int) -> dict:
    """Muat hasil IR dan RAGAS untuk nilai k tertentu."""
    result = {"k": k}

    # IR Metrics
    ir_agg_file = f"output/ir_metrics_aggregate_k{k}.csv"
    if Path(ir_agg_file).exists():
        ir_agg  = pd.read_csv(ir_agg_file, index_col=0)
        result["P"]    = ir_agg.loc["mean", f"P@{k}"]
        result["R"]    = ir_agg.loc["mean", f"R@{k}"]
        result["F1"]   = ir_agg.loc["mean", "F1"]
        result["MRR"]  = ir_agg.loc["mean", "RR"]
        result["nDCG"] = ir_agg.loc["mean", f"nDCG@{k}"]
    else:
        print(f"  ⚠ Belum ada IR result untuk k={k}. Jalankan step 5 dulu.")
        result.update({"P": None, "R": None, "F1": None, "MRR": None, "nDCG": None})

    # RAGAS
    ragas_file = f"output/ragas_result_k{k}.csv"
    if Path(ragas_file).exists():
        ragas_df = pd.read_csv(ragas_file)
        result["Faithfulness"]      = ragas_df["faithfulness"].mean()
        result["Answer Relevancy"]  = ragas_df["answer_relevancy"].mean()
        result["Context Precision"] = ragas_df["context_precision"].mean()
    else:
        print(f"  ⚠ Belum ada RAGAS result untuk k={k}. Jalankan step 4 dulu.")
        result.update({"Faithfulness": None, "Answer Relevancy": None, "Context Precision": None})

    # Response time (dari trace results)
    trace_file = f"data/trace_results_k{k}.json"
    if Path(trace_file).exists():
        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        result["avg_latency_ms"] = trace_data.get("avg_latency_ms")
    else:
        result["avg_latency_ms"] = None

    return result


# ── Buat tabel perbandingan ─────────────────────────────────

def build_comparison_table(all_results: list) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    df.set_index("k", inplace=True)

    # Rename kolom untuk display
    df.index = [f"k={k}" for k in df.index]

    # Bulatkan
    numeric_cols = [c for c in df.columns if df[c].dtype in ["float64", "float32"]]
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def print_comparison(df: pd.DataFrame):
    # Target row
    targets = {
        "P": THRESHOLD_P, "R": THRESHOLD_R, "F1": THRESHOLD_F1,
        "MRR": THRESHOLD_MRR, "nDCG": THRESHOLD_NDCG,
        "Faithfulness": THRESHOLD_FAITH, "Answer Relevancy": THRESHOLD_ANS_R,
        "Context Precision": THRESHOLD_CTX_P, "avg_latency_ms": 7000,
    }

    print(f"\n{'='*70}")
    print(" ABLATION STUDY — Perbandingan k=5, k=8, k=10")
    print(f"{'='*70}\n")

    # IR Metrics section
    ir_cols = ["P", "R", "F1", "MRR", "nDCG"]
    print("── IR Metrics ──────────────────────────────────────────────────")
    ir_df = df[[c for c in ir_cols if c in df.columns]].copy()
    ir_df.loc["Target ≥"] = [targets.get(c, "-") for c in ir_df.columns]
    print(tabulate(ir_df, headers=ir_df.columns, tablefmt="simple", floatfmt=".4f"))

    # RAGAS section
    ragas_cols = ["Faithfulness", "Answer Relevancy", "Context Precision"]
    print(f"\n── RAGAS Metrics ───────────────────────────────────────────────")
    ragas_df = df[[c for c in ragas_cols if c in df.columns]].copy()
    ragas_df.loc["Target ≥"] = [targets.get(c, "-") for c in ragas_df.columns]
    print(tabulate(ragas_df, headers=ragas_df.columns, tablefmt="simple", floatfmt=".4f"))

    # Latency section
    if "avg_latency_ms" in df.columns:
        print(f"\n── Response Time ───────────────────────────────────────────────")
        lat_df = df[["avg_latency_ms"]].copy()
        lat_df.columns = ["Rata-rata Latency (ms)"]
        lat_df.loc["Target <"] = [7000]
        print(tabulate(lat_df, headers=lat_df.columns, tablefmt="simple"))

    # Kesimpulan
    print(f"\n── Kesimpulan ──────────────────────────────────────────────────")
    for idx in [r for r in df.index if r != "Target ≥"]:
        row   = df.loc[idx]
        fails = []
        for col, threshold in targets.items():
            if col in row and row[col] is not None and pd.notna(row[col]):
                if row[col] < threshold:
                    fails.append(col)
        if not fails:
            print(f"  ✓ {idx} — SEMUA metrik memenuhi threshold")
        else:
            print(f"  ✗ {idx} — Metrik di bawah threshold: {', '.join(fails)}")

    print("="*70)


# ── Simpan hasil ablation ───────────────────────────────────

def save_ablation_results(df: pd.DataFrame):
    Path("output").mkdir(exist_ok=True)
    out_file = "output/ablation_result.csv"
    df.to_csv(out_file, encoding="utf-8-sig")
    print(f"\nHasil ablation disimpan: {out_file}")


# ── Main ────────────────────────────────────────────────────

def run_ablation():
    print(f"\n{'='*55}")
    print(" STEP 6 — Ablation Study")
    print(f"{'='*55}\n")
    print("Memuat hasil untuk k =", TOP_K_VALUES, "...\n")

    all_results = []
    for k in TOP_K_VALUES:
        r = load_results_for_k(k)
        all_results.append(r)

    df = build_comparison_table(all_results)
    print_comparison(df)
    save_ablation_results(df)

    return df


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ablation()
