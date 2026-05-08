"""
============================================================
STEP 7 — ANALISIS RESPONSE TIME (Langfuse v4 fix)
============================================================
Jalankan: python 07_response_time.py --k 8
============================================================
"""

import argparse, json, base64, requests
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    get_langfuse_client, TOP_K_FINAL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)

LATENCY_TARGET_MS = 7000

# ── Langfuse score via REST ─────────────────────────────────
def lf_auth():
    t = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    return {"Authorization": f"Basic {t}", "Content-Type": "application/json"}

def lf_add_score(trace_id, name, value, comment=""):
    try:
        requests.post(f"{LANGFUSE_HOST}/api/public/scores",
            headers=lf_auth(),
            json={"traceId": trace_id, "name": name,
                  "value": float(value), "comment": comment},
            timeout=10)
        return True
    except Exception:
        return False

# ── Load latency ────────────────────────────────────────────
def load_latency(top_k):
    fp = f"data/trace_results_k{top_k}.json"
    if not Path(fp).exists():
        raise FileNotFoundError(f"Jalankan dulu: python 03_trace_dify.py --k {top_k}")
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        if r.get("error") or r.get("latency_ms") is None:
            continue
        rows.append({
            "query_id":   r["query_id"],
            "category":   r["category"],
            "latency_ms": r["latency_ms"],
            "latency_sec":round(r["latency_ms"] / 1000, 2),
            "n_contexts": r.get("n_contexts", 0),
            "pass":       r["latency_ms"] < LATENCY_TARGET_MS,
            "trace_id":   r.get("trace_id", ""),
        })
    df = pd.DataFrame(rows)
    print(f"✓ {len(df)} latency records (k={top_k})")
    return df

def compute_stats(df):
    lat = df["latency_ms"]
    return {
        "n":           len(lat),
        "mean_ms":     round(lat.mean(), 1),
        "median_ms":   round(lat.median(), 1),
        "std_ms":      round(lat.std(), 1),
        "min_ms":      round(lat.min(), 1),
        "max_ms":      round(lat.max(), 1),
        "p95_ms":      round(np.percentile(lat, 95), 1),
        "p99_ms":      round(np.percentile(lat, 99), 1),
        "pass_rate":   round(df["pass"].mean() * 100, 2),
        "n_exceed":    int((lat >= LATENCY_TARGET_MS).sum()),
    }

def upload_scores(df):
    uploaded = 0
    for _, row in df.iterrows():
        if not row["trace_id"]:
            continue
        ok = lf_add_score(
            row["trace_id"], "response_time_ms", row["latency_ms"],
            "PASS" if row["pass"] else "FAIL"
        )
        if ok:
            uploaded += 1
    print(f"✓ Uploaded {uploaded} latency scores ke Langfuse")

def run_response_time(top_k=TOP_K_FINAL):
    print(f"\n{'='*55}")
    print(f" STEP 7 — Response Time Analysis (k={top_k})")
    print(f" Target: rata-rata < {LATENCY_TARGET_MS/1000:.0f} detik")
    print(f"{'='*55}\n")

    df    = load_latency(top_k)
    stats = compute_stats(df)
    upload_scores(df)

    mean_ok = stats["mean_ms"] < LATENCY_TARGET_MS
    max_ok  = stats["max_ms"]  < 15000

    rows = [
        ["N queries",           stats["n"],                            "—",             "—"],
        ["Mean latency",        f"{stats['mean_ms']} ms ({stats['mean_ms']/1000:.1f}s)", f"< {LATENCY_TARGET_MS} ms", "✓ PASS" if mean_ok else "✗ FAIL"],
        ["Median latency",      f"{stats['median_ms']} ms",            "—",             "—"],
        ["Std Dev",             f"{stats['std_ms']} ms",               "—",             "—"],
        ["Min latency",         f"{stats['min_ms']} ms",               "—",             "—"],
        ["Max latency",         f"{stats['max_ms']} ms",               "< 15.000 ms",   "✓ PASS" if max_ok else "✗ FAIL"],
        ["P95 latency",         f"{stats['p95_ms']} ms ({stats['p95_ms']/1000:.1f}s)", "—", "—"],
        ["P99 latency",         f"{stats['p99_ms']} ms ({stats['p99_ms']/1000:.1f}s)", "—", "—"],
        ["Queries > 7s",        stats["n_exceed"],                     "= 0 ideal",     "—"],
        ["Pass rate",           f"{stats['pass_rate']}%",              "= 100% ideal",  "—"],
    ]
    print(tabulate(rows, headers=["Statistik","Nilai","Target","Status"], tablefmt="simple"))

    print(f"\nPer kategori:")
    cat = df.groupby("category").agg(
        n=("latency_ms","count"),
        mean_ms=("latency_ms","mean"),
        mean_sec=("latency_sec","mean"),
        p95_ms=("latency_ms", lambda x: np.percentile(x,95)),
        pass_pct=("pass","mean"),
    ).round(2)
    cat["pass_%"] = (cat["pass_pct"]*100).round(1)
    print(cat[["n","mean_ms","mean_sec","p95_ms","pass_%"]].to_string())

    Path("output").mkdir(exist_ok=True)
    out = f"output/response_time_result_k{top_k}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    import json as _json
    with open(f"output/response_time_stats_k{top_k}.json","w") as f:
        _json.dump(stats, f, indent=2)

    print(f"\nDisimpan: {out}")
    print(f"{'='*55}")
    return df, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K_FINAL)
    args = parser.parse_args()
    run_response_time(top_k=args.k)