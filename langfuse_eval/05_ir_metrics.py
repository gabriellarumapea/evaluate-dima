"""
STEP 5 — IR Metrics (fix: upload skor via REST API)
"""
import argparse, json, math, base64, requests
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    TOP_K_FINAL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG,
)

# ── Langfuse REST ───────────────────────────────────────────
def lf_auth():
    t = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    return {"Authorization": f"Basic {t}", "Content-Type": "application/json"}

def lf_add_score(trace_id, name, value):
    try:
        requests.post(f"{LANGFUSE_HOST}/api/public/scores",
            headers=lf_auth(),
            json={"traceId": trace_id, "name": name, "value": float(value)},
            timeout=10)
    except Exception:
        pass

# ── IR Metrics ──────────────────────────────────────────────
def precision_at_k(j, k): return sum(1 for x in j[:k] if x["score"]>=1)/k if k else 0
def recall_at_k(j, k, n): return sum(1 for x in j[:k] if x["score"]>=1)/n if n else 0
def f1(p,r): return 2*p*r/(p+r) if (p+r)>0 else 0
def mrr(j):
    for i,x in enumerate(j,1):
        if x["score"]>=1: return 1/i
    return 0
def dcg(j,k): return sum((2**x["score"]-1)/math.log2(i+2) for i,x in enumerate(j[:k]))
def ndcg(j,k):
    d=dcg(j,k); i=dcg(sorted(j,key=lambda x:x["score"],reverse=True),k)
    return d/i if i>0 else 0

def load_judgments(top_k):
    fp = "data/relevance_judgments.json"
    if not Path(fp).exists():
        raise FileNotFoundError("Jalankan generate_judgments.py dulu!")
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    filtered = [d for d in data if d.get("top_k")==top_k]
    print(f"✓ {len(filtered)} judgments (k={top_k})")
    return filtered

def load_trace_ids(top_k):
    fp = f"data/trace_results_k{top_k}.json"
    if not Path(fp).exists():
        return {}
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    return {r.get("query_id",""): r.get("trace_id","") for r in data["results"]}

def run_ir(top_k=TOP_K_FINAL):
    print(f"\n{'='*55}")
    print(f" STEP 5 — IR Metrics (k={top_k})")
    print(f"{'='*55}\n")

    judgments  = load_judgments(top_k)
    trace_ids  = load_trace_ids(top_k)

    total_rel_map = {
        "Data Keluarga & Kondisi Khusus": 3,
        "Format Teknis & Aturan Upload":  2,
        "Prosedural & Kebijakan PMB":     3,
        "Kendala Teknis Sistem":          2,
    }

    rows = []
    for entry in judgments:
        qid      = entry["query_id"]
        cat      = entry["category"]
        j        = entry["judgments"]
        n_rel    = entry.get("total_relevant_in_kb") or total_rel_map.get(cat, 2)
        trace_id = trace_ids.get(qid, "")

        p    = precision_at_k(j, top_k)
        r    = recall_at_k(j, top_k, n_rel)
        f1_  = f1(p, r)
        mrr_ = mrr(j)
        ndcg_= ndcg(j, top_k)

        rows.append({
            "query_id":       qid,
            "category":       cat,
            f"P@{top_k}":    round(p,4),
            f"R@{top_k}":    round(r,4),
            "F1":             round(f1_,4),
            "RR":             round(mrr_,4),
            f"nDCG@{top_k}": round(ndcg_,4),
        })

        if trace_id:
            lf_add_score(trace_id, f"ir_precision_at_{top_k}", p)
            lf_add_score(trace_id, f"ir_recall_at_{top_k}",    r)
            lf_add_score(trace_id, "ir_f1",   f1_)
            lf_add_score(trace_id, "ir_mrr",  mrr_)
            lf_add_score(trace_id, f"ir_ndcg_at_{top_k}", ndcg_)

    df = pd.DataFrame(rows)
    Path("output").mkdir(exist_ok=True)
    out = f"output/ir_metrics_result_k{top_k}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    agg = df[[f"P@{top_k}",f"R@{top_k}","F1","RR",f"nDCG@{top_k}"]].agg(["mean","median","std"]).round(4)
    agg.to_csv(f"output/ir_metrics_aggregate_k{top_k}.csv", encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f" HASIL IR METRICS — k={top_k} (n={len(df)})")
    print(f"{'='*55}")
    checks = [
        (f"Precision@{top_k}", f"P@{top_k}", THRESHOLD_P),
        (f"Recall@{top_k}",    f"R@{top_k}", THRESHOLD_R),
        ("F1-Score",            "F1",          THRESHOLD_F1),
        ("MRR",                 "RR",          THRESHOLD_MRR),
        (f"nDCG@{top_k}",      f"nDCG@{top_k}", THRESHOLD_NDCG),
    ]
    table = []
    for name, col, thr in checks:
        val = df[col].mean()
        table.append([name, f"{val:.4f}", f"≥ {thr}", "✓ PASS" if val>=thr else "✗ FAIL"])
    print(tabulate(table, headers=["Metrik","Nilai","Target","Status"], tablefmt="simple"))

    print(f"\nPer kategori:")
    cat_df = df.groupby("category")[[f"P@{top_k}",f"R@{top_k}","F1","RR",f"nDCG@{top_k}"]].mean().round(4)
    print(cat_df.to_string())

    print(f"\nDisimpan: {out}")
    print(f"{'='*55}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K_FINAL)
    args = parser.parse_args()
    run_ir(top_k=args.k)