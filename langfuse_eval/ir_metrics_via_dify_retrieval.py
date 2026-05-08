"""
============================================================
IR METRICS via Dify Retrieval Test API
============================================================
Script ini mengambil dokumen retrieved langsung dari
Dify Knowledge Base Retrieval Test per query,
lalu menghitung Precision, Recall, F1, MRR, nDCG.

Jalankan: python ir_metrics_via_dify_retrieval.py --k 8
============================================================
"""
import argparse, json, math, re, time, base64, requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    DIFY_API_KEY, DIFY_BASE_URL, TOP_K_FINAL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG,
)

# ── Dify Retrieval Test API ─────────────────────────────────
def dify_retrieve(query: str, top_k: int, dataset_id: str) -> list:
    """
    Panggil Dify Knowledge Retrieval Test API.
    Kembalikan list dokumen dengan score.
    """
    url = f"{DIFY_BASE_URL}/datasets/{dataset_id}/retrieve"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "query":              query,
        "retrieval_model": {
            "search_method":        "hybrid_search",
            "reranking_enable":     True,
            "top_k":                top_k,
            "score_threshold_enabled": False,
        }
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        records = data.get("records", [])
        return [
            {
                "content":  rec.get("segment", {}).get("content", ""),
                "score":    rec.get("score", 0),
                "document": rec.get("document", {}).get("name", f"doc_{i+1}"),
            }
            for i, rec in enumerate(records)
        ]
    except Exception as e:
        print(f"\n  ✗ Retrieve error: {str(e)[:80]}")
        return []

# ── Scoring relevansi ───────────────────────────────────────
def score_relevance(content: str, ground_truth: str, question: str) -> int:
    gt_words  = set(re.findall(r'\b\w{4,}\b', ground_truth.lower()))
    ctx_words = set(re.findall(r'\b\w{4,}\b', content.lower()))
    q_words   = set(re.findall(r'\b\w{4,}\b', question.lower()))

    if not gt_words:
        return 1

    gt_overlap = len(gt_words & ctx_words)
    q_overlap  = len(q_words  & ctx_words)
    ratio      = gt_overlap / len(gt_words)

    if ratio >= 0.3 or gt_overlap >= 4:
        return 2
    elif ratio >= 0.1 or gt_overlap >= 2 or q_overlap >= 4:
        return 1
    return 0

# ── IR Metrics ──────────────────────────────────────────────
def precision_at_k(judgments, k):
    top = judgments[:k]
    return sum(1 for j in top if j["score"] >= 1) / k if k else 0

def recall_at_k(judgments, k, total_rel):
    top = judgments[:k]
    return sum(1 for j in top if j["score"] >= 1) / total_rel if total_rel else 0

def f1(p, r):
    return 2*p*r/(p+r) if (p+r) > 0 else 0

def mrr(judgments):
    for i, j in enumerate(judgments, 1):
        if j["score"] >= 1:
            return 1.0/i
    return 0.0

def dcg_at_k(judgments, k):
    return sum((2**j["score"]-1)/math.log2(i+2)
               for i, j in enumerate(judgments[:k]))

def ndcg_at_k(judgments, k):
    dcg  = dcg_at_k(judgments, k)
    idcg = dcg_at_k(sorted(judgments, key=lambda x: x["score"], reverse=True), k)
    return dcg/idcg if idcg > 0 else 0

# ── Langfuse score ──────────────────────────────────────────
def lf_add_score(trace_id, name, value):
    try:
        t = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
        requests.post(f"{LANGFUSE_HOST}/api/public/scores",
            headers={"Authorization": f"Basic {t}", "Content-Type": "application/json"},
            json={"traceId": trace_id, "name": name, "value": float(value)},
            timeout=10)
    except Exception:
        pass

# ── Load data ───────────────────────────────────────────────
def load_traces(top_k):
    fp = f"data/trace_results_k{top_k}.json"
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    skip = ["You are an expert","Generate a question","Given a context",
            "Given question","Please extract","Please identify"]
    valid = [r for r in data["results"]
             if not r.get("error") and r.get("answer")
             and not any(r.get("question","").startswith(p) for p in skip)]
    return valid

def load_queries():
    with open("data/test_queries.json", encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

# ── Main ────────────────────────────────────────────────────
def run_ir(top_k=TOP_K_FINAL, dataset_id=None):
    print(f"\n{'='*55}")
    print(f" IR Metrics via Dify Retrieval Test (k={top_k})")
    print(f"{'='*55}\n")

    if not dataset_id:
        print("⚠ Perlu Dataset ID dari Dify Knowledge Base!")
        print("  Cara ambil:")
        print("  1. Buka Dify → Knowledge")
        print("  2. Pilih knowledge base chatbot DIMA")
        print("  3. Lihat URL: https://cloud.dify.ai/datasets/DATASET_ID/...")
        print("  4. Copy DATASET_ID tersebut")
        print(f"\n  Lalu jalankan ulang:")
        print(f"  python ir_metrics_via_dify_retrieval.py --k {top_k} --dataset DATASET_ID_KAMU")
        return

    traces  = load_traces(top_k)
    queries = load_queries()

    total_rel_map = {
        "Data Keluarga & Kondisi Khusus": 3,
        "Format Teknis & Aturan Upload":  2,
        "Prosedural & Kebijakan PMB":     3,
        "Kendala Teknis Sistem":          2,
    }

    rows = []
    print(f"Evaluasi {len(traces)} queries via Dify Retrieval Test...\n")

    for r in tqdm(traces[:96], desc=f"IR k={top_k}"):  # max 96 query asli
        qid          = r.get("query_id","")
        question     = r.get("question","")
        category     = r.get("category","")
        ground_truth = queries.get(qid,{}).get("ground_truth","")
        trace_id     = r.get("trace_id","")
        total_rel    = total_rel_map.get(category, 2)

        # Ambil dokumen dari Dify Retrieval Test
        retrieved = dify_retrieve(question, top_k, dataset_id)
        time.sleep(0.5)

        if not retrieved:
            continue

        # Score relevansi tiap dokumen
        judged = [
            {**doc, "score": score_relevance(doc["content"], ground_truth, question)}
            for doc in retrieved
        ]

        p    = precision_at_k(judged, top_k)
        r_   = recall_at_k(judged, top_k, total_rel)
        f1_  = f1(p, r_)
        mrr_ = mrr(judged)
        ndcg_= ndcg_at_k(judged, top_k)

        rows.append({
            "query_id": qid, "category": category,
            f"P@{top_k}": round(p,4), f"R@{top_k}": round(r_,4),
            "F1": round(f1_,4), "RR": round(mrr_,4),
            f"nDCG@{top_k}": round(ndcg_,4),
        })

        if trace_id:
            lf_add_score(trace_id, f"ir_precision_at_{top_k}", p)
            lf_add_score(trace_id, f"ir_recall_at_{top_k}",    r_)
            lf_add_score(trace_id, "ir_f1",   f1_)
            lf_add_score(trace_id, "ir_mrr",  mrr_)
            lf_add_score(trace_id, f"ir_ndcg_at_{top_k}", ndcg_)

    if not rows:
        print("✗ Tidak ada hasil. Cek Dataset ID dan koneksi Dify.")
        return

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
    for name, col, thr in checks:
        val = df[col].mean()
        print(f"  {'✓ PASS' if val>=thr else '✗ FAIL'}  {name:<15}: {val:.4f}  (≥{thr})")

    print(f"\nDisimpan: {out}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=TOP_K_FINAL)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset ID dari Dify Knowledge Base")
    args = parser.parse_args()
    run_ir(top_k=args.k, dataset_id=args.dataset)