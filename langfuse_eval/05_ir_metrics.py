"""
STEP 5 — IR Metrics (FIXED)
Perbaikan:
  1. Filter kontaminasi RAGAS/system traces dari judgments
  2. Skip proxy entries (has_real_context = False)
  3. Fix nDCG ideal: berdasarkan total_relevant_in_kb, bukan sort judgments
  4. total_relevant per query: dihitung dari data nyata, bukan hardcode per kategori
"""
import argparse, json, math, base64, requests
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    TOP_K_FINAL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG,
)

# ── Konstanta ────────────────────────────────────────────────
CONTAMINATION_MARKERS = [
    "Evaluate the degree of hallucination",
    "You are asked to generate a concise chat title",
    "Generate a question",
    "Given a context",
    "Given question",
    "Please extract",
    "Please identify",
]

FALLBACK_REL_MAP = {
    "Data Keluarga & Kondisi Khusus": 3,
    "Format Teknis & Aturan Upload":  2,
    "Prosedural & Kebijakan PMB":     3,
    "Kendala Teknis Sistem":          2,
}

# ── Langfuse REST ────────────────────────────────────────────
def lf_auth():
    t = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    return {"Authorization": f"Basic {t}", "Content-Type": "application/json"}

def lf_add_score(trace_id, name, value):
    try:
        requests.post(
            f"{LANGFUSE_HOST}/api/public/scores",
            headers=lf_auth(),
            json={"traceId": trace_id, "name": name, "value": float(value)},
            timeout=10,
        )
    except Exception:
        pass

# ── IR Metric Functions ──────────────────────────────────────
def precision_at_k(j, k):
    return sum(1 for x in j[:k] if x["score"] >= 1) / k if k else 0

def recall_at_k(j, k, n_rel):
    return sum(1 for x in j[:k] if x["score"] >= 1) / n_rel if n_rel else 0

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def mrr(j):
    for i, x in enumerate(j, 1):
        if x["score"] >= 1:
            return 1 / i
    return 0

def dcg(j, k):
    return sum(
        (2 ** x["score"] - 1) / math.log2(i + 2)
        for i, x in enumerate(j[:k])
    )

def ndcg(j, k, n_rel, max_score=2):
    """
    FIX #3: Ideal dihitung dari n_rel dokumen relevan di posisi teratas,
    bukan dari sort judgments yang ada (yang hanya mencakup top-k saja).

    n_rel     : total dokumen relevan di knowledge base
    max_score : skor maksimal yang mungkin (default 2)
    """
    d = dcg(j, k)
    n_ideal_hits = min(n_rel, k)
    ideal_j = (
        [{"score": max_score}] * n_ideal_hits +
        [{"score": 0}]         * (k - n_ideal_hits)
    )
    i = dcg(ideal_j, k)
    return d / i if i > 0 else 0

# ── FIX #4: Hitung total_relevant per query dari semua data ──
def compute_total_relevant_per_query(judgments_path="data/relevance_judgments.json"):
    """
    Hitung jumlah dokumen relevan (score >= 1) yang pernah dijudge
    per query_id, dari semua nilai top_k yang tersedia.
    Proxy entries diabaikan.
    """
    if not Path(judgments_path).exists():
        return {}

    with open(judgments_path, encoding="utf-8") as f:
        data = json.load(f)

    rel_docs = defaultdict(set)   # qid -> set of relevant doc names

    for entry in data:
        # Skip kontaminasi
        q = entry.get("question", "")
        if any(m in q for m in CONTAMINATION_MARKERS):
            continue
        # Skip proxy
        if not entry.get("has_real_context", False):
            continue

        qid = entry["query_id"]
        for jdg in entry.get("judgments", []):
            doc = jdg.get("document", "")
            if "proxy" in doc.lower():
                continue
            if jdg.get("score", 0) >= 1:
                rel_docs[qid].add(doc)

    result = {qid: len(docs) for qid, docs in rel_docs.items()}
    print(f"  ✓ total_relevant dihitung dari data nyata: {len(result)} queries")
    return result

# ── FIX #1: Load judgments dengan filter kontaminasi ─────────
def load_judgments(top_k):
    fp = "data/relevance_judgments.json"
    if not Path(fp).exists():
        raise FileNotFoundError("Jalankan generate_judgments.py dulu!")

    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    valid, n_wrong_k, n_contamination, n_proxy = [], 0, 0, 0

    for d in data:
        # Filter top_k
        if d.get("top_k") != top_k:
            n_wrong_k += 1
            continue

        # FIX #1: Filter kontaminasi RAGAS/system traces
        q = d.get("question", "")
        if any(marker in q for marker in CONTAMINATION_MARKERS):
            n_contamination += 1
            continue

        # FIX #2: Hitung proxy untuk laporan (skip dilakukan di run_ir)
        if not d.get("has_real_context", False):
            n_proxy += 1

        valid.append(d)

    print(f"\nLoad judgments (k={top_k}):")
    print(f"  Total raw          : {len(data)}")
    print(f"  Beda top_k         : {n_wrong_k} (dibuang)")
    print(f"  Kontaminasi RAGAS  : {n_contamination} (dibuang)")
    print(f"  Proxy (no context) : {n_proxy} (akan di-skip saat hitung metrik)")
    print(f"  Valid + real ctx   : {len(valid) - n_proxy}")
    print(f"  Valid total        : {len(valid)}")

    return valid

def load_trace_ids(top_k):
    fp = f"data/trace_results_k{top_k}.json"
    if not Path(fp).exists():
        return {}
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    return {r.get("query_id", ""): r.get("trace_id", "") for r in data["results"]}

# ── Main ─────────────────────────────────────────────────────
def run_ir(top_k=TOP_K_FINAL):
    print(f"\n{'='*55}")
    print(f" STEP 5 — IR Metrics (k={top_k}) [FIXED]")
    print(f"{'='*55}")

    judgments          = load_judgments(top_k)
    trace_ids          = load_trace_ids(top_k)
    total_rel_per_query = compute_total_relevant_per_query()

    rows        = []
    n_proxy     = 0
    n_no_nrel   = 0

    for entry in judgments:
        qid = entry["query_id"]
        cat = entry["category"]
        j   = entry["judgments"]

        # FIX #2: Skip proxy entries
        if not entry.get("has_real_context", False):
            n_proxy += 1
            continue

        # FIX #4: Resolusi total_relevant — urutan prioritas:
        #   1. Dihitung dari semua judgments nyata yang ada
        #   2. Dari field total_relevant_in_kb di JSON (jika diisi manual)
        #   3. Fallback hardcode per kategori
        n_rel = (
            total_rel_per_query.get(qid)
            or entry.get("total_relevant_in_kb")
            or FALLBACK_REL_MAP.get(cat, 2)
        )
        if not total_rel_per_query.get(qid):
            n_no_nrel += 1

        trace_id = trace_ids.get(qid, "")

        p     = precision_at_k(j, top_k)
        r     = recall_at_k(j, top_k, n_rel)
        f1_   = f1(p, r)
        mrr_  = mrr(j)
        ndcg_ = ndcg(j, top_k, n_rel)   # FIX #3: pakai n_rel

        rows.append({
            "query_id":        qid,
            "category":        cat,
            "total_relevant":  n_rel,
            f"P@{top_k}":     round(p,    4),
            f"R@{top_k}":     round(r,    4),
            "F1":              round(f1_,  4),
            "RR":              round(mrr_, 4),
            f"nDCG@{top_k}":  round(ndcg_,4),
        })

        if trace_id:
            lf_add_score(trace_id, f"ir_precision_at_{top_k}", p)
            lf_add_score(trace_id, f"ir_recall_at_{top_k}",    r)
            lf_add_score(trace_id, "ir_f1",                    f1_)
            lf_add_score(trace_id, "ir_mrr",                   mrr_)
            lf_add_score(trace_id, f"ir_ndcg_at_{top_k}",     ndcg_)

    # ── Simpan hasil ─────────────────────────────────────────
    df = pd.DataFrame(rows)
    Path("output").mkdir(exist_ok=True)
    out = f"output/ir_metrics_result_k{top_k}_fixed.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    agg_cols = [f"P@{top_k}", f"R@{top_k}", "F1", "RR", f"nDCG@{top_k}"]
    agg = df[agg_cols].agg(["mean", "median", "std"]).round(4)
    agg.to_csv(f"output/ir_metrics_aggregate_k{top_k}_fixed.csv", encoding="utf-8-sig")

    # ── Tampilkan hasil ──────────────────────────────────────
    print(f"\n{'='*55}")
    print(f" HASIL IR METRICS — k={top_k} (n={len(df)} queries valid)")
    print(f" [Proxy di-skip: {n_proxy} | Fallback n_rel: {n_no_nrel}]")
    print(f"{'='*55}")

    checks = [
        (f"Precision@{top_k}", f"P@{top_k}",      THRESHOLD_P),
        (f"Recall@{top_k}",    f"R@{top_k}",      THRESHOLD_R),
        ("F1-Score",            "F1",              THRESHOLD_F1),
        ("MRR",                 "RR",              THRESHOLD_MRR),
        (f"nDCG@{top_k}",      f"nDCG@{top_k}",  THRESHOLD_NDCG),
    ]

    table = []
    for name, col, thr in checks:
        val    = df[col].mean()
        median = df[col].median()
        std    = df[col].std()
        status = "✓ PASS" if val >= thr else "✗ FAIL"
        table.append([name, f"{val:.4f}", f"{median:.4f}", f"{std:.4f}", f"≥ {thr}", status])

    print(tabulate(
        table,
        headers=["Metrik", "Mean", "Median", "Std", "Target", "Status"],
        tablefmt="simple",
    ))

    print(f"\nPer kategori:")
    cat_df = (
        df.groupby("category")[agg_cols]
        .agg(["mean", "count"])
        .round(4)
    )
    # Tampilkan mean saja agar ringkas
    cat_mean = df.groupby("category")[agg_cols].mean().round(4)
    cat_count = df.groupby("category")["query_id"].count().rename("n")
    cat_summary = pd.concat([cat_count, cat_mean], axis=1)
    print(cat_summary.to_string())

    print(f"\nDistribusi total_relevant per query:")
    rel_dist = df["total_relevant"].value_counts().sort_index()
    for val, cnt in rel_dist.items():
        print(f"  n_rel={val}: {cnt} queries")

    print(f"\nDisimpan: {out}")
    print(f"{'='*55}")
    return df


def print_comparison(top_k=TOP_K_FINAL):
    """
    Opsional: bandingkan hasil fixed vs lama jika keduanya ada.
    """
    old_fp = f"output/ir_metrics_result_k{top_k}.csv"
    new_fp = f"output/ir_metrics_result_k{top_k}_fixed.csv"

    if not Path(old_fp).exists() or not Path(new_fp).exists():
        return

    old = pd.read_csv(old_fp)
    new = pd.read_csv(new_fp)

    cols = [f"P@{top_k}", f"R@{top_k}", "F1", "RR", f"nDCG@{top_k}"]
    cols_old = [c for c in cols if c in old.columns]
    cols_new = [c for c in cols if c in new.columns]

    print(f"\n{'='*55}")
    print(f" PERBANDINGAN: Lama vs Fixed (k={top_k})")
    print(f"{'='*55}")
    print(f"  n lama : {len(old)} | n fixed: {len(new)}")
    print()

    rows = []
    for col in cols_new:
        if col in cols_old:
            v_old = old[col].mean()
            v_new = new[col].mean()
            delta = v_new - v_old
            rows.append([col, f"{v_old:.4f}", f"{v_new:.4f}",
                         f"{delta:+.4f}", "↑" if delta > 0 else "↓"])

    print(tabulate(rows, headers=["Metrik", "Lama", "Fixed", "Delta", ""], tablefmt="simple"))
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=TOP_K_FINAL)
    parser.add_argument("--compare", action="store_true",
                        help="Tampilkan perbandingan hasil lama vs fixed")
    args = parser.parse_args()

    df = run_ir(top_k=args.k)

    if args.compare:
        print_comparison(top_k=args.k)