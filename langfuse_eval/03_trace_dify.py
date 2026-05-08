"""
============================================================
STEP 3 — JALANKAN QUERY KE DIFY + LOG TRACE KE LANGFUSE
         (Langfuse v4.x — endpoint fix)
============================================================
Jalankan:
    python 03_trace_dify.py --k 5
    python 03_trace_dify.py --k 8
    python 03_trace_dify.py --k 10
============================================================
"""

import argparse, json, os, time, base64, uuid, requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    DIFY_API_KEY, DIFY_BASE_URL, TOP_K_FINAL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    DATASET_NAME,
)

# ── Langfuse REST helpers ───────────────────────────────────

def lf_auth():
    token = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

def lf_get(path, params=None):
    r = requests.get(f"{LANGFUSE_HOST}{path}", headers=lf_auth(), params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def lf_post(path, payload):
    r = requests.post(f"{LANGFUSE_HOST}{path}", headers=lf_auth(), json=payload, timeout=15)
    r.raise_for_status()
    return r

def lf_get_dataset_items(dataset_name) -> list:
    """Ambil semua items via /api/public/dataset-items?datasetName=..."""
    items = []
    page  = 1
    while True:
        data  = lf_get("/api/public/dataset-items",
                       {"datasetName": dataset_name, "page": page, "limit": 100})
        batch = data.get("data", [])
        if not batch:
            break
        items.extend(batch)
        if page >= data.get("meta", {}).get("totalPages", 1):
            break
        page += 1
    return items

def lf_create_trace(name, input_data, tags, metadata, session_id) -> str:
    trace_id = str(uuid.uuid4())
    lf_post("/api/public/traces", {
        "id":        trace_id,
        "name":      name,
        "input":     input_data,
        "tags":      tags,
        "metadata":  metadata,
        "sessionId": session_id,
    })
    return trace_id

def lf_update_trace(trace_id, output, metadata):
    lf_post("/api/public/traces", {
        "id":       trace_id,
        "output":   output,
        "metadata": metadata,
    })

def lf_add_score(trace_id, name, value, comment=""):
    lf_post("/api/public/scores", {
        "traceId": trace_id,
        "name":    name,
        "value":   float(value),
        "comment": comment,
    })

def lf_link_run(item_id, trace_id, run_name):
    """Link trace ke dataset item run."""
    try:
        lf_post(f"/api/public/dataset-items/{item_id}/runs", {
            "runName": run_name,
            "traceId": trace_id,
        })
    except Exception:
        pass  # link gagal tidak kritis

# ── Dify API ────────────────────────────────────────────────

def call_dify(question: str) -> dict:
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs":          {},
        "query":           question,
        "response_mode":   "blocking",
        "conversation_id": "",
        "user":            "evaluasi-ta-dima",
    }
    t0   = time.time()
    resp = requests.post(f"{DIFY_BASE_URL}/chat-messages",
                         headers=headers, json=payload, timeout=60)
    lat  = (time.time() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    contexts = []
    for src in data.get("metadata", {}).get("retriever_resources", []):
        contexts.append({
            "content":  src.get("content", ""),
            "score":    src.get("score", 0),
            "document": src.get("document_name", ""),
        })

    return {
        "answer":     data.get("answer", ""),
        "contexts":   contexts,
        "latency_ms": round(lat, 1),
        "usage":      data.get("metadata", {}).get("usage", {}),
    }

# ── Trace single query ──────────────────────────────────────

def trace_query(item: dict, top_k: int, run_name: str) -> dict:
    inp      = item.get("input", {})
    question = inp.get("question", "") if isinstance(inp, dict) else str(inp)
    category = inp.get("category", "") if isinstance(inp, dict) else ""
    item_id  = item.get("id", "")

    result = {
        "query_id": item.get("metadata", {}).get("id", item_id) if isinstance(item.get("metadata"), dict) else item_id,
        "question": question,
        "category": category,
        "top_k":    top_k,
        "error":    None,
    }

    try:
        trace_id = lf_create_trace(
            name       = f"dima-eval-k{top_k}",
            input_data = {"question": question, "category": category},
            tags       = [f"k={top_k}", category, run_name],
            metadata   = {"item_id": item_id, "top_k": top_k},
            session_id = run_name,
        )

        dify = call_dify(question)

        lf_update_trace(
            trace_id,
            output   = {"answer": dify["answer"], "n_contexts": len(dify["contexts"])},
            metadata = {"latency_ms": dify["latency_ms"]},
        )

        lf_add_score(trace_id, "response_time_ms", dify["latency_ms"],
                     "PASS" if dify["latency_ms"] < 7000 else "FAIL")

        lf_link_run(item_id, trace_id, run_name)

        result.update({
            "trace_id":     trace_id,
            "answer":       dify["answer"],
            "contexts":     dify["contexts"],
            "latency_ms":   dify["latency_ms"],
            "n_contexts":   len(dify["contexts"]),
            "ground_truth": item.get("expectedOutput") or "",
        })

    except Exception as e:
        result["error"] = str(e)
        print(f"\n  ✗ {item_id}: {e}")

    return result

# ── Main ────────────────────────────────────────────────────

def run_traces(top_k: int = TOP_K_FINAL):
    run_name = f"eval-k{top_k}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    print(f"\n{'='*55}")
    print(f" STEP 3 — Tracing ke Dify + Langfuse (k={top_k})")
    print(f" Run: {run_name}")
    print(f"{'='*55}\n")

    print(f"Mengambil items dari dataset '{DATASET_NAME}'...")
    items = lf_get_dataset_items(DATASET_NAME)
    print(f"✓ {len(items)} items siap\n")

    results, success, failed, total_lat = [], 0, 0, 0.0

    for item in tqdm(items, desc=f"k={top_k}"):
        r = trace_query(item, top_k, run_name)
        results.append(r)
        if r["error"]:
            failed += 1
        else:
            success   += 1
            total_lat += r["latency_ms"]
        time.sleep(0.3)

    Path("data").mkdir(exist_ok=True)
    out = f"data/trace_results_k{top_k}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "run_name":       run_name,
            "top_k":          top_k,
            "timestamp":      datetime.now().isoformat(),
            "total":          len(items),
            "success":        success,
            "failed":         failed,
            "avg_latency_ms": round(total_lat / max(success, 1), 1),
            "results":        results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f" SELESAI — k={top_k}")
    print(f"  Berhasil : {success}/{len(items)}")
    print(f"  Gagal    : {failed}")
    if success:
        avg = total_lat / success
        print(f"  Avg lat  : {avg:.0f} ms ({avg/1000:.1f}s) {'✓ PASS' if avg < 7000 else '✗ FAIL'}")
    print(f"  Disimpan : {out}")
    if top_k == 5:
        print(f"\n→ Berikutnya: set k=8 di Dify, lalu: python 03_trace_dify.py --k 8")
    elif top_k == 8:
        print(f"\n→ Berikutnya: set k=10 di Dify, lalu: python 03_trace_dify.py --k 10")
    elif top_k == 10:
        print(f"\n→ Semua ablation selesai! Lanjut: python 04_ragas_eval.py --k 8")
    print(f"{'='*55}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K_FINAL)
    args = parser.parse_args()

    print(f"\n⚠ Pastikan Top-K di Dify = {args.k}")
    print("  Dify → App → Settings → Retrieval → Top K")
    input("Tekan Enter jika sudah...\n")

    run_traces(top_k=args.k)