"""
============================================================
STEP 4 — RAGAS dengan fallback otomatis:
  1. Coba GPT-4o-mini (OpenAI) dulu
  2. Jika quota habis → fallback ke Ollama llama3.2
============================================================
Jalankan: python 04_ragas_eval.py --k 8
============================================================
"""
import argparse, json, re, time, base64, requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
from config_01 import (
    TOP_K_FINAL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST, THRESHOLD_FAITH, THRESHOLD_ANS_R, THRESHOLD_CTX_P,
    OPENAI_API_KEY,
)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
USE_OPENAI   = False  # akan di-set otomatis

# ── Langfuse ────────────────────────────────────────────────
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

# ── Cek ketersediaan evaluator ──────────────────────────────
def check_openai():
    if not OPENAI_API_KEY:
        return False
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "Reply: 0.9"}],
                  "max_tokens": 5},
            timeout=15
        )
        if r.status_code == 200:
            return True
        print(f"  OpenAI status {r.status_code}: {r.json().get('error',{}).get('message','')[:80]}")
        return False
    except Exception as e:
        print(f"  OpenAI error: {e}")
        return False

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return OLLAMA_MODEL in models
    except Exception:
        return False

# ── Evaluasi via OpenAI ─────────────────────────────────────
def eval_openai(question, answer, contexts, ground_truth) -> dict:
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OPENAI_API_KEY))
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
    ]
    ctx = contexts if contexts else [ground_truth or "Tidak ada konteks."]
    ds = HFDataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [ctx], "ground_truth": [ground_truth or ""],
    })
    for attempt in range(3):
        try:
            result = evaluate(dataset=ds, metrics=metrics,
                              llm=llm, embeddings=emb, raise_exceptions=False)
            s = result.to_pandas().iloc[0]
            return {
                "faithfulness":      float(s.get("faithfulness")      or 0),
                "answer_relevancy":  float(s.get("answer_relevancy")  or 0),
                "context_precision": float(s.get("context_precision") or 0),
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"\n  ⚠ Rate limit, tunggu {wait}s...")
                time.sleep(wait)
            else:
                break
    return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

# ── Evaluasi via Ollama HTTP (fallback) ─────────────────────
def ask_ollama(prompt, timeout=120):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0, "num_predict": 15},
        }, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        return f"ERROR:{e}"

def parse_score(text):
    nums = re.findall(r"\b(0\.\d+|1\.0+|0|1)\b", text or "")
    if nums:
        try:
            return round(min(max(float(nums[0]), 0.0), 1.0), 4)
        except Exception:
            pass
    return None

def eval_ollama(question, answer, contexts, ground_truth) -> dict:
    ctx = contexts[:2] if contexts else [ground_truth or "Tidak ada konteks."]
    ctx_text = " | ".join(c[:200] for c in ctx)

    f = parse_score(ask_ollama(
        f"Context: {ctx_text[:400]}\nAnswer: {answer[:300]}\n"
        f"Is the answer faithful to the context? Reply ONE number 0.0-1.0:"))
    ar = parse_score(ask_ollama(
        f"Question: {question[:200]}\nAnswer: {answer[:300]}\n"
        f"How relevant is the answer? Reply ONE number 0.0-1.0:"))
    cp = parse_score(ask_ollama(
        f"Question: {question[:200]}\nContexts: {ctx_text[:400]}\n"
        f"What proportion of contexts are relevant? Reply ONE number 0.0-1.0:"))
    return {"faithfulness": f, "answer_relevancy": ar, "context_precision": cp}

# ── Load traces ─────────────────────────────────────────────
def load_traces(top_k):
    fp = f"data/trace_results_k{top_k}.json"
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    skip = ["You are an expert","Generate a question","Given a context",
            "Given question","Please extract","Please identify"]
    valid = [r for r in data["results"]
             if not r.get("error") and r.get("answer")
             and not any(r.get("question","").startswith(p) for p in skip)]
    print(f"✓ {len(valid)} valid traces")
    # Cek apakah contexts ada (Citation aktif)
    with_ctx = sum(1 for r in valid if r.get("contexts"))
    print(f"✓ Traces dengan contexts: {with_ctx}/{len(valid)}", 
          "← Citation Dify AKTIF ✓" if with_ctx > 0 else "← Citation belum aktif, pakai proxy")
    return valid

# ── Main ────────────────────────────────────────────────────
def run_ragas(top_k=TOP_K_FINAL):
    global USE_OPENAI

    print(f"\n{'='*55}")
    print(f" STEP 4 — RAGAS Evaluation (k={top_k})")
    print(f"{'='*55}\n")

    # Tentukan evaluator
    print("Cek ketersediaan evaluator...")
    if check_openai():
        USE_OPENAI = True
        print("✓ Menggunakan GPT-4o-mini (OpenAI)\n")
    elif check_ollama():
        USE_OPENAI = False
        print("⚠ OpenAI tidak tersedia → fallback ke Ollama llama3.2\n")
    else:
        print("✗ Tidak ada evaluator tersedia!")
        print("  Pastikan Ollama aktif atau saldo OpenAI cukup.")
        return None

    traces = load_traces(top_k)

    # Auto-resume
    out_file  = f"output/ragas_result_k{top_k}.csv"
    done_ids  = set()
    rows_done = []
    if Path(out_file).exists():
        df_ex    = pd.read_csv(out_file)
        df_valid = df_ex.dropna(subset=["faithfulness"])
        done_ids  = set(df_valid["query_id"].tolist())
        rows_done = df_valid.to_dict("records")
        if done_ids:
            print(f"✓ Resume: {len(done_ids)} queries sudah selesai\n")

    todo = [r for r in traces if r.get("query_id") not in done_ids]
    eval_fn = eval_openai if USE_OPENAI else eval_ollama
    mode    = "GPT-4o-mini" if USE_OPENAI else "Ollama llama3.2"
    print(f"Sisa: {len(todo)} queries | Evaluator: {mode}\n")

    rows_new = []
    for i, r in enumerate(tqdm(todo, desc=f"RAGAS ({mode})")):
        ctx_texts = [c["content"] for c in r.get("contexts",[]) if c.get("content")]
        scores = eval_fn(
            question     = r["question"],
            answer       = r.get("answer",""),
            contexts     = ctx_texts,
            ground_truth = r.get("ground_truth",""),
        )
        row = {
            "query_id":          r["query_id"],
            "category":          r["category"],
            "question":          r["question"][:100],
            "trace_id":          r.get("trace_id",""),
            "evaluator":         mode,
            "faithfulness":      scores["faithfulness"],
            "answer_relevancy":  scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
        }
        rows_new.append(row)

        if r.get("trace_id") and scores["faithfulness"] is not None:
            lf_add_score(r["trace_id"], "ragas_faithfulness",      scores["faithfulness"])
            lf_add_score(r["trace_id"], "ragas_answer_relevancy",  scores["answer_relevancy"])
            lf_add_score(r["trace_id"], "ragas_context_precision", scores["context_precision"])

        if (i+1) % 10 == 0:
            Path("output").mkdir(exist_ok=True)
            pd.DataFrame(rows_done + rows_new).to_csv(
                out_file, index=False, encoding="utf-8-sig")
            tqdm.write(f"  💾 Saved ({len(rows_done)+len(rows_new)} queries)")

        if USE_OPENAI:
            time.sleep(1)

    Path("output").mkdir(exist_ok=True)
    df_final = pd.DataFrame(rows_done + rows_new)
    df_final.to_csv(out_file, index=False, encoding="utf-8-sig")

    valid = df_final.dropna(subset=["faithfulness"])

    print(f"\n{'='*55}")
    print(f" HASIL RAGAS — k={top_k} ({len(valid)}/{len(df_final)} berhasil)")
    print(f" Evaluator: {mode}")
    print(f"{'='*55}")

    if len(valid) > 0:
        for name, col, thr in [
            ("Faithfulness",      "faithfulness",      THRESHOLD_FAITH),
            ("Answer Relevancy",  "answer_relevancy",  THRESHOLD_ANS_R),
            ("Context Precision", "context_precision", THRESHOLD_CTX_P),
        ]:
            val = valid[col].mean()
            print(f"  {'✓ PASS' if val>=thr else '✗ FAIL'}  {name:<22}: {val:.4f}  (target ≥ {thr})")

        print(f"\nPer kategori:")
        cat = valid.groupby("category")[
            ["faithfulness","answer_relevancy","context_precision"]
        ].mean().round(4)
        print(cat.to_string())

    print(f"\nDisimpan: {out_file}")
    print(f"{'='*55}")
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K_FINAL)
    args = parser.parse_args()
    run_ragas(top_k=args.k)