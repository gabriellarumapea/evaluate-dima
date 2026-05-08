"""
Generate relevance_judgments.json — versi final
Fix: total_relevant_in_kb tidak boleh lebih kecil dari jumlah
dokumen relevan yang ditemukan (recall tidak boleh > 1.0)
"""
import json, re
from pathlib import Path
from collections import Counter

with open("data/test_queries.json", encoding="utf-8") as f:
    queries = {q["id"]: q for q in json.load(f)}

skip_prefixes = ["You are an expert","Generate a question","Given a context",
                 "Given question","Please extract","Please identify"]

DOMAIN_KW = [
    "meninggal","wafat","almarhum","almarhumah","penghasilan","pekerjaan",
    "nik","ktp","kartu keluarga","upload","format","deadline","snbp","snbt",
    "kip","submit","error","gagal","login","tanggungan","saudara","ibu rumah",
    "wali","ijazah","nisn","domisili","registrasi","daftar ulang","formulir",
    "dim","data","isian","mahasiswa","pmb","itera","pengisian",
]

def tokenize(text):
    return set(re.findall(r'\b\w{3,}\b', text.lower()))

def score_context(content, ground_truth, question):
    content_words = tokenize(content)
    gt_words      = tokenize(ground_truth)
    q_words       = tokenize(question)
    domain_hits   = sum(1 for kw in DOMAIN_KW if kw in content.lower())

    if gt_words:
        gt_overlap = len(gt_words & content_words)
        ratio      = gt_overlap / len(gt_words)
    else:
        gt_overlap, ratio = 0, 0

    q_overlap = len(q_words & content_words)

    if ratio >= 0.15 or gt_overlap >= 3 or (domain_hits >= 3 and ratio >= 0.08):
        return 2
    elif ratio >= 0.06 or gt_overlap >= 1 or q_overlap >= 4 or domain_hits >= 3:
        return 1
    else:
        return 0

def score_by_answer(answer, ground_truth, question):
    ans_words = tokenize(answer)
    gt_words  = tokenize(ground_truth)
    if not gt_words:
        return 1
    ratio = len(gt_words & ans_words) / len(gt_words)
    if ratio >= 0.30: return 2
    elif ratio >= 0.10: return 1
    else: return 0

# Base total_relevant_in_kb per kategori
BASE_TOTAL_REL = {
    "Data Keluarga & Kondisi Khusus": 3,
    "Format Teknis & Aturan Upload":  2,
    "Prosedural & Kebijakan PMB":     3,
    "Kendala Teknis Sistem":          3,
}

judgments_all = []

for top_k in [5, 8, 10]:
    fp = f"data/trace_results_k{top_k}.json"
    if not Path(fp).exists():
        continue

    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    valid = []
    for r in data["results"]:
        if r.get("error"): continue
        q = r.get("question","")
        if any(q.startswith(p) for p in skip_prefixes): continue
        if not r.get("answer"): continue
        valid.append(r)

    with_ctx = sum(1 for r in valid if r.get("contexts"))
    print(f"k={top_k}: {len(valid)} traces, {with_ctx} dengan konteks asli")

    for r in valid:
        qid          = r.get("query_id","")
        question     = r.get("question","")
        answer       = r.get("answer","")
        contexts     = r.get("contexts",[])
        category     = r.get("category","")
        gt_data      = queries.get(qid, {})
        ground_truth = gt_data.get("ground_truth","")

        if contexts:
            judged = []
            for i, ctx in enumerate(contexts):
                content = ctx.get("content","")
                doc     = ctx.get("document","chunk")
                score   = score_context(content, ground_truth, question)
                judged.append({
                    "rank":            i + 1,
                    "document":        doc,
                    "content_preview": content[:120] + "...",
                    "score":           score,
                })
        else:
            ans_score = score_by_answer(answer, ground_truth, question)
            judged = []
            for i in range(top_k):
                s = ans_score if i == 0 else max(0, ans_score-1) if i == 1 else 0
                judged.append({
                    "rank":            i + 1,
                    "document":        f"proxy_{i+1}",
                    "content_preview": "[proxy]",
                    "score":           s,
                })

        # Hitung jumlah dokumen relevan yang ditemukan
        n_found_relevant = sum(1 for j in judged if j["score"] >= 1)

        # total_relevant_in_kb HARUS >= n_found_relevant agar recall <= 1.0
        base_total = BASE_TOTAL_REL.get(category, 3)
        total_rel  = max(base_total, n_found_relevant)

        judgments_all.append({
            "query_id":             qid,
            "category":             category,
            "question":             question,
            "top_k":                top_k,
            "answer_preview":       answer[:100],
            "ground_truth":         ground_truth,
            "judgments":            judged,
            "total_relevant_in_kb": total_rel,
            "has_real_context":     bool(contexts),
        })

Path("data").mkdir(exist_ok=True)
with open("data/relevance_judgments.json","w",encoding="utf-8") as f:
    json.dump(judgments_all, f, ensure_ascii=False, indent=2)

print(f"\n✓ {len(judgments_all)} judgments disimpan")
all_scores = [jc["score"] for j in judgments_all for jc in j["judgments"]]
if all_scores:
    cnt = Counter(all_scores); total = len(all_scores)
    print(f"  Score 2: {cnt[2]:4d} ({cnt[2]/total*100:.1f}%)")
    print(f"  Score 1: {cnt[1]:4d} ({cnt[1]/total*100:.1f}%)")
    print(f"  Score 0: {cnt[0]:4d} ({cnt[0]/total*100:.1f}%)")
    print(f"  Total relevan ≥1: {cnt[1]+cnt[2]:4d} ({(cnt[1]+cnt[2])/total*100:.1f}%)")
print(f"\n✓ Jalankan: python 05_ir_metrics.py --k 8")