import json
from pathlib import Path

fp = "data/trace_results_k8.json"
if not Path(fp).exists():
    print(f"File {fp} tidak ditemukan!")
else:
    d = json.load(open(fp, encoding="utf-8"))
    results = d["results"]
    print(f"Total traces: {len(results)}")
    print()
    for r in results[:10]:
        print(f"[{r['query_id']}] {r['category']}")
        print(f"  Q: {r['question'][:80]}")
        print(f"  A: {r.get('answer','')[:80]}")
        print(f"  Contexts: {len(r.get('contexts',[]))}")
        print()