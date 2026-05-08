import json
d = json.load(open('data/trace_results_k8.json', encoding='utf-8'))
results = d['results']
with_ctx = [r for r in results if r.get('contexts')]
print(f'Contexts: {len(with_ctx)}/{len(results)}')
if with_ctx:
    print('Sample:', with_ctx[0]['contexts'][0]['content'][:100])
    print('Citation Dify: AKTIF ✓')
else:
    print('Citation Dify: BELUM AKTIF ✗')