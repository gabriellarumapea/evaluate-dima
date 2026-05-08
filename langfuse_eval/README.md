# Panduan Evaluasi Chatbot DIMA — Langfuse + RAGAS
## Tugas Akhir: Gabriella Natalya Rumapea — 122140056

---

## Urutan Eksekusi

```
00_setup.sh              → Install semua dependency
01_config.py             → Konfigurasi API keys & parameter
02_build_dataset.py      → Upload 96 test queries ke Langfuse Dataset
03_trace_dify.py         → Jalankan query ke Dify + log trace ke Langfuse
04_ragas_eval.py         → Hitung Faithfulness, Answer Relevancy, Context Precision
05_ir_metrics.py         → Hitung Precision, Recall, F1, MRR, nDCG (manual judgment)
06_ablation_study.py     → Bandingkan k=5, k=8, k=10
07_response_time.py      → Analisis response time dari traces Langfuse
08_visualisasi.py        → Buat semua grafik untuk laporan TA
```

---

## Struktur Folder

```
langfuse_eval/
├── README.md
├── 00_setup.sh
├── 01_config.py
├── 02_build_dataset.py
├── 03_trace_dify.py
├── 04_ragas_eval.py
├── 05_ir_metrics.py
├── 06_ablation_study.py
├── 07_response_time.py
├── 08_visualisasi.py
├── data/
│   ├── test_queries.json       ← 96 test queries (kamu isi)
│   └── relevance_judgments.json ← hasil penilaian relevansi manual
└── output/
    ├── ir_metrics_result.csv
    ├── ragas_result.csv
    ├── ablation_result.csv
    └── grafik/
```
