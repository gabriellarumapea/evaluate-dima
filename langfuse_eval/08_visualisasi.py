"""
============================================================
STEP 8 — VISUALISASI HASIL UNTUK LAPORAN TUGAS AKHIR
============================================================
Menghasilkan 6 grafik siap masuk laporan:
  1. Bar chart: IR Metrics vs threshold (k=8)
  2. Grouped bar: IR Metrics ablation study (k=5,8,10)
  3. Radar chart: RAGAS per kategori query
  4. Box plot: distribusi response time
  5. Bar chart: RAGAS metrics vs threshold
  6. Heatmap: skor per metrik per kategori (ringkasan eksekutif)

Jalankan setelah semua step sebelumnya selesai:
    python 08_visualisasi.py
============================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

matplotlib.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi":    150,
})

from dotenv import load_dotenv
load_dotenv()
from config_01 import (
    TOP_K_FINAL, TOP_K_VALUES,
    THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG,
    THRESHOLD_FAITH, THRESHOLD_ANS_R, THRESHOLD_CTX_P,
    CATEGORIES,
)

COLORS = {
    "pass":      "#3B82F6",   # biru
    "fail":      "#EF4444",   # merah
    "threshold": "#F59E0B",   # amber
    "k5":        "#6366F1",   # indigo
    "k8":        "#10B981",   # emerald
    "k10":       "#F97316",   # orange
    "bg":        "#F8FAFC",
}

Path("output/grafik").mkdir(parents=True, exist_ok=True)


# ── Helper ──────────────────────────────────────────────────

def load_ir_agg(k):
    f = f"output/ir_metrics_aggregate_k{k}.csv"
    if not Path(f).exists():
        return None
    return pd.read_csv(f, index_col=0)

def load_ragas(k):
    f = f"output/ragas_result_k{k}.csv"
    if not Path(f).exists():
        return None
    return pd.read_csv(f)

def load_response_time(k):
    f = f"output/response_time_result_k{k}.csv"
    if not Path(f).exists():
        return None
    return pd.read_csv(f)


# ── Grafik 1: IR Metrics vs Threshold (k=8) ────────────────

def plot_ir_metrics_bar(k=TOP_K_FINAL):
    agg = load_ir_agg(k)
    if agg is None:
        print(f"  ⚠ Skip grafik 1: tidak ada data IR k={k}")
        return

    metrics = {
        f"Precision@{k}": (agg.loc["mean", f"P@{k}"],   THRESHOLD_P),
        f"Recall@{k}":    (agg.loc["mean", f"R@{k}"],   THRESHOLD_R),
        "F1-Score":       (agg.loc["mean", "F1"],         THRESHOLD_F1),
        "MRR":            (agg.loc["mean", "RR"],          THRESHOLD_MRR),
        f"nDCG@{k}":     (agg.loc["mean", f"nDCG@{k}"], THRESHOLD_NDCG),
    }

    labels = list(metrics.keys())
    values = [v[0] for v in metrics.values()]
    thresholds = [v[1] for v in metrics.values()]
    bar_colors = [COLORS["pass"] if v >= t else COLORS["fail"]
                  for v, t in zip(values, thresholds)]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(labels, values, color=bar_colors, height=0.55, zorder=3)

    # Garis threshold per metrik
    for i, (t, label) in enumerate(zip(thresholds, labels)):
        ax.vlines(t, i - 0.4, i + 0.4, colors=COLORS["threshold"],
                  linewidths=2, linestyles="--", zorder=4, label=None)

    # Label nilai
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    # Legend
    pass_patch  = mpatches.Patch(color=COLORS["pass"],      label="Memenuhi target")
    fail_patch  = mpatches.Patch(color=COLORS["fail"],      label="Di bawah target")
    thr_patch   = mpatches.Patch(color=COLORS["threshold"], label="Garis threshold")
    ax.legend(handles=[pass_patch, fail_patch, thr_patch],
              loc="lower right", fontsize=9)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Skor")
    ax.set_title(f"IR Metrics Chatbot DIMA (k={k}, n=96 queries)", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.invert_yaxis()

    out = f"output/grafik/01_ir_metrics_k{k}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 1 disimpan: {out}")


# ── Grafik 2: Ablation Study grouped bar ───────────────────

def plot_ablation_grouped():
    data = {}
    for k in TOP_K_VALUES:
        agg = load_ir_agg(k)
        if agg is None:
            print(f"  ⚠ Skip k={k} di ablation chart")
            continue
        data[f"k={k}"] = {
            f"P@K":   agg.loc["mean", f"P@{k}"],
            f"R@K":   agg.loc["mean", f"R@{k}"],
            "F1":     agg.loc["mean", "F1"],
            "MRR":    agg.loc["mean", "RR"],
            f"nDCG@K": agg.loc["mean", f"nDCG@{k}"],
        }

    if not data:
        print("  ⚠ Skip grafik 2: tidak ada data ablation")
        return

    df_abl = pd.DataFrame(data).T
    metric_labels = list(df_abl.columns)

    x     = np.arange(len(metric_labels))
    width = 0.22
    colors_abl = [COLORS["k5"], COLORS["k8"], COLORS["k10"]]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for i, (label, row) in enumerate(df_abl.iterrows()):
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, row.values, width, label=label,
                        color=colors_abl[i], alpha=0.88, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    # Garis threshold
    thresholds_val = [THRESHOLD_P, THRESHOLD_R, THRESHOLD_F1, THRESHOLD_MRR, THRESHOLD_NDCG]
    for xi, t in zip(x, thresholds_val):
        ax.hlines(t, xi - 0.38, xi + 0.38, colors=COLORS["threshold"],
                  linewidths=1.5, linestyles="--", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Skor")
    ax.set_title("Ablation Study: Perbandingan k=5, k=8, k=10", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    out = "output/grafik/02_ablation_study.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 2 disimpan: {out}")


# ── Grafik 3: RAGAS per kategori (radar chart) ─────────────

def plot_ragas_radar(k=TOP_K_FINAL):
    ragas_df = load_ragas(k)
    if ragas_df is None:
        print(f"  ⚠ Skip grafik 3: tidak ada data RAGAS k={k}")
        return

    cat_means = ragas_df.groupby("category")[
        ["faithfulness", "answer_relevancy", "context_precision"]
    ].mean()

    categories_radar = list(cat_means.index)
    metrics_radar    = ["faithfulness", "answer_relevancy", "context_precision"]
    metric_labels    = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision"]

    N      = len(metrics_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6),
                           subplot_kw=dict(polar=True),
                           facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    palette = plt.cm.Set2(np.linspace(0, 1, len(categories_radar)))
    for i, (cat, row) in enumerate(cat_means.iterrows()):
        values_r = list(row[metrics_radar]) + [row[metrics_radar[0]]]
        ax.plot(angles, values_r, "o-", linewidth=1.8, color=palette[i],
                label=cat[:30])
        ax.fill(angles, values_r, alpha=0.08, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=8)
    ax.set_title(f"RAGAS per Kategori Query (k={k})",
                 pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    out = f"output/grafik/03_ragas_radar_k{k}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 3 disimpan: {out}")


# ── Grafik 4: Box plot response time ───────────────────────

def plot_response_time_boxplot(k=TOP_K_FINAL):
    rt_df = load_response_time(k)
    if rt_df is None:
        print(f"  ⚠ Skip grafik 4: tidak ada data response time k={k}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=COLORS["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(COLORS["bg"])

    # Box plot per kategori
    cats   = rt_df["category"].unique()
    data_per_cat = [rt_df[rt_df["category"] == c]["latency_ms"].values for c in cats]
    cat_short = [c.split("&")[0].strip()[:20] for c in cats]

    bp = ax1.boxplot(data_per_cat, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    palette = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax1.axhline(7000, color=COLORS["threshold"], linestyle="--",
                linewidth=1.5, label="Target 7.000 ms")
    ax1.set_xticklabels(cat_short, rotation=15, ha="right", fontsize=8.5)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Distribusi Response Time per Kategori")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Histogram keseluruhan
    ax2.hist(rt_df["latency_ms"], bins=20, color=COLORS["pass"],
             edgecolor="white", alpha=0.85)
    ax2.axvline(7000, color=COLORS["threshold"], linestyle="--",
                linewidth=2, label=f"Target 7.000 ms")
    ax2.axvline(rt_df["latency_ms"].mean(), color="#1E3A5F", linestyle="-",
                linewidth=2, label=f"Mean {rt_df['latency_ms'].mean():.0f} ms")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Frekuensi")
    ax2.set_title("Distribusi Response Time Keseluruhan")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    out = f"output/grafik/04_response_time_k{k}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 4 disimpan: {out}")


# ── Grafik 5: RAGAS Metrics vs Threshold ───────────────────

def plot_ragas_bar(k=TOP_K_FINAL):
    ragas_df = load_ragas(k)
    if ragas_df is None:
        print(f"  ⚠ Skip grafik 5: tidak ada data RAGAS k={k}")
        return

    metrics = {
        "Faithfulness":      (ragas_df["faithfulness"].mean(),      THRESHOLD_FAITH),
        "Answer Relevancy":  (ragas_df["answer_relevancy"].mean(),  THRESHOLD_ANS_R),
        "Context Precision": (ragas_df["context_precision"].mean(), THRESHOLD_CTX_P),
    }

    labels     = list(metrics.keys())
    values     = [v[0] for v in metrics.values()]
    thresholds = [v[1] for v in metrics.values()]
    bar_colors = [COLORS["pass"] if v >= t else COLORS["fail"]
                  for v, t in zip(values, thresholds)]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.bar(labels, values, color=bar_colors, width=0.45, zorder=3)
    for i, (t, label) in enumerate(zip(thresholds, labels)):
        ax.hlines(t, i - 0.26, i + 0.26, colors=COLORS["threshold"],
                  linewidths=2, linestyles="--", zorder=4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    pass_p  = mpatches.Patch(color=COLORS["pass"],      label="Memenuhi target")
    fail_p  = mpatches.Patch(color=COLORS["fail"],      label="Di bawah target")
    thr_p   = mpatches.Patch(color=COLORS["threshold"], label="Garis threshold")
    ax.legend(handles=[pass_p, fail_p, thr_p], fontsize=9)

    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Skor")
    ax.set_title(f"RAGAS Evaluation Chatbot DIMA (k={k})", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    out = f"output/grafik/05_ragas_bar_k{k}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 5 disimpan: {out}")


# ── Grafik 6: Heatmap ringkasan semua metrik per kategori ──

def plot_summary_heatmap(k=TOP_K_FINAL):
    ragas_df = load_ragas(k)
    ir_df    = None
    ir_file  = f"output/ir_metrics_result_k{k}.csv"
    if Path(ir_file).exists():
        ir_df = pd.read_csv(ir_file)

    if ragas_df is None and ir_df is None:
        print(f"  ⚠ Skip grafik 6: tidak ada data")
        return

    rows = {}
    cats = list(CATEGORIES.keys())

    for cat in cats:
        row = {}
        if ragas_df is not None:
            sub = ragas_df[ragas_df["category"] == cat]
            if not sub.empty:
                row["Faithfulness"]      = sub["faithfulness"].mean()
                row["Answer Relevancy"]  = sub["answer_relevancy"].mean()
                row["Context Precision"] = sub["context_precision"].mean()
        if ir_df is not None:
            sub = ir_df[ir_df["category"] == cat]
            if not sub.empty:
                row[f"P@{k}"]    = sub[f"P@{k}"].mean()
                row[f"nDCG@{k}"] = sub[f"nDCG@{k}"].mean()
                row["MRR"]       = sub["RR"].mean()
        rows[cat[:25]] = row

    heatmap_df = pd.DataFrame(rows).T.round(3)
    if heatmap_df.empty:
        print("  ⚠ Skip grafik 6: data kosong")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    sns.heatmap(heatmap_df.astype(float), annot=True, fmt=".3f",
                cmap="YlGn", vmin=0.5, vmax=1.0,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title(f"Ringkasan Skor per Kategori Query (k={k})", pad=12,
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    out = f"output/grafik/06_heatmap_summary_k{k}.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik 6 disimpan: {out}")


# ── Main ────────────────────────────────────────────────────

def run_all_visualizations(k=TOP_K_FINAL):
    print(f"\n{'='*55}")
    print(f" STEP 8 — Visualisasi Hasil (k={k})")
    print(f"{'='*55}\n")

    print("Membuat grafik...")
    plot_ir_metrics_bar(k)
    plot_ablation_grouped()
    plot_ragas_bar(k)
    plot_ragas_radar(k)
    plot_response_time_boxplot(k)
    plot_summary_heatmap(k)

    print(f"\n{'='*55}")
    print(f" Semua grafik tersimpan di: output/grafik/")
    print(f"{'='*55}")
    print("""
 Daftar grafik:
   01_ir_metrics_k8.png        → masukkan di Bab IV bagian IR Metrics
   02_ablation_study.png       → masukkan di Bab IV bagian Ablation Study
   03_ragas_radar_k8.png       → masukkan di Bab IV bagian RAGAS per kategori
   04_response_time_k8.png     → masukkan di Bab IV bagian Non-Fungsional
   05_ragas_bar_k8.png         → masukkan di Bab IV bagian RAGAS overview
   06_heatmap_summary_k8.png   → masukkan di Bab IV bagian ringkasan/kesimpulan
""")


if __name__ == "__main__":
    run_all_visualizations(k=TOP_K_FINAL)
