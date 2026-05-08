"""
Script finalisasi: jalankan RAGAS k=5 dan k=10,
lalu response time semua k, lalu visualisasi.
Jalankan: python finalisasi.py
"""
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode

# RAGAS untuk k=5 dan k=10
run(f"{sys.executable} 04_ragas_eval.py --k 5")
run(f"{sys.executable} 04_ragas_eval.py --k 10")

# Response time
run(f"{sys.executable} 07_response_time.py --k 5")
run(f"{sys.executable} 07_response_time.py --k 8")
run(f"{sys.executable} 07_response_time.py --k 10")

# Ablation ulang (setelah semua RAGAS ada)
run(f"{sys.executable} 06_ablation_study.py")

# Visualisasi
run(f"{sys.executable} 08_visualisasi.py")

print("\n✓ SELESAI! Semua output ada di folder output/")