"""
Jalankan ini dulu untuk cek versi Langfuse:
    python cek_langfuse_methods.py
"""
import langfuse
from dotenv import load_dotenv
load_dotenv()
from config_01 import get_langfuse_client

print(f"Versi Langfuse: {langfuse.__version__}")

lf = get_langfuse_client()

# Tampilkan semua method yang mengandung kata trace/fetch/get
methods = [m for m in dir(lf) if not m.startswith("_")]
trace_methods = [m for m in methods if any(k in m.lower() for k in ["trace", "fetch", "get", "list", "search"])]
print("\nMethod tersedia:")
for m in trace_methods:
    print(f"  {m}")