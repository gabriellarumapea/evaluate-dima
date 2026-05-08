"""
Cek endpoint Langfuse API yang benar untuk versi ini.
Jalankan: python cek_api_endpoint.py
"""
import os, base64, requests
from dotenv import load_dotenv
load_dotenv()

PK   = os.getenv("LANGFUSE_PUBLIC_KEY")
SK   = os.getenv("LANGFUSE_SECRET_KEY")
HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

def auth():
    token = base64.b64encode(f"{PK}:{SK}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

DATASET_NAME = "DIMA-PMB-ITERA-96queries"

# Coba beberapa endpoint yang mungkin
endpoints = [
    f"/api/public/datasets/{DATASET_NAME}/items",
    f"/api/public/datasets/{DATASET_NAME}",
    f"/api/public/datasets",
    f"/api/public/dataset-items?datasetName={DATASET_NAME}",
    f"/api/public/dataset-items",
]

for ep in endpoints:
    url = HOST + ep
    try:
        r = requests.get(url, headers=auth(), timeout=10)
        print(f"  [{r.status_code}] {ep}")
        if r.status_code == 200:
            data = r.json()
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            print(f"         → keys: {keys}")
    except Exception as e:
        print(f"  [ERR] {ep} → {e}")