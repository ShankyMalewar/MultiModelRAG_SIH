import requests, json

HOST = "http://localhost:6333"
COL = "asklyne_collection"

body = {
    "vectors": {
        "default": {"size": 1024, "distance": "Cosine"}
    }
}

url = f"{HOST}/collections/{COL}"
r = requests.put(url, json=body, timeout=10)
print(r.status_code, r.text)
