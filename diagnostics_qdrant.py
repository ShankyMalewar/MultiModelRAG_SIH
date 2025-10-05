# diagnostics_qdrant.py
import os, json, requests, sys
host = os.getenv("QDRANT_HOST", "http://localhost:6333").rstrip("/")
coll = os.getenv("QDRANT_COLLECTION", "asklyne_collection")
headers = {"Content-Type": "application/json"}
api_key = os.getenv("QDRANT_API_KEY", "")
if api_key:
    headers["api-key"] = api_key

def pretty(r):
    print("STATUS:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2)[:8000])
    except Exception:
        print("NON-JSON BODY:", r.text[:2000])

print("\n== Qdrant health ==")
try:
    r = requests.get(f"{host}/health", headers=headers, timeout=6)
    pretty(r)
except Exception as e:
    print("health check failed:", e)
    sys.exit(1)

print("\n== Collection info ==")
try:
    r = requests.get(f"{host}/collections/{coll}", headers=headers, timeout=6)
    pretty(r)
except Exception as e:
    print("collection info failed:", e)

print("\n== Collections list ==")
try:
    r = requests.get(f"{host}/collections", headers=headers, timeout=6)
    pretty(r)
except Exception as e:
    print("collections list failed:", e)

print("\n== Scroll first 10 points (if any) ==")
try:
    body = {"limit": 10}
    r = requests.post(f"{host}/collections/{coll}/points/scroll", headers=headers, json=body, timeout=10)
    pretty(r)
    pts = r.json().get("result", {}).get("points") or r.json().get("result") or []
    if isinstance(pts, list) and pts:
        print("\nSample point ids:")
        for p in pts:
            # each p might be dict including 'id' or 'payload'
            pid = p.get("id") if isinstance(p, dict) else None
            print(" -", pid)
except Exception as e:
    print("scroll failed:", e)

# If we found at least one vector, pick a vector to test search. Otherwise create a dummy vector.
test_vector = None
if 'pts' in locals() and pts:
    # try to extract vector from first point's stored vector(s)
    first = pts[0]
    if isinstance(first, dict):
        # try payload->embedding or vector fields
        if "vector" in first:
            test_vector = first["vector"]
        elif "vectors" in first and isinstance(first["vectors"], dict):
            # pick first named vector available
            v = list(first["vectors"].values())[0]
            test_vector = v
        else:
            # try payload fields
            payload = first.get("payload", {})
            for k,v in payload.items():
                if isinstance(v, list) and len(v) > 10 and all(isinstance(x,(int,float)) for x in v):
                    test_vector = v
                    break

if test_vector is None:
    # fall back to a random but proper-length vector (384 dims likely)
    test_vector = [0.01]*384
    print("\nNo vector found in stored points; using a dummy vector of len", len(test_vector))

print("\n== Try searches (3 shapes) ==")
shapes = {
    "named_vector": {"vector": {"default": test_vector}, "limit": 3, "with_payload": True},
    "legacy_vector_field": {"vector": test_vector, "limit": 3, "with_payload": True},
    "query_vector_tuple_style": {"query_vector": ("default", test_vector), "limit": 3, "with_payload": True},
}
for name, body in shapes.items():
    print("\n-- Shape:", name)
    try:
        r = requests.post(f"{host}/collections/{coll}/points/search", headers=headers, json=body, timeout=10)
        print("REQUEST BODY (first 300 chars):", json.dumps(body)[:300])
        pretty(r)
    except Exception as e:
        print("search attempt failed:", e)

print("\n== Try fetch one point by id if we saw ids above ==")
if 'pts' in locals() and pts:
    pid = None
    for p in pts:
        if isinstance(p, dict) and p.get("id"):
            pid = p.get("id"); break
    if pid:
        try:
            r = requests.get(f"{host}/collections/{coll}/points/{pid}", headers=headers, timeout=6)
            print("Fetch id", pid)
            pretty(r)
        except Exception as e:
            print("fetch point failed:", e)
else:
    print("No ids to fetch.")
