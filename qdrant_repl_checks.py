# qdrant_repl_checks.py
# Usage: activate venv, run: python qdrant_repl_checks.py

import os, sys, pprint, uuid, traceback
from pprint import pprint

# Adjust these imports to match your project if paths differ:
try:
    from app.services.qdrant_service import QdrantService
except Exception as e:
    print("ERROR: couldn't import QdrantService:", e)
    raise

# Replace with your project's embedder import path.
# This script tries common names; edit if necessary.
Embedder = None
try:
    # common locations you may have used
    from app.services.embedder import Embedder as E1
    Embedder = E1
except Exception:
    try:
        from app.core.embedder import Embedder as E2
        Embedder = E2
    except Exception:
        Embedder = None

print("\n=== ENV and basic info ===")
print("EMBED_MODEL_TEXT:", os.environ.get("EMBED_MODEL_TEXT"))
svc = None
try:
    svc = QdrantService()
    print("QdrantService connected. named_vectors:", svc._named_vectors, "vector_size:", svc._vector_size)
except Exception:
    print("Failed to init QdrantService:")
    traceback.print_exc()
    sys.exit(2)

# Verify embedder exists and returns consistent dims
if Embedder is None:
    print("\nWARNING: Could not auto-import an Embedder class from app.*. Edit this script to import your embedder.")
else:
    try:
        emb = Embedder()
        sample = "hello world"
        v = emb.embed_text(sample) if hasattr(emb,"embed_text") else emb.embed(sample)
        print("Embed returned type:", type(v), "len:", len(v))
        if len(v) != svc._vector_size:
            print("ERROR: embed length mismatch: embed_len=%s svc._vector_size=%s" % (len(v), svc._vector_size))
        else:
            print("Embed length matches svc._vector_size ✅")
    except Exception:
        print("Embedder test failed:")
        traceback.print_exc()

# Upsert a test point (generated id)
vec = [0.01] * (svc._vector_size or 16)
point_id = str(uuid.uuid4())
chunk = {"id": None, "embedding": vec, "payload": {"text": "smoke test", "session_id": "smoke"}}

print("\n=== Attempting upsert of one test point ===")
try:
    ok, failed = svc.upsert_chunks([chunk], batch_size=1, wait=True)
    print("upsert ok:", ok, "failed:", failed)
    print("svc.last_payload_sent (upsert):")
    pprint(getattr(svc, "last_payload_sent", None))
    print("svc.last_server_response (upsert):")
    pprint(getattr(svc, "last_server_response", None))
except Exception:
    print("Upsert raised exception:")
    traceback.print_exc()

print("\n=== Attempting search_by_vector ===")
try:
    hits = svc.search_by_vector(vec, top_k=3, filter_meta={"must":[{"key":"session_id","match":{"value":"smoke"}}]}, with_payload=True)
    print("search hits:")
    pprint(hits)
    print("svc.last_payload_sent (search):")
    pprint(getattr(svc, "last_payload_sent", None))
    print("svc.last_server_response (search):")
    pprint(getattr(svc, "last_server_response", None))
except Exception:
    print("Search-by-vector raised exception:")
    traceback.print_exc()

print("\nDone. If any step printed an ERROR or mismatch, copy the printed blocks and paste them when asking for help.")
