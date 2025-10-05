# test_embed_qdrant.py
import os, time, traceback
print("TEST EMBED+QDRANT start")
os.environ.setdefault("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
try:
    from app.core.embedder import Embedder
    from app.services.qdrant_service import QdrantService
except Exception as e:
    print("Import error:", e)
    traceback.print_exc()
    raise SystemExit(1)

try:
    e = Embedder(mode="text")
    print("Embedder loaded:", e.model_name)
    text = "This is a short test sentence to embed."
    t0 = time.time()
    vec = e.embed_text(text)
    print("Embedding ok in {:.2f}s vector-len={}".format(time.time()-t0, len(vec)))
except Exception as ex:
    print("Embedding failed:", ex)
    traceback.print_exc()

try:
    qs = QdrantService()
    print("Qdrant client ok (host):", qs.host)
    # no need to upsert; just ping collections
    cols = qs._client.get_collections()
    print("Collections:", [c.name for c in cols.collections])
except Exception as ex:
    print("Qdrant test failed:", ex)
    traceback.print_exc()

print("TEST EMBED+QDRANT done")
