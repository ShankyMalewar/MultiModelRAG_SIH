# debug_steps.py
import os, sys, time, traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def now(): return time.strftime("%H:%M:%S")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

log("START debug_steps.py")

# 1) quick imports
try:
    log("IMPORT: app.core.embedder")
    from app.core.embedder import Embedder
    log("OK: Embedder imported")
except Exception:
    log("FAILED to import Embedder")
    traceback.print_exc()
    raise SystemExit(1)

try:
    log("IMPORT: app.extractors.ocr_handler")
    from app.extractors.ocr_handler import OCRHandler
    log("OK: OCRHandler imported")
except Exception:
    log("FAILED to import OCRHandler")
    traceback.print_exc()
    raise SystemExit(1)

try:
    log("IMPORT: app.core.chunker")
    from app.core.chunker import Chunker, chunk_paragraphs, chunk_text, estimate_tokens
    log("OK: Chunker + helpers imported")
except Exception:
    log("FAILED to import chunker")
    traceback.print_exc()
    raise SystemExit(1)

try:
    log("IMPORT: app.services.qdrant_service")
    from app.services.qdrant_service import QdrantService
    log("OK: QdrantService imported")
except Exception:
    log("FAILED to import QdrantService")
    traceback.print_exc()
    raise SystemExit(1)

# Optional ASR import (will not init model since we made it lazy)
try:
    log("IMPORT (optional): app.extractors.asr_handler")
    from app.extractors.asr_handler import ASRHandler
    log("OK: ASRHandler imported (lazy init)")
except Exception:
    log("ASRHandler import failed (non-fatal)")
    traceback.print_exc()

# 2) OCR test (small)
IMG = "data/vault/sample.jpg"
if os.path.exists(IMG):
    try:
        log(f"OCR: testing OCRHandler.extract_text on {IMG}")
        o = OCRHandler()
        t0 = time.time()
        txt = o.extract_text(IMG)
        dt = time.time() - t0
        log(f"OCR: done in {dt:.2f}s; len(text)={len(txt)}; snippet: {repr(txt[:200])}")
    except Exception:
        log("OCR: failed")
        traceback.print_exc()
else:
    log(f"OCR: image not found at {IMG}; skipping OCR test")

# 3) Embedder test
try:
    log("EMBED: creating Embedder(mode='text')")
    e = Embedder(mode="text")
    log(f"EMBED: model_name = {getattr(e, 'model_name', '<unknown>')}")
    sample = "Quick test sentence."
    t0 = time.time()
    vec = e.embed_text(sample)
    dt = time.time() - t0
    log(f"EMBED: embed ok in {dt:.2f}s vector_len={len(vec)}")
except Exception:
    log("EMBED: failed")
    traceback.print_exc()
    raise SystemExit(1)

# 4) Qdrant: ping, create collection, small upsert & search
try:
    log("QDRANT: init client")
    qs = QdrantService()
    log(f"QDRANT: host={qs.host} collection={qs.collection}")
    log("QDRANT: ensure collection exists (vector_size from embed)")
    try:
        qs.create_collection_if_not_exists(vector_size=len(vec))
        log("QDRANT: create_collection_if_not_exists returned")
    except Exception:
        log("QDRANT: create_collection_if_not_exists raised (continuing)")
    # Make a tiny Chunk-like payload and upsert one point
    from app.core.schema import ChunkDoc
    sample_id = "dbg-" + str(int(time.time()))
    chunk = ChunkDoc.create(text="debug point", filename="debug", modality="text", id=sample_id)
    chunk.embedding = vec
    log(f"QDRANT: upserting single debug point id={sample_id}")
    try:
        upserted, failed = qs.upsert_chunks([chunk], batch_size=1)
        log(f"QDRANT: upsert_chunks result upserted={upserted} failed={failed}")
    except Exception:
        log("QDRANT: upsert failed")
        traceback.print_exc()
    # Try a simple search by that vector
    try:
        res = qs.search_by_vector(vec, top_k=3)
        log(f"QDRANT: search_by_vector returned {len(res)} hits")
        if len(res):
            log("QDRANT: first hit payload keys: " + str(list(res[0].get('payload', {}).keys())))
    except Exception:
        log("QDRANT: search_by_vector failed")
        traceback.print_exc()
except Exception:
    log("QDRANT: failed fully")
    traceback.print_exc()
    raise SystemExit(1)

log("DEBUG STEPS COMPLETE")
