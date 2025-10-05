# test_imports.py
"""
Simple import test harness to confirm schema + ingestors import cleanly.
Run: python test_imports.py
"""

import traceback

print("=== Testing import of app.core.schema ===")
try:
    import app.core.schema as s
    print("OK: app.core.schema imported")
    print("Has ChunkDoc.create:", hasattr(s.ChunkDoc, "create"))
except Exception:
    traceback.print_exc()

print("\n=== Testing ingest modules (doc/image/audio) ===")
for mod in (
    "app.ingest.doc_ingestor",
    "app.ingest.image_ingestor",
    "app.ingest.audio_ingestor",
):
    try:
        m = __import__(mod, fromlist=["*"])
        print(f"OK: imported {mod}")
    except Exception:
        print(f"FAILED to import {mod}")
        traceback.print_exc()
