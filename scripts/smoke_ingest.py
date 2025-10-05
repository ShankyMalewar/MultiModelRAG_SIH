"""
scripts/smoke_ingest.py - robust smoke test runner.

Features:
 - Ensures repo root is on sys.path before importing app.* modules.
 - Tries import from package, and falls back to dynamic module load from file.
 - Per-file timeout to avoid hanging indefinitely.
"""

import argparse
import importlib
import importlib.util
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import List

# Make sure repository root is on sys.path so app.* modules import correctly
# This must happen before we try to import any app.* module.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_ingest")


def dynamic_import_module(module_name: str, file_path: Path):
    """
    Try to import a module by name; if that fails, attempt to load it from file_path
    (pure fallback if the package structure is not set up).
    """
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.debug("Package import failed for %s: %s. Trying file fallback: %s", module_name, e, file_path)
        if not file_path.exists():
            logger.debug("Fallback file does not exist: %s", file_path)
            raise

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None:
            raise ImportError(f"Cannot create spec for {file_path}")
        mod = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            raise ImportError(f"No loader for spec {spec}")
        loader.exec_module(mod)
        # also register under module_name for future imports
        sys.modules[module_name] = mod
        return mod


# Attempt imports (but using dynamic_import_module ensures robustness).
try:
    doc_ingestor = dynamic_import_module("app.ingest.doc_ingestor", repo_root / "app" / "ingest" / "doc_ingestor.py")
except Exception:
    doc_ingestor = None

try:
    image_ingestor = dynamic_import_module("app.ingest.image_ingestor", repo_root / "app" / "ingest" / "image_ingestor.py")
except Exception:
    image_ingestor = None

try:
    audio_ingestor = dynamic_import_module("app.ingest.audio_ingestor", repo_root / "app" / "ingest" / "audio_ingestor.py")
except Exception:
    audio_ingestor = None


def choose_ingestor_for_path(p: Path):
    ext = p.suffix.lower()
    if ext in [".pdf", ".docx", ".doc"]:
        return "doc"
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"]:
        return "image"
    if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"]:
        return "audio"
    return None


def run_doc_ingest(file_path: str):
    if doc_ingestor is None:
        raise RuntimeError("doc_ingestor module not importable")
    if hasattr(doc_ingestor, "ingest_document"):
        return doc_ingestor.ingest_document(file_path)
    if hasattr(doc_ingestor, "ingest_cli"):
        return doc_ingestor.ingest_cli(file_path)
    # Many doc ingestors expose a main() or ingest_file function — attempt common fallbacks
    for name in ("main", "ingest_file", "ingest_pdf"):
        if hasattr(doc_ingestor, name):
            return getattr(doc_ingestor, name)(file_path)
    raise RuntimeError("doc_ingestor has no known ingest entrypoint")


def run_image_ingest(file_path: str):
    if image_ingestor is None:
        raise RuntimeError("image_ingestor module not importable")
    if hasattr(image_ingestor, "ImageIngestor"):
        inst = image_ingestor.ImageIngestor()
        if hasattr(inst, "ingest_image"):
            return inst.ingest_image(file_path)
    # fallback functions
    for name in ("ingest_image", "ingest", "ingest_file"):
        if hasattr(image_ingestor, name):
            return getattr(image_ingestor, name)(file_path)
    raise RuntimeError("image_ingestor has no known ingest entrypoint")


def run_audio_ingest(file_path: str):
    if audio_ingestor is None:
        raise RuntimeError("audio_ingestor module not importable")
    if hasattr(audio_ingestor, "AudioIngestor"):
        inst = audio_ingestor.AudioIngestor()
        # common method names
        for name in ("ingest_audio", "ingest", "transcribe_and_ingest"):
            if hasattr(inst, name):
                return getattr(inst, name)(file_path)
    # fallback module-level functions
    for name in ("ingest_audio", "ingest", "transcribe_and_ingest"):
        if hasattr(audio_ingestor, name):
            return getattr(audio_ingestor, name)(file_path)
    raise RuntimeError("audio_ingestor has no known ingest entrypoint")


def process_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        return {"file": file_path, "ok": False, "error": "not_found"}

    kind = choose_ingestor_for_path(path)
    if kind is None:
        return {"file": file_path, "ok": False, "error": f"unsupported_ext:{path.suffix}"}

    logger.info("Ingest start: %s (type=%s)", file_path, kind)
    start = time.time()
    try:
        if kind == "doc":
            res = run_doc_ingest(str(path))
        elif kind == "image":
            res = run_image_ingest(str(path))
        elif kind == "audio":
            res = run_audio_ingest(str(path))
        else:
            raise RuntimeError("unknown_ingestor")

        duration = time.time() - start
        logger.info("Ingest done: %s (t=%.2fs) result=%s", file_path, duration, getattr(res, "chunks_created", "N/A"))
        return {"file": file_path, "ok": True, "result": res}
    except Exception as e:
        duration = time.time() - start
        logger.exception("Ingest failed: %s (t=%.2fs) err=%s", file_path, duration, e)
        return {"file": file_path, "ok": False, "error": str(e)}


def main(files: List[str], timeout: int):
    # Normalize file list
    file_list = []
    for f in files:
        if "," in f:
            file_list.extend([x.strip() for x in f.split(",") if x.strip()])
        else:
            file_list.append(f)

    logger.info("Smoke ingest: files=%s timeout_per_file=%ds", file_list, timeout)

    results = []
    for file_path in file_list:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(process_file, file_path)
            try:
                res = future.result(timeout=timeout)
                results.append(res)
            except TimeoutError:
                logger.error("Timeout while ingesting %s (limit %ds). Marking failed.", file_path, timeout)
                results.append({"file": file_path, "ok": False, "error": f"timeout_{timeout}s"})
            except Exception as e:
                logger.exception("Unexpected error running ingestion for %s: %s", file_path, e)
                results.append({"file": file_path, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    fail_count = len(results) - ok_count
    logger.info("Smoke summary: total=%d ok=%d failed=%d", len(results), ok_count, fail_count)
    for r in results:
        if not r.get("ok"):
            logger.warning("FAILED: %s -> %s", r["file"], r.get("error"))
        else:
            logger.info("OK: %s", r["file"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test: ingest files into local stack (qdrant, models).")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="One or more files (comma-separated allowed).")
    parser.add_argument("--timeout", "-t", type=int, default=180, help="Seconds timeout per file (default 180s).")
    args = parser.parse_args()

    main(args.files, timeout=args.timeout)
