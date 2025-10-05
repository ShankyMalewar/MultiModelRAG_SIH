#!/usr/bin/env python3
"""
scripts/preload_models.py

Pre-downloads ML models needed for Asklyne Offline so the system can run fully offline.
Creates a local `models_cache/` directory and downloads:
 - sentence-transformers embedding model (BAAI/bge-large-en-v1.5)
 - optional reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) if RERANKER_ENABLED=1
 - faster-whisper models (tiny, small) if requested
 - (optional) other model names provided via env vars

Usage:
    python scripts/preload_models.py --models embed,reranker,whisper_small
    python scripts/preload_models.py    # default: embeddings + whisper_tiny (fast)
"""

from pathlib import Path
import os
import argparse
import shutil
import subprocess
import sys
from typing import List

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

DEFAULT_CACHE_DIR = Path.cwd() / "models_cache"

# Defaults aligned with chosen defaults
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_TEXT", "BAAI/bge-large-en-v1.5")
DEFAULT_RERANKER = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_WHISPER_TINY = "openai/whisper-tiny"  # names for faster-whisper / whisper lookups (used for hub snapshot convenience)
DEFAULT_WHISPER_SMALL = "openai/whisper-small"

SYSTEM_DEPENDENCIES = {
    "tesseract": ["tesseract", "--version"],
    "ffmpeg": ["ffmpeg", "-version"],
    "pdftoppm": ["pdftoppm", "-h"],  # poppler pdf->image
}

def check_system_deps():
    missing = []
    for name, cmd in SYSTEM_DEPENDENCIES.items():
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            missing.append(name)
    return missing

def ensure_cache_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Using cache dir: {path.resolve()}")

def download_snapshot(repo_id: str, dest: Path, use_hf: bool = True) -> bool:
    """
    Download a snapshot of a HF repo into dest directory.
    Returns True on success.
    """
    if snapshot_download is None:
        print("[!] huggingface_hub.snapshot_download not available. Install huggingface-hub to predownload models.")
        return False

    print(f"[+] Downloading {repo_id} into {dest}")
    try:
        snapshot_download(repo_id=repo_id, cache_dir=str(dest), allow_patterns=None, local_dir_use_symlinks=False)
        print(f"[+] Downloaded {repo_id}")
        return True
    except Exception as e:
        print(f"[!] Failed to download {repo_id}: {e}")
        return False

def download_sentence_transformer(model_name: str, cache_dir: Path) -> bool:
    """
    For sentence-transformers, huggingface snapshot is usually fine. We'll call snapshot_download.
    """
    return download_snapshot(model_name, cache_dir, use_hf=True)

def download_whisper_model(model_name: str, cache_dir: Path) -> bool:
    """
    For faster-whisper / whisper, pulling the HF model snapshot is helpful.
    """
    return download_snapshot(model_name, cache_dir, use_hf=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", "-c", default=str(DEFAULT_CACHE_DIR), help="Directory to store downloaded models")
    p.add_argument("--models", "-m", default="embed,whisper_tiny", help="Comma-separated models to download: embed, reranker, whisper_tiny, whisper_small")
    p.add_argument("--force", "-f", action="store_true", help="Force re-download even if present")
    return p.parse_args()

def model_already_present(cache_dir: Path, marker: str) -> bool:
    # Look for a simple marker: a directory containing files
    p = cache_dir / marker
    return p.exists() and any(p.iterdir())

def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    ensure_cache_dir(cache_dir)

    missing_deps = check_system_deps()
    if missing_deps:
        print("[!] System dependencies missing (install via system package manager):", ", ".join(missing_deps))
        print("    These are required for OCR/ffmpeg/pdf processing but not strictly for model downloads.")
    else:
        print("[+] System dependencies present.")

    requested = [s.strip().lower() for s in args.models.split(",") if s.strip()]
    print("[+] Requested model groups:", requested)

    # embedding model
    if "embed" in requested or "emb" in requested:
        marker = DEFAULT_EMBED_MODEL.replace("/", "_")
        target = cache_dir / marker
        if model_already_present(cache_dir, marker) and not args.force:
            print(f"[+] Embed model already cached at {target}. Skip.")
        else:
            ok = download_sentence_transformer(DEFAULT_EMBED_MODEL, cache_dir)
            if not ok:
                print(f"[!] Embed model download failed: {DEFAULT_EMBED_MODEL}")

    # reranker
    if "reranker" in requested or "ranker" in requested:
        marker = DEFAULT_RERANKER.replace("/", "_")
        target = cache_dir / marker
        if model_already_present(cache_dir, marker) and not args.force:
            print(f"[+] Reranker model already cached at {target}. Skip.")
        else:
            ok = download_snapshot(DEFAULT_RERANKER, cache_dir) if snapshot_download else False
            if not ok:
                print(f"[!] Reranker download failed: {DEFAULT_RERANKER}")

    # whisper tiny
    if "whisper_tiny" in requested or "whisper" in requested:
        marker = DEFAULT_WHISPER_TINY.replace("/", "_")
        target = cache_dir / marker
        if model_already_present(cache_dir, marker) and not args.force:
            print(f"[+] Whisper tiny model already cached at {target}. Skip.")
        else:
            ok = download_whisper_model(DEFAULT_WHISPER_TINY, cache_dir)
            if not ok:
                print(f"[!] Whisper tiny download failed: {DEFAULT_WHISPER_TINY}")

    # whisper small
    if "whisper_small" in requested:
        marker = DEFAULT_WHISPER_SMALL.replace("/", "_")
        target = cache_dir / marker
        if model_already_present(cache_dir, marker) and not args.force:
            print(f"[+] Whisper small model already cached at {target}. Skip.")
        else:
            ok = download_whisper_model(DEFAULT_WHISPER_SMALL, cache_dir)
            if not ok:
                print(f"[!] Whisper small download failed: {DEFAULT_WHISPER_SMALL}")

    print("[+] Pre-download script finished. NOTE: Models may still be lazily loaded by downstream libs at runtime.")
    print(f"[+] Check {cache_dir} to confirm presence. You can set EMBED_MODEL, RERANKER_MODEL, and other env vars to point exactly to these model names if needed.")

if __name__ == "__main__":
    main()
