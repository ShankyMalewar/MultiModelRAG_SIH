# # app/ingest/ingest.py
# """
# Unified ingest orchestrator for the offline multimodal RAG.

# Key points:
# - Avoid heavy native/model imports at module import time.
# - Use only pure-Python fast logic by default (extension + mimetypes).
# - Optional: enable python-magic detection by setting env var USE_MAGIC=1.
# - Always lazy-import ingestors and heavy services inside functions.

# Usage:
#     from app.ingest.ingest import ingest_file
#     res = ingest_file("/path/to/file.pdf")
# """

# from __future__ import annotations

# import os
# import logging
# import mimetypes
# from typing import Optional

# logger = logging.getLogger("asklyne.ingest")
# # Don't call basicConfig if the app config already did logging; safe fallback:
# if not logging.getLogger().handlers:
#     logging.basicConfig(level=logging.INFO)

# # canonical result dataclass (fallback if schema import fails)
# try:
#     from app.core.schema import IngestResult
# except Exception:
#     class IngestResult:
#         def __init__(self, file_path: str, file_name: str, chunks_created: int = 0, chunk_ids=None, errors=None):
#             self.file_path = file_path
#             self.file_name = file_name
#             self.chunks_created = chunks_created
#             self.chunk_ids = chunk_ids or []
#             self.errors = errors or []

#         def to_dict(self):
#             return {
#                 "file_path": self.file_path,
#                 "file_name": self.file_name,
#                 "chunks_created": self.chunks_created,
#                 "chunk_ids": self.chunk_ids,
#                 "errors": self.errors,
#             }


# # extension sets
# _DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx"}
# _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif", ".heic"}
# _AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

# # Environment switch: only attempt python-magic detection if explicitly enabled.
# # This avoids importing native libmagic at module-import time on Windows.
# _USE_MAGIC = os.getenv("USE_MAGIC", "0") == "1"


# def _detect_type_by_ext(path: str) -> Optional[str]:
#     """Fast extension-based detection (pure Python)."""
#     ext = os.path.splitext(path)[1].lower()
#     if ext in _DOC_EXTS:
#         return "document"
#     if ext in _IMAGE_EXTS:
#         return "image"
#     if ext in _AUDIO_EXTS:
#         return "audio"
#     # fallback to mimetypes
#     mt, _ = mimetypes.guess_type(path)
#     if mt:
#         if mt.startswith("image/"):
#             return "image"
#         if mt.startswith("audio/"):
#             return "audio"
#         if mt in ("application/pdf", "application/msword") or mt.startswith("text/"):
#             return "document"
#     return None


# def _detect_type_by_magic(path: str) -> Optional[str]:
#     """
#     Optional detection using python-magic.
#     This function lazy-imports python-magic and catches any Exception.
#     It will only be called when _USE_MAGIC is True.
#     """
#     try:
#         # import inside function to avoid module-level import-time side-effects
#         import importlib
#         magic = importlib.import_module("magic")
#         # python-magic has different APIs depending on package version
#         try:
#             m = magic.Magic(mime=True)
#             mt = m.from_file(path)
#         except Exception:
#             # fallback to magic.from_file
#             mt = magic.from_file(path, mime=True)
#         if mt:
#             if mt.startswith("image/"):
#                 return "image"
#             if mt.startswith("audio/"):
#                 return "audio"
#             if mt in ("application/pdf", "application/msword") or mt.startswith("text/"):
#                 return "document"
#     except Exception as e:
#         # Be noisy at debug level only; do not raise.
#         logger.debug("python-magic detection failed or not available: %s", e)
#     return None


# def ingest_file(file_path: str, filename: Optional[str] = None, force_type: Optional[str] = None) -> IngestResult:
#     """
#     Ingest a single file and route to the appropriate ingestor.

#     Args:
#         file_path: local path to the file
#         filename: optional friendly filename to record
#         force_type: optional override ("document"|"image"|"audio")

#     Returns:
#         IngestResult produced by the chosen ingestor or an error IngestResult.
#     """
#     filename = filename or os.path.basename(file_path)
#     if not os.path.exists(file_path):
#         logger.error("File not found: %s", file_path)
#         return IngestResult(file_path, filename, chunks_created=0, errors=[f"file_not_found:{file_path}"])

#     # 1) determine file type (fast heuristics)
#     file_type = force_type
#     if not file_type:
#         file_type = _detect_type_by_ext(file_path)
#     if not file_type and _USE_MAGIC:
#         # only call magic if explicitly enabled via env var
#         logger.info("Using python-magic for type detection (USE_MAGIC=1)")
#         file_type = _detect_type_by_magic(file_path)

#     if not file_type:
#         logger.warning("Could not detect file type for %s (ext guess + optional magic failed). Returning error.", file_path)
#         return IngestResult(file_path, filename, chunks_created=0, errors=[f"unknown_file_type:{file_path}"])

#     file_type = file_type.lower()
#     logger.info("Ingest detected type=%s for %s", file_type, file_path)

#     # 2) lazy import ingestors and run
#     try:
#         if file_type == "document":
#             # Lazy import to avoid heavy model loads / native libs at import time
#             from app.ingest.doc_ingestor import DocIngestor
#             ing = DocIngestor()
#             return ing.ingest_document(file_path, filename=filename)
#         elif file_type == "image":
#             from app.ingest.image_ingestor import ImageIngestor
#             ing = ImageIngestor()
#             return ing.ingest_image(file_path, filename=filename)
#         elif file_type == "audio":
#             from app.ingest.audio_ingestor import AudioIngestor
#             ing = AudioIngestor()
#             return ing.ingest_audio(file_path, filename=filename)
#         else:
#             logger.error("Unsupported detected type: %s", file_type)
#             return IngestResult(file_path, filename, chunks_created=0, errors=[f"unsupported_type:{file_type}"])
#     except Exception as e:
#         logger.exception("Error during ingestion of %s: %s", file_path, e)
#         return IngestResult(file_path, filename, chunks_created=0, errors=[str(e)])


# def run_ingest(file_path: str):
#     """
#     Compatibility shim — delegates to ingest_file.
#     Keep this thin so it's safe to import.
#     """
#     return ingest_file(file_path)


# # CLI for manual testing (kept safe: only executed if run as script)
# if __name__ == "__main__":
#     import argparse
#     import pprint
#     logging.basicConfig(level=logging.INFO)

#     parser = argparse.ArgumentParser(prog="ingest")
#     parser.add_argument("file", help="Path to file to ingest")
#     parser.add_argument("--force-type", choices=["document", "image", "audio"], help="Force a file type (bypass detection)")
#     args = parser.parse_args()

#     res = ingest_file(args.file, force_type=args.force_type)
#     out = res.to_dict() if hasattr(res, "to_dict") else res.__dict__
#     pprint.pprint(out)





# app/ingest/ingest.py
"""
Unified ingest orchestrator for the offline multimodal RAG.

Key points:
- Adds optional `session_id` propagation so ingested chunks can be tagged with a session.
- Backwards-compatible: if session_id is not provided, behavior is unchanged.
- Avoid heavy imports at module import time; ingestors are lazy-imported.
"""

from __future__ import annotations

import os
import logging
import mimetypes
from typing import Optional

logger = logging.getLogger("asklyne.ingest")
# Don't call basicConfig if the app config already did logging; safe fallback:
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# canonical result dataclass (fallback if schema import fails)
try:
    from app.core.schema import IngestResult
except Exception:
    class IngestResult:
        def __init__(self, file_path: str, file_name: str, chunks_created: int = 0, chunk_ids=None, errors=None):
            self.file_path = file_path
            self.file_name = file_name
            self.chunks_created = chunks_created
            self.chunk_ids = chunk_ids or []
            self.errors = errors or []

        def to_dict(self):
            return {
                "file_path": self.file_path,
                "file_name": self.file_name,
                "chunks_created": self.chunks_created,
                "chunk_ids": self.chunk_ids,
                "errors": self.errors,
            }


# extension sets
_DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif", ".heic"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

# Environment switch: only attempt python-magic detection if explicitly enabled.
# This avoids importing native libmagic at module-import time on Windows.
_USE_MAGIC = os.getenv("USE_MAGIC", "0") == "1"


def _detect_type_by_ext(path: str) -> Optional[str]:
    """Fast extension-based detection (pure Python)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _DOC_EXTS:
        return "document"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    # fallback to mimetypes
    mt, _ = mimetypes.guess_type(path)
    if mt:
        if mt.startswith("image/"):
            return "image"
        if mt.startswith("audio/"):
            return "audio"
        if mt in ("application/pdf", "application/msword") or mt.startswith("text/"):
            return "document"
    return None


def _detect_type_by_magic(path: str) -> Optional[str]:
    """
    Optional detection using python-magic.
    This function lazy-imports python-magic and catches any Exception.
    It will only be called when _USE_MAGIC is True.
    """
    try:
        # import inside function to avoid module-level import-time side-effects
        import importlib
        magic = importlib.import_module("magic")
        # python-magic has different APIs depending on package version
        try:
            m = magic.Magic(mime=True)
            mt = m.from_file(path)
        except Exception:
            # fallback to magic.from_file
            mt = magic.from_file(path, mime=True)
        if mt:
            if mt.startswith("image/"):
                return "image"
            if mt.startswith("audio/"):
                return "audio"
            if mt in ("application/pdf", "application/msword") or mt.startswith("text/"):
                return "document"
    except Exception as e:
        # Be noisy at debug level only; do not raise.
        logger.debug("python-magic detection failed or not available: %s", e)
    return None


def ingest_file(file_path: str, filename: Optional[str] = None, force_type: Optional[str] = None, session_id: Optional[str] = None) -> IngestResult:
    """
    Ingest a single file and route to the appropriate ingestor.

    Args:
        file_path: local path to the file
        filename: optional friendly filename to record
        force_type: optional override ("document"|"image"|"audio")
        session_id: optional session id to tag created chunks with

    Returns:
        IngestResult produced by the chosen ingestor or an error IngestResult.
    """
    filename = filename or os.path.basename(file_path)
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return IngestResult(file_path, filename, chunks_created=0, errors=[f"file_not_found:{file_path}"])

    # 1) determine file type (fast heuristics)
    file_type = force_type
    if not file_type:
        file_type = _detect_type_by_ext(file_path)
    if not file_type and _USE_MAGIC:
        # only call magic if explicitly enabled via env var
        logger.info("Using python-magic for type detection (USE_MAGIC=1)")
        file_type = _detect_type_by_magic(file_path)

    if not file_type:
        logger.warning("Could not detect file type for %s (ext guess + optional magic failed). Returning error.", file_path)
        return IngestResult(file_path, filename, chunks_created=0, errors=[f"unknown_file_type:{file_path}"])

    file_type = file_type.lower()
    logger.info("Ingest detected type=%s for %s (session_id=%s)", file_type, file_path, session_id)

    # 2) lazy import ingestors and run
    try:
        if file_type == "document":
            # Lazy import to avoid heavy model loads / native libs at import time
            from app.ingest.doc_ingestor import DocIngestor
            ing = DocIngestor(session_id=session_id)
            return ing.ingest_document(file_path, filename=filename)
        elif file_type == "image":
            from app.ingest.image_ingestor import ImageIngestor
            ing = ImageIngestor(session_id=session_id)
            return ing.ingest_image(file_path, filename=filename)
        elif file_type == "audio":
            from app.ingest.audio_ingestor import AudioIngestor
            ing = AudioIngestor(session_id=session_id)
            return ing.ingest_audio(file_path, filename=filename)
        else:
            logger.error("Unsupported detected type: %s", file_type)
            return IngestResult(file_path, filename, chunks_created=0, errors=[f"unsupported_type:{file_type}"])
    except Exception as e:
        logger.exception("Error during ingestion of %s: %s", file_path, e)
        return IngestResult(file_path, filename, chunks_created=0, errors=[str(e)])


def run_ingest(file_path: str, session_id: Optional[str] = None):
    """
    Compatibility shim — delegates to ingest_file.
    Keep this thin so it's safe to import.
    """
    return ingest_file(file_path, session_id=session_id)


# CLI for manual testing (kept safe: only executed if run as script)
if __name__ == "__main__":
    import argparse
    import pprint
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("file", help="Path to file to ingest")
    parser.add_argument("--force-type", choices=["document", "image", "audio"], help="Force a file type (bypass detection)")
    parser.add_argument("--session-id", help="Optional session id to tag chunks", default=None)
    args = parser.parse_args()

    res = ingest_file(args.file, force_type=args.force_type, session_id=args.session_id)
    out = res.to_dict() if hasattr(res, "to_dict") else res.__dict__
    pprint.pprint(out)
