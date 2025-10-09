# app/ingest/doc_ingestor.py
"""
Robust Document ingestor (patched)

Improvements in this patch:
- Always produce a plain dict payload for Qdrant upsert (id, embedding, payload) in addition
  to attempting to create ChunkDoc objects. This avoids schema mismatches between different
  ChunkDoc implementations and the Qdrant upsert code.
- Validate and normalize embeddings returned from Embedder: convert numpy/torch tensors to
  python lists of floats, ensure the vector length matches Qdrant collection dimension and
  log/skip otherwise.
- Clear, actionable logging when chunks are skipped (missing embedding or wrong dim).
- Keep backward compatible `ChunkDoc` creation attempts while ensuring upsert will get
  correct payloads.

This file is intended to replace your current doc_ingestor.py. After saving it, restart
your app and re-run an ingest. If you still see issues, paste the logs for the `DocIngestor`
and the `QdrantService` last_payload / last_server_response.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Any
from inspect import signature

from app.core.chunker import chunk_paragraphs, chunk_text
from app.core.embedder import Embedder
from app.core.schema import ChunkDoc, IngestResult
from app.services.qdrant_service import QdrantService

# Extraction libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx
except Exception:
    docx = None

try:
    from app.extractors.ocr_handler import OCRHandler
except Exception:
    OCRHandler = None

logger = logging.getLogger("asklyne.ingest.doc")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def extract_text_from_pdf(path: str) -> List[str]:
    pages_paragraphs: List[str] = []
    if fitz is None:
        logger.error("PyMuPDF not installed; skipping text extraction.")
        return []

    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.exception("Failed to open PDF: %s", e)
        return []

    for page_num, page in enumerate(doc):
        try:
            text = page.get_text("text")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            pages_paragraphs.append("\n\n".join(paragraphs))
        except Exception as e:
            logger.exception("Failed to read page %d: %s", page_num + 1, e)
            pages_paragraphs.append("")
    doc.close()
    return pages_paragraphs


def extract_text_from_docx(path: str) -> List[str]:
    if docx is None:
        logger.error("python-docx not installed; cannot read DOCX.")
        return []
    paragraphs: List[str] = []
    try:
        doc = docx.Document(path)
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                paragraphs.append(txt)
    except Exception as e:
        logger.exception("Failed to parse DOCX: %s", e)
        return []
    return paragraphs


def _create_chunk_doc_with_fallback(ChunkDocCls, **kwargs):
    """
    Robust creation wrapper for ChunkDoc.create that tolerates different
    schema field names for the vector/embedding field.
    """
    base_kwargs = dict(kwargs)
    emb = base_kwargs.pop("embedding", None)

    candidates = ["embedding", "vector", "embedding_vector", "vec", "embedding_vec", "emb", "vector_embedding"]

    attempts = []
    if emb is not None:
        for name in candidates:
            kop = base_kwargs.copy()
            kop[name] = emb
            attempts.append(kop)
    else:
        attempts.append(base_kwargs.copy())

    last_exc = None
    for attempt_kwargs in attempts:
        try:
            return ChunkDocCls.create(**attempt_kwargs)
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            raise

    try:
        sig = signature(ChunkDocCls.create)
        params = list(sig.parameters.keys())
        mapped = {}
        for p in params:
            if p in kwargs:
                mapped[p] = kwargs[p]
            elif p in ("vector", "embedding_vector", "vec", "emb", "embedding") and emb is not None:
                mapped[p] = emb
        return ChunkDocCls.create(**mapped)
    except Exception as e:
        raise RuntimeError(f"ChunkDoc.create fallback attempts failed. Last error: {last_exc or e}") from (last_exc or e)


class DocIngestor:
    def __init__(self, session_id: Optional[str] = None):
        self.embedder = Embedder()
        self.qdrant = QdrantService()
        self.session_id = session_id
        # lazily instantiate OCRHandler only when needed
        self._ocr_handler = None

    @property
    def ocr_handler(self):
        if self._ocr_handler is None and OCRHandler is not None:
            try:
                self._ocr_handler = OCRHandler()
            except Exception:
                self._ocr_handler = None
        return self._ocr_handler

    def _normalize_embedding(self, emb: Any) -> Optional[List[float]]:
        """Convert embedding tensor/array to python list of floats and validate dim.
        Returns None if conversion fails or dim mismatch.
        """
        if emb is None:
            return None
        # convert numpy/torch/other to list
        try:
            if hasattr(emb, "tolist"):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb)
        except Exception:
            logger.exception("Failed to convert embedding to list: %s", type(emb))
            return None

        # If nested numpy returns e.g. array([...], dtype=float32) -> ensure flat list of floats
        try:
            emb_list = [float(x) for x in emb_list]
        except Exception:
            # try flatten if it's an array-like of arrays
            try:
                flat = []
                for sub in emb_list:
                    for x in sub:
                        flat.append(float(x))
                emb_list = flat
            except Exception:
                logger.exception("Failed to coerce embedding elements to float")
                return None

        # Validate length against Qdrant collection expected dimension
        expected = getattr(self.qdrant, "_vector_size", None)
        # If Qdrant has named vectors mapping, try to use first entry
        if expected is None:
            named = getattr(self.qdrant, "_named_vectors", None)
            if isinstance(named, dict) and len(named) > 0:
                # pick first value
                expected = list(named.values())[0]

        if expected is not None and len(emb_list) != expected:
            logger.warning("Embedding dimension mismatch: expected %s got %s. This chunk will be skipped.", expected, len(emb_list))
            return None

        return emb_list

    def ingest_document(self, file_path: str, filename: Optional[str] = None, ocr_if_scanned=True) -> IngestResult:
        start_t = time.time()
        path = Path(file_path)
        filename = filename or path.name
        result = IngestResult(file_path=str(path), file_name=filename, chunks_created=0, errors=[])

        logger.info("DocIngestor: starting ingest for %s (session=%s)", file_path, self.session_id)
        ext = path.suffix.lower()
        paragraphs: List[str] = []

        # 1. Extract text
        if ext == ".pdf":
            logger.info("DocIngestor: extracting text from PDF...")
            paragraphs = []
            pages = extract_text_from_pdf(str(path))
            for pg in pages:
                if pg.strip():
                    paragraphs.extend([p.strip() for p in pg.split("\n\n") if p.strip()])

            # OCR fallback for scanned PDFs
            if len("".join(paragraphs)) < 200 and ocr_if_scanned and self.ocr_handler is not None:
                logger.info("DocIngestor: low-text PDF, performing OCR fallback...")
                try:
                    ocr_text = self.ocr_handler.extract_text_from_pdf(str(path))
                    if isinstance(ocr_text, str):
                        paragraphs = [p for p in ocr_text.split("\n\n") if p.strip()]
                    elif isinstance(ocr_text, list):
                        paragraphs = [p for page in ocr_text for p in page.split("\n\n") if p.strip()]
                except Exception as e:
                    logger.exception("DocIngestor: OCR fallback failed: %s", e)
        elif ext in [".doc", ".docx"]:
            logger.info("DocIngestor: extracting text from DOCX...")
            paragraphs = extract_text_from_docx(str(path))
        else:
            result.errors.append(f"unsupported_file_type:{ext}")
            logger.error("DocIngestor: unsupported extension %s", ext)
            return result

        if not paragraphs:
            result.errors.append("no_text_extracted")
            logger.error("DocIngestor: no text extracted.")
            return result

        logger.info("DocIngestor: extracted %d paragraphs.", len(paragraphs))

        # 2. Chunkify
        logger.info("DocIngestor: chunking paragraphs...")
        try:
            if isinstance(paragraphs, list):
                joined_text = "\n\n".join([p for p in paragraphs if isinstance(p, str)])
            else:
                joined_text = paragraphs

            chunks_texts = chunk_paragraphs(joined_text, min_length=20)

            if not chunks_texts:
                chunks_texts = chunk_text("\n\n".join(paragraphs))
        except Exception as e:
            logger.exception("DocIngestor: chunking failed: %s", e)
            result.errors.append(f"chunking_failed:{e}")
            return result

        if not chunks_texts:
            result.errors.append("no_chunks_generated")
            logger.error("DocIngestor: no chunks produced.")
            return result
        logger.info("DocIngestor: produced %d chunks.", len(chunks_texts))

        # 3. Embed
        try:
            logger.info("DocIngestor: embedding %d chunks...", len(chunks_texts))
            embeddings = self.embedder.embed_chunks(chunks_texts)
            logger.info("DocIngestor: embedding complete.")
        except Exception as e:
            logger.exception("DocIngestor: embedding failed: %s", e)
            result.errors.append(f"embedding_failed:{e}")
            return result

        # 4. Build Qdrant upsert payloads (robust) and optional ChunkDoc objects
        upsert_payloads = []
        chunk_objs: List[ChunkDoc] = []
        for idx, (text, emb) in enumerate(zip(chunks_texts, embeddings)):
            try:
                emb_list = self._normalize_embedding(emb)
                if emb_list is None:
                    logger.warning("DocIngestor: skipping chunk idx=%s due to invalid embedding.", idx)
                    result.errors.append(f"invalid_embedding_idx:{idx}")
                    continue

                meta = {"chunk_index": idx}
                if self.session_id:
                    meta["session_id"] = self.session_id

                # Build a plain payload dict that matches QdrantService expectations
                payload = {
                    "id": None,
                    "embedding": emb_list,
                    "payload": {
                        "text": text,
                        "filename": filename,
                        "source_path": str(path),
                        **meta,
                    },
                }
                upsert_payloads.append(payload)

                # Try creating a ChunkDoc if available, but don't depend on it for upsert
                try:
                    create_kwargs = dict(
                        text=text,
                        modality="text",
                        filename=filename,
                        source_path=str(path),
                        meta=meta,
                        embedding=emb_list,
                    )
                    cd = _create_chunk_doc_with_fallback(ChunkDoc, **create_kwargs)
                    chunk_objs.append(cd)
                except Exception:
                    # non-fatal
                    logger.debug("DocIngestor: ChunkDoc.create failed for idx=%s; continuing with plain payload.", idx)

            except Exception as e:
                logger.exception("DocIngestor: failed to prepare chunk idx=%s: %s", idx, e)
                result.errors.append(f"chunk_prepare_failed:{idx}:{e}")
                continue

        if not upsert_payloads:
            result.errors.append("no_valid_embeddings_to_upsert")
            logger.error("DocIngestor: no valid payloads to upsert; aborting.")
            return result

        # 5. Upsert to Qdrant
        try:
            logger.info("DocIngestor: upserting %d chunks to Qdrant...", len(upsert_payloads))
            upserted, failed = self.qdrant.upsert_chunks(upsert_payloads, batch_size=64)
            logger.info("DocIngestor: upsert complete (success=%d, failed=%d).", upserted, len(failed))

            ids = []
            for p in upsert_payloads:
                # Qdrant may assign id server-side; if our payload had id None, Qdrant upsert returns ids.
                # Keep any id present in local payloads.
                try:
                    cid = p.get("id")
                    if cid:
                        ids.append(cid)
                except Exception:
                    continue

            # If QdrantService.upsert_chunks returns ids, the caller can inspect svc.last_server_response
            # to map assigned ids — we record the number inserted here.
            result.chunk_ids = ids
            result.chunks_created = upserted
        except Exception as e:
            logger.exception("DocIngestor: Qdrant upsert failed: %s", e)
            result.errors.append(f"qdrant_upsert_failed:{e}")

        logger.info(
            "DocIngestor done: %s chunks=%d errors=%s (%.2fs)",
            filename,
            result.chunks_created,
            result.errors,
            time.time() - start_t,
        )
        return result
