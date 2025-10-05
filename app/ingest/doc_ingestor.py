# app/ingest/doc_ingestor.py
"""
Document ingestion for PDF / DOCX
"""

import logging
import time
from pathlib import Path
from typing import List, Optional
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

    - Tries common candidate names for the embedding vector (embedding, vector, vec, etc).
    - If none of them works, inspects ChunkDocCls.create signature and attempts to map.
    - Raises the final Exception if creation fails.
    """
    base_kwargs = dict(kwargs)
    emb = base_kwargs.pop("embedding", None)

    # candidate names to try (in order)
    candidates = ["embedding", "vector", "embedding_vector", "vec", "embedding_vec", "emb", "vector_embedding"]

    attempts = []
    # If embedding present, create attempts with different key names
    if emb is not None:
        for name in candidates:
            kop = base_kwargs.copy()
            kop[name] = emb
            attempts.append(kop)
    else:
        # If no embedding provided, just attempt the provided kwargs
        attempts.append(base_kwargs.copy())

    # Try each attempt
    last_exc = None
    for attempt_kwargs in attempts:
        try:
            return ChunkDocCls.create(**attempt_kwargs)
        except TypeError as e:
            # wrong kwarg names likely; try next
            last_exc = e
            continue
        except Exception as e:
            # other failures (validation, DB) should be surfaced
            raise

    # Introspect signature and try to map parameters heuristically
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
    def __init__(self):
        self.embedder = Embedder()
        self.qdrant = QdrantService()

    def ingest_document(self, file_path: str, filename: Optional[str] = None, ocr_if_scanned=True) -> IngestResult:
        start_t = time.time()
        path = Path(file_path)
        filename = filename or path.name
        result = IngestResult(file_path=str(path), file_name=filename, chunks_created=0, errors=[])

        logger.info("DocIngestor: starting ingest for %s", file_path)
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

            # OCR fallback
            if len("".join(paragraphs)) < 200 and ocr_if_scanned and OCRHandler is not None:
                logger.info("DocIngestor: low-text PDF, performing OCR fallback...")
                try:
                    ocr = OCRHandler()
                    ocr_text = ocr.extract_text_from_pdf(str(path))
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
            # ensure paragraphs is a single text string for chunker
            if isinstance(paragraphs, list):
                joined_text = "\n\n".join([p for p in paragraphs if isinstance(p, str)])
            else:
                joined_text = paragraphs

            chunks_texts = chunk_paragraphs(joined_text, min_length=20)

            if not chunks_texts:
                # fallback to chunk_text if chunk_paragraphs produced nothing
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

        # 4. Create ChunkDocs (robust)
        chunk_objs: List[ChunkDoc] = []
        for idx, (text, emb) in enumerate(zip(chunks_texts, embeddings)):
            try:
                # ensure embedding is a plain list of floats
                if hasattr(emb, "tolist"):
                    emb_list = list(map(float, emb.tolist()))
                else:
                    emb_list = list(map(float, emb))

                # prepare kwargs commonly used by ChunkDoc.create
                create_kwargs = dict(
                    text=text,
                    modality="text",
                    filename=filename,
                    source_path=str(path),
                    meta={"chunk_index": idx},
                    embedding=emb_list,
                )

                # use robust wrapper to tolerate schema differences
                chunk_obj = _create_chunk_doc_with_fallback(ChunkDoc, **create_kwargs)

                # ensure the chunk object exposes the embedding in common attribute names
                # (some schemas expect .vector or .embedding)
                try:
                    # if returned object is a dict-like object set keys
                    if isinstance(chunk_obj, dict):
                        # set common keys
                        chunk_obj["embedding"] = emb_list
                        chunk_obj["vector"] = emb_list
                    else:
                        # set attributes if not present
                        if not hasattr(chunk_obj, "embedding"):
                            try:
                                setattr(chunk_obj, "embedding", emb_list)
                            except Exception:
                                pass
                        if not hasattr(chunk_obj, "vector"):
                            try:
                                setattr(chunk_obj, "vector", emb_list)
                            except Exception:
                                pass
                except Exception:
                    # not fatal; proceed but log
                    logger.debug("DocIngestor: couldn't attach embedding attributes to chunk_obj; proceeding.")

                chunk_objs.append(chunk_obj)
            except Exception as e:
                logger.exception("DocIngestor: failed to create chunk doc for idx=%s: %s", idx, e)
                result.errors.append(f"chunk_create_failed:{e}")
                # continue to next chunk (do not abort entire ingest for a single chunk)
                continue

        if not chunk_objs:
            result.errors.append("no_chunk_objects_created")
            logger.error("DocIngestor: no chunk objects created; aborting upsert.")
            return result

        # 5. Upsert
        try:
            logger.info("DocIngestor: upserting %d chunks to Qdrant...", len(chunk_objs))
            upserted, failed = self.qdrant.upsert_chunks(chunk_objs, batch_size=64)
            logger.info("DocIngestor: upsert complete (success=%d, failed=%d).", upserted, len(failed))
            # Attempt to collect ids if available
            ids = []
            for c in chunk_objs:
                try:
                    cid = getattr(c, "id", None)
                    if cid is None and isinstance(c, dict):
                        cid = c.get("id") or c.get("chunk_id") or c.get("chunk_id")
                    if cid:
                        ids.append(cid)
                except Exception:
                    continue
            result.chunk_ids = ids
            result.chunks_created = len(ids)
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
