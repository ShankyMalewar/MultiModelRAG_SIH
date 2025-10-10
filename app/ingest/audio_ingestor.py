# app/ingest/audio_ingestor.py (final patched)
import os
import logging
import time
from typing import Optional, List

from app.core.schema import ChunkDoc, IngestResult
from app.core.chunker import chunk_text
from app.core.embedder import Embedder
from app.services.qdrant_service import QdrantService

logger = logging.getLogger("asklyne.ingest.audio")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def get_asr_handler(model_name: Optional[str] = None):
    from app.extractors.asr_handler import ASRHandler
    return ASRHandler(model_name=model_name or "tiny")


class AudioIngestor:
    def __init__(self, session_id: Optional[str] = None, asr=None, embedder=None, qdrant=None):
        self.session_id = session_id
        self._asr = asr
        self._embedder = embedder
        self._qdrant = qdrant

    @property
    def asr(self):
        if self._asr is None:
            logger.info("AudioIngestor: initializing ASR model...")
            self._asr = get_asr_handler()
            logger.info("AudioIngestor: ASR model ready.")
        return self._asr

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info("AudioIngestor: initializing embedder...")
            self._embedder = Embedder(mode="text")
            logger.info("AudioIngestor: embedder ready.")
        return self._embedder

    @property
    def qdrant(self):
        if self._qdrant is None:
            logger.info("AudioIngestor: connecting to Qdrant...")
            self._qdrant = QdrantService()
            logger.info("AudioIngestor: Qdrant connection ready.")
        return self._qdrant

    def ingest_audio(self, file_path: str, filename: Optional[str] = None) -> IngestResult:
        start_t = time.time()
        filename = filename or os.path.basename(file_path)
        result = IngestResult(file_path=file_path, file_name=filename, chunk_ids=[], chunks_created=0, errors=[])

        logger.info("AudioIngestor: starting pipeline for %s (session=%s)", file_path, self.session_id)

        # 1️⃣ Transcribe
        transcription = self.asr.transcribe(file_path)
        text = transcription.get("text", "").strip()
        segments = transcription.get("segments", [])
        if not text:
            result.errors.append("no_text_extracted")
            logger.error("AudioIngestor: no text extracted.")
            return result

        # 2️⃣ Chunking
        chunks: List[ChunkDoc] = []
        for seg in segments or [{"text": text, "start": 0.0, "end": None}]:
            sub_chunks = chunk_text(seg.get("text", ""))
            for sc in sub_chunks:
                meta = {
                    "filename": filename,
                    "source": file_path,
                    "segment_start": seg.get("start"),
                    "segment_end": seg.get("end"),
                    "type": "audio",
                    "modality": "audio",  # ✅ consistent lowercase modality
                    "is_audio": True,
                    "media_type": "audio",
                }
                if self.session_id:
                    meta["session_id"] = self.session_id
                chunk = ChunkDoc.create(
                    text=sc,
                    modality="audio",
                    filename=filename,
                    meta=meta
                )
                chunks.append(chunk)

        result.chunks_created = len(chunks)
        logger.info("AudioIngestor: total %d text chunks created.", len(chunks))

        # 3️⃣ Embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_chunks(texts)
        for c, e in zip(chunks, embeddings):
            if hasattr(e, "tolist"):
                e = e.tolist()
            c.embedding = e

        # 4️⃣ Upsert with enforced modality
        for c in chunks:
            if "modality" not in c.meta:
                c.meta["modality"] = "audio"
            if not getattr(c, "payload", None):
                c.payload = {}
            c.payload.update({
                "text": c.text,
                "session_id": self.session_id,
                "modality": "audio",
            })

        upserted, failed = self.qdrant.upsert_chunks(chunks)
        result.chunk_ids = [c.id for c in chunks]
        logger.info("AudioIngestor: upsert finished (success=%s, failed=%d).", upserted, len(failed))

        logger.info(
            "AudioIngestor done: %s chunks=%d errors=%s (%.2fs)",
            filename, result.chunks_created, result.errors, time.time() - start_t,
        )
        return result
