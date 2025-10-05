# app/ingest/audio_ingestor.py
import os
import logging
import time
from typing import Optional, List

from app.core.schema import ChunkDoc, IngestResult
from app.core.chunker import chunk_text
from app.core.embedder import Embedder
from app.services.qdrant_service import QdrantService

logger = logging.getLogger("asklyne.ingest.audio")
logging.basicConfig(level=logging.INFO)


def get_asr_handler(model_name: Optional[str] = None):
    from app.extractors.asr_handler import ASRHandler
    return ASRHandler(model_name=model_name or "tiny")


class AudioIngestor:
    def __init__(self, asr=None, embedder=None, qdrant=None):
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

        result = IngestResult(file_name=filename, file_path=file_path, chunk_ids=[], chunks_created=0, errors=[])

        logger.info("AudioIngestor: starting pipeline for %s", file_path)

        # 1. Transcribe
        try:
            logger.info("AudioIngestor: starting ASR transcription...")
            transcription = self.asr.transcribe(file_path)
            logger.info("AudioIngestor: ASR transcription finished.")
        except Exception as e:
            logger.exception("AudioIngestor: transcription failed: %s", e)
            result.errors.append(f"transcription_failed:{e}")
            return result

        text = transcription.get("text", "").strip()
        segments = transcription.get("segments", [])
        if not text:
            result.errors.append("no_text_extracted")
            logger.error("AudioIngestor: no text extracted.")
            return result

        # 2. Chunk
        logger.info("AudioIngestor: chunking %d ASR segments...", len(segments))
        chunks: List[ChunkDoc] = []
        for seg in segments:
            sub_chunks = chunk_text(seg.get("text", ""))
            for sc in sub_chunks:
                meta = {
                    "filename": filename,
                    "modality": "audio",
                    "segment_start": seg.get("start"),
                    "segment_end": seg.get("end"),
                    "source": file_path,
                }
                chunks.append(ChunkDoc.create(text=sc, modality="audio", filename=filename, meta=meta))
        result.chunks_created = len(chunks)
        logger.info("AudioIngestor: total %d text chunks created.", len(chunks))

        # 3. Embed
        try:
            logger.info("AudioIngestor: embedding %d chunks...", len(chunks))
            vectors = self.embedder.embed_batch([c.text for c in chunks])
            for c, v in zip(chunks, vectors):
                c.embedding = v
            logger.info("AudioIngestor: embedding completed.")
        except Exception as e:
            logger.exception("AudioIngestor: embedding failed: %s", e)
            result.errors.append(f"embedding_failed:{e}")

        # 4. Upsert
        try:
            logger.info("AudioIngestor: upserting %d chunks to Qdrant...", len(chunks))
            upserted, failed = self.qdrant.upsert_chunks(chunks)
            result.chunk_ids = [c.id for c in chunks if c.id not in failed]
            logger.info("AudioIngestor: upsert finished (success=%d, failed=%d).", upserted, len(failed))
        except Exception as e:
            logger.exception("AudioIngestor: Qdrant upsert failed: %s", e)
            result.errors.append(f"qdrant_upsert_failed:{e}")

        logger.info(
            "AudioIngestor done: %s chunks=%d errors=%s (%.2fs)",
            filename, result.chunks_created, result.errors, time.time() - start_t,
        )
        return result
