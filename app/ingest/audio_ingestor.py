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
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def get_asr_handler(model_name: Optional[str] = None):
    from app.extractors.asr_handler import ASRHandler
    return ASRHandler(model_name=model_name or "tiny")


class AudioIngestor:
    def __init__(self, session_id: Optional[str] = None, asr=None, embedder=None, qdrant=None):
        """
        Args:
            session_id: optional session id to tag created chunk metadata with
            asr/embedder/qdrant: optional injected dependencies for easier testing
        """
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
                if self.session_id:
                    meta["session_id"] = self.session_id

                try:
                    chunk = ChunkDoc.create(text=sc, modality="audio", filename=filename, meta=meta)
                except Exception:
                    # fallback: create a simple dict-like chunk if ChunkDoc.create not available
                    chunk = {
                        "text": sc,
                        "modality": "audio",
                        "filename": filename,
                        "meta": meta,
                    }
                chunks.append(chunk)

        result.chunks_created = len(chunks)
        logger.info("AudioIngestor: total %d text chunks created.", len(chunks))

        # 3. Embed
        try:
            logger.info("AudioIngestor: embedding %d chunks...", len(chunks))
            texts = [c["text"] if isinstance(c, dict) else getattr(c, "text", "") for c in chunks]

            # Prefer embed_chunks, fallback to embed_batch for backward compatibility
            embeddings = None
            if hasattr(self.embedder, "embed_chunks"):
                embeddings = self.embedder.embed_chunks(texts)
            elif hasattr(self.embedder, "embed_batch"):
                embeddings = self.embedder.embed_batch(texts)
            else:
                # try a generic embed_text or embed method
                if hasattr(self.embedder, "embed"):
                    embeddings = [self.embedder.embed(t) for t in texts]
                else:
                    raise RuntimeError("Embedder has no known embed method")

            # attach embeddings back to chunk objects
            for c, emb in zip(chunks, embeddings):
                try:
                    if isinstance(c, dict):
                        c["embedding"] = emb
                    else:
                        setattr(c, "embedding", emb)
                except Exception:
                    logger.debug("Could not attach embedding to chunk; continuing.")
            logger.info("AudioIngestor: embedding completed.")
        except Exception as e:
            logger.exception("AudioIngestor: embedding failed: %s", e)
            result.errors.append(f"embedding_failed:{e}")

        # 4. Upsert
        try:
            logger.info("AudioIngestor: upserting %d chunks to Qdrant...", len(chunks))
            upserted, failed = self.qdrant.upsert_chunks(chunks)
            # attempt to collect chunk ids, tolerant of dict or object chunks
            ids = []
            for c in chunks:
                try:
                    cid = getattr(c, "id", None)
                    if cid is None and isinstance(c, dict):
                        cid = c.get("id") or c.get("chunk_id")
                    if cid:
                        ids.append(cid)
                except Exception:
                    continue
            result.chunk_ids = ids
            logger.info("AudioIngestor: upsert finished (success=%s, failed=%d).", upserted, len(failed))
        except Exception as e:
            logger.exception("AudioIngestor: Qdrant upsert failed: %s", e)
            result.errors.append(f"qdrant_upsert_failed:{e}")

        logger.info(
            "AudioIngestor done: %s chunks=%d errors=%s (%.2fs)",
            filename, result.chunks_created, result.errors, time.time() - start_t,
        )
        return result


# Quick CLI test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--session-id", help="Optional session id", default=None)
    args = parser.parse_args()

    ing = AudioIngestor(session_id=args.session_id)
    res = ing.ingest_audio(args.file)
    print("Result:", getattr(res, "to_dict", lambda: res.__dict__)())
